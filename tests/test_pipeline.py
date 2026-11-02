"""
Unit Tests for Track 1 Telemetry Pipeline & Predictive Maintenance
==================================================================
Tests:
1. Pipeline composition (imputer, scaler, classifier).
2. Missing value imputation behavior with SimpleImputer(strategy='median').
3. Data loader telemetry feature extraction and deduplication.
4. Model serialization to disk and reload consistency.
5. Machine failure prediction function behavior.
6. Train and test accuracy calculation in evaluate_model.
7. Backward compatibility for predict_defect.
8. Tracked model loads and predicts under the pinned dependency versions.
9. Dataset generator reproduces the tracked CSV offline and rejects changed raw data.
"""

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.pipeline import (
    TELEMETRY_FEATURES,
    TELEMETRY_TARGET,
    add_interaction_features,
    create_pipeline,
    load_and_clean_data,
    load_unseen_ai4i,
)
from src.predict import load_model, predict_defect, predict_machine_failure
from src.train import cross_validate_failure_metrics, evaluate_model
from scripts.adapt_ai4i_dataset import (
    RAW_CSV_PATH,
    adapt_to_manufacturing_schema,
    fetch_raw_ai4i,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestManufacturingPipeline(unittest.TestCase):

    def setUp(self):
        # Small sample dataset using the Track 1 telemetry schema
        self.sample_telemetry = pd.DataFrame({
            "product_id": ["PART-0001", "PART-0002", "PART-0003", "PART-0004"],
            "air_temperature_k": [298.1, 298.2, 303.5, np.nan],
            "process_temperature_k": [308.6, 308.7, 313.2, 308.5],
            "rotational_speed_rpm": [1551, 1408, 1380, 1500],
            "torque_nm": [42.8, 46.3, 68.5, 40.0],
            "tool_wear_min": [0, 3, 225, 50],
            "source_machine_failure": [0, 0, 1, 0],
            "defective": [False, False, True, False],
            "weight_g": [250.0, 251.0, 260.0, 240.0],
            "length_mm": [100.0, 100.05, 100.4, 99.6],
        })

    def test_pipeline_structure(self):
        """Pipeline should contain 'imputer', 'scaler', and 'classifier'."""
        pipe = create_pipeline()
        self.assertIsInstance(pipe, Pipeline)
        self.assertIn("imputer", pipe.named_steps)
        self.assertIn("scaler", pipe.named_steps)
        self.assertIn("classifier", pipe.named_steps)
        self.assertIsInstance(pipe.named_steps["imputer"], SimpleImputer)
        self.assertIsInstance(pipe.named_steps["scaler"], StandardScaler)
        self.assertIsInstance(pipe.named_steps["classifier"], DecisionTreeClassifier)

    def test_missing_value_imputation(self):
        """Imputer must fill missing telemetry sensor values with median."""
        X_train = pd.DataFrame({
            "air_temperature_k": [298.0, 300.0, 302.0],  # Median = 300.0
            "process_temperature_k": [308.0, 310.0, 312.0],
            "rotational_speed_rpm": [1400.0, 1500.0, 1600.0],
            "torque_nm": [35.0, 40.0, 45.0],
            "tool_wear_min": [50.0, 100.0, 150.0],
        })
        y_train = pd.Series([0, 0, 1])

        pipe = create_pipeline(random_state=42)
        pipe.fit(X_train, y_train)

        # Check imputer learned column medians
        imputer = pipe.named_steps["imputer"]
        expected_medians = [300.0, 310.0, 1500.0, 40.0, 100.0]
        np.testing.assert_array_equal(imputer.statistics_, expected_medians)

        # 1. Test every individual feature missing in isolation
        for feat in TELEMETRY_FEATURES:
            row = {f: 300.0 if "temp" in f else (1500.0 if "rpm" in f else (40.0 if "torque" in f else 100.0)) for f in TELEMETRY_FEATURES}
            row[feat] = np.nan
            X_single_missing = pd.DataFrame([row], columns=TELEMETRY_FEATURES)
            pred = pipe.predict(X_single_missing)
            self.assertEqual(len(pred), 1)

        # 2. Test missing-value handling after serialization and reloading
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            joblib.dump(pipe, tmp_path)
            reloaded = joblib.load(tmp_path)
            X_all_nan = pd.DataFrame([{f: np.nan for f in TELEMETRY_FEATURES}], columns=TELEMETRY_FEATURES)
            pred_orig = pipe.predict(X_all_nan)
            pred_reloaded = reloaded.predict(X_all_nan)
            np.testing.assert_array_equal(pred_orig, pred_reloaded)
            # Replaced with median matches prediction on median row directly
            X_median = pd.DataFrame([dict(zip(TELEMETRY_FEATURES, expected_medians))])
            pred_median = pipe.predict(X_median)
            self.assertEqual(pred_orig[0], pred_median[0])
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_data_loader_telemetry_features_and_deduplication(self):
        """DataLoader must extract TELEMETRY_FEATURES and remove duplicates."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
            self.sample_telemetry.to_csv(tmp_path, index=False)
            # Append duplicate
            with open(tmp_path, "a") as f:
                f.write("PART-0001,298.1,308.6,1551,42.8,0,0,False,250.0,100.0\n")

        try:
            X, y = load_and_clean_data(tmp_path, feature_set="telemetry")
            # 4 unique rows
            self.assertEqual(len(X), 4)
            self.assertEqual(len(y), 4)
            self.assertListEqual(list(X.columns), TELEMETRY_FEATURES)
            self.assertEqual(y.name, TELEMETRY_TARGET)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_pipeline_serialization_and_reload(self):
        """Pipeline serialized with joblib must reload and predict identically."""
        X = self.sample_telemetry[TELEMETRY_FEATURES]
        y = self.sample_telemetry[TELEMETRY_TARGET]

        pipe = create_pipeline(random_state=42)
        pipe.fit(X, y)
        orig_preds = pipe.predict(X)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_model_path = tmp.name

        try:
            joblib.dump(pipe, tmp_model_path)
            reloaded_pipe = joblib.load(tmp_model_path)
            reloaded_preds = reloaded_pipe.predict(X)
            np.testing.assert_array_equal(orig_preds, reloaded_preds)
        finally:
            Path(tmp_model_path).unlink(missing_ok=True)

    def test_predict_machine_failure_function(self):
        """predict_machine_failure should output structured status and probability."""
        X = self.sample_telemetry[TELEMETRY_FEATURES]
        y = self.sample_telemetry[TELEMETRY_TARGET]

        pipe = create_pipeline(random_state=42)
        pipe.fit(X, y)

        # Nominal operating machine
        res_normal = predict_machine_failure(
            air_temperature_k=298.1,
            process_temperature_k=308.6,
            rotational_speed_rpm=1551,
            torque_nm=42.8,
            tool_wear_min=0,
            pipeline=pipe,
        )
        self.assertIn("failure", res_normal)
        self.assertIn("status", res_normal)
        self.assertFalse(res_normal["failure"])
        self.assertEqual(res_normal["status"], "NORMAL OPERATION")

        # Critical breakdown machine
        res_fail = predict_machine_failure(
            air_temperature_k=303.5,
            process_temperature_k=313.2,
            rotational_speed_rpm=1380,
            torque_nm=68.5,
            tool_wear_min=225,
            pipeline=pipe,
        )
        self.assertTrue(res_fail["failure"])
        # Machine with missing sensor value (None handled via imputer)
        res_missing = predict_machine_failure(
            air_temperature_k=None,
            process_temperature_k=308.6,
            rotational_speed_rpm=1551,
            torque_nm=42.8,
            tool_wear_min=None,
            pipeline=pipe,
        )
        self.assertIn("failure", res_missing)
        self.assertIn("status", res_missing)

    def test_evaluate_model_training_and_test_accuracy(self):
        """evaluate_model must accurately compute both training and test accuracy."""
        X_train = pd.DataFrame({
            "air_temperature_k": [298.0, 298.2, 303.0, 304.0],
            "process_temperature_k": [308.0, 308.5, 313.0, 314.0],
            "rotational_speed_rpm": [1500, 1520, 1350, 1300],
            "torque_nm": [40.0, 42.0, 65.0, 70.0],
            "tool_wear_min": [10, 20, 220, 240],
        })
        y_train = pd.Series([0, 0, 1, 1])

        X_test = pd.DataFrame({
            "air_temperature_k": [298.1, 303.5],
            "process_temperature_k": [308.2, 313.5],
            "rotational_speed_rpm": [1510, 1340],
            "torque_nm": [41.0, 68.0],
            "tool_wear_min": [15, 230],
        })
        y_test = pd.Series([0, 1])

        pipe = create_pipeline(random_state=42)
        pipe.fit(X_train, y_train)

        metrics = evaluate_model(
            "Telemetry Model",
            pipe,
            X_test,
            y_test,
            X_train=X_train,
            y_train=y_train,
        )

        self.assertIn("train_accuracy", metrics)
        self.assertIn("test_accuracy", metrics)
        self.assertEqual(metrics["train_accuracy"], 1.0)
        self.assertEqual(metrics["test_accuracy"], 1.0)

    def test_backward_compatible_predict_defect(self):
        """predict_defect forwards telemetry kwargs and rejects the old weight/length signature."""
        X = self.sample_telemetry[TELEMETRY_FEATURES]
        y = self.sample_telemetry[TELEMETRY_TARGET]

        pipe = create_pipeline(random_state=42)
        pipe.fit(X, y)

        # Telemetry kwargs are forwarded unchanged to predict_machine_failure
        res = predict_defect(
            air_temperature_k=298.1,
            process_temperature_k=308.6,
            rotational_speed_rpm=1551,
            torque_nm=42.8,
            tool_wear_min=0,
            pipeline=pipe,
        )
        self.assertIn("failure", res)
        self.assertIn("status", res)

        # Old dimensional signature must fail loudly, not invent sensor values
        with self.assertRaises(TypeError):
            predict_defect(weight_g=250.0, length_mm=100.0, pipeline=pipe)
        with self.assertRaises(TypeError):
            predict_defect(250.0, 100.0)

    def test_cross_validation_scores_every_fold(self):
        """Cross-validation must score each fold, and handle missing values inside the pipeline."""
        X = pd.DataFrame({
            "a": [float(i) for i in range(10)] + [float(100 + i) for i in range(10)],
            "b": [np.nan] + [1.0] * 19,
        })
        y = pd.Series([0] * 10 + [1] * 10)

        cv = cross_validate_failure_metrics(create_pipeline(random_state=42), X, y, n_splits=5, n_repeats=1)

        self.assertEqual(len(cv["recall"]), 5)
        self.assertEqual(float(cv["recall"].mean()), 1.0)
        self.assertEqual(float(cv["precision"].mean()), 1.0)
        self.assertEqual(float(cv["f1"].mean()), 1.0)

    def test_interaction_features_match_uci_formulas(self):
        """Derived features must use the documented formulas and keep the sensor columns."""
        X = pd.DataFrame([{
            "air_temperature_k": 300.0,
            "process_temperature_k": 308.5,
            "rotational_speed_rpm": 1500.0,
            "torque_nm": 40.0,
            "tool_wear_min": 200.0,
        }])
        result = add_interaction_features(X).iloc[0]
        self.assertAlmostEqual(result["temperature_difference_k"], 8.5)
        self.assertAlmostEqual(result["power_w"], 40.0 * 1500.0 * 2 * np.pi / 60)
        self.assertAlmostEqual(result["wear_torque_min_nm"], 8000.0)
        self.assertEqual(result["torque_nm"], 40.0)

    def test_interaction_pipeline_handles_missing_sensor(self):
        """A missing sensor must not crash the interaction pipeline after a save and reload."""
        X = self.sample_telemetry[TELEMETRY_FEATURES]
        y = self.sample_telemetry[TELEMETRY_TARGET]
        pipe = create_pipeline(random_state=42, include_interactions=True).fit(X, y)
        self.assertEqual(pipe.named_steps["imputer"].n_features_in_, 8)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump(pipe, path)
            reloaded = joblib.load(path)
        res = predict_machine_failure(None, 308.6, 1551, None, 0, pipeline=reloaded)
        self.assertIn(res["status"], ("NORMAL OPERATION", "MACHINE FAILURE RISK"))

    def test_decision_threshold_is_applied(self):
        """A probability counts as a failure only at or above the 0.4 threshold."""
        class FixedProbability:
            classes_ = np.array([0, 1])

            def __init__(self, probability):
                self.probability = probability

            def predict_proba(self, X):
                return np.array([[1 - self.probability, self.probability]])

        for probability, expected in ((0.39, False), (0.40, True), (0.45, True)):
            res = predict_machine_failure(300, 310, 1500, 40, 100, pipeline=FixedProbability(probability))
            self.assertEqual(res["failure"], expected)
            self.assertAlmostEqual(res["failure_probability"], probability)

    def test_unseen_rows_exclude_training_sample(self):
        """The unseen check must use only raw UCI rows outside the training sample."""
        sample = REPO_ROOT / "manufacturing_samples.csv"
        X, y = load_unseen_ai4i(str(sample), str(RAW_CSV_PATH))
        self.assertEqual(len(X), 9000)
        self.assertEqual(int(y.sum()), 189)
        self.assertListEqual(list(X.columns), TELEMETRY_FEATURES)

        with tempfile.TemporaryDirectory() as tmp:
            bad_sample = Path(tmp) / "sample.csv"
            bad_sample.write_text("product_id\nPART-99999\n")
            with self.assertRaises(ValueError):
                load_unseen_ai4i(str(bad_sample), str(RAW_CSV_PATH))

    def test_tracked_model_loads_and_predicts(self):
        """The committed model must work under the pinned versions, on the Track 1 features."""
        pipe = load_model(str(REPO_ROOT / "models" / "defect_pipeline.joblib"))
        self.assertListEqual(list(pipe.feature_names_in_), TELEMETRY_FEATURES)
        self.assertIn("features", pipe.named_steps)

        X, y = load_and_clean_data(str(REPO_ROOT / "manufacturing_samples.csv"))
        self.assertEqual(len(pipe.predict(X)), len(X))

        res = predict_machine_failure(298.2, 308.7, 1500, 40.0, 40, pipeline=pipe)
        self.assertFalse(res["failure"])

    def test_generator_reproduces_tracked_dataset_offline(self):
        """The local raw copy alone must regenerate the tracked CSV byte for byte."""
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network used")):
            raw = fetch_raw_ai4i()
        generated = adapt_to_manufacturing_schema(raw, total_samples=1000).to_csv(index=False)
        self.assertEqual(generated.encode(), (REPO_ROOT / "manufacturing_samples.csv").read_bytes())

    def test_fetch_raw_rejects_changed_local_copy(self):
        """A local raw file with the wrong checksum must fail loudly."""
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "ai4i2020.csv"
            bad.write_bytes(RAW_CSV_PATH.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                fetch_raw_ai4i(raw_path=bad)

    def test_fetch_raw_download_saves_only_verified_data(self):
        """A missing copy is downloaded and saved if valid; a changed source is not saved."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "raw" / "ai4i2020.csv"

            with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(b"changed,data\n")):
                with self.assertRaises(ValueError):
                    fetch_raw_ai4i(raw_path=target)
            self.assertFalse(target.exists())

            with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(RAW_CSV_PATH.read_bytes())):
                self.assertEqual(len(fetch_raw_ai4i(raw_path=target)), 10000)
            self.assertEqual(target.read_bytes(), RAW_CSV_PATH.read_bytes())


if __name__ == "__main__":
    unittest.main()
