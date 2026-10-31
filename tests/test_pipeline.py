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
"""

import tempfile
import unittest
from pathlib import Path
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
    create_pipeline,
    load_and_clean_data,
)
from src.predict import predict_defect, predict_machine_failure
from src.train import evaluate_model


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


if __name__ == "__main__":
    unittest.main()
