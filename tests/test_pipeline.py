"""
Unit Tests for Manufacturing Defect Prediction Pipeline
======================================================
Tests:
1. Pipeline composition and step naming.
2. Missing value imputation behavior with SimpleImputer(strategy='median').
3. Deduplication and feature selection in data loader.
4. Model training, fitting, and prediction.
5. Model serialization to disk and reload consistency.
6. Single-item prediction function behavior (including missing inputs).
"""

import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.pipeline import FEATURE_NAMES, create_pipeline, load_and_clean_data
from src.predict import predict_defect


class TestManufacturingPipeline(unittest.TestCase):

    def setUp(self):
        # Small synthetic dataset mimicking manufacturing dimensions
        self.sample_data = pd.DataFrame({
            "product_id": ["PART-A", "PART-B", "PART-C", "PART-D"],
            "weight_g": [250.0, np.nan, 260.0, 240.0],
            "length_mm": [100.0, 100.05, 100.4, 99.6],
            "width_mm": [50.0, 50.0, 50.2, 49.8],
            "height_mm": [25.0, 25.0, 25.1, 24.9],
            "tolerance_deviation_mm": [0.0, 0.05, 0.4, -0.4],
            "defective": [False, False, True, True],
        })

    def test_pipeline_structure(self):
        """Pipeline should contain 'imputer' (SimpleImputer) and 'classifier'."""
        pipe = create_pipeline()
        self.assertIsInstance(pipe, Pipeline)
        self.assertIn("imputer", pipe.named_steps)
        self.assertIn("classifier", pipe.named_steps)
        self.assertIsInstance(pipe.named_steps["imputer"], SimpleImputer)
        self.assertEqual(pipe.named_steps["imputer"].strategy, "median")
        self.assertIsInstance(pipe.named_steps["classifier"], DecisionTreeClassifier)

    def test_missing_value_imputation(self):
        """Imputer must fill missing weight values with the median of training data."""
        X_train = pd.DataFrame({
            "weight_g": [248.0, 250.0, 252.0],  # Median is 250.0
            "length_mm": [100.0, 100.0, 100.0],
        })
        y_train = pd.Series([False, False, False])

        pipe = create_pipeline()
        pipe.fit(X_train, y_train)

        # Imputer statistics check
        imputer = pipe.named_steps["imputer"]
        self.assertEqual(imputer.statistics_[0], 250.0)

        # Test prediction on a sample containing NaN for weight_g
        X_test = pd.DataFrame({
            "weight_g": [np.nan],
            "length_mm": [100.0],
        })
        pred = pipe.predict(X_test)
        self.assertEqual(len(pred), 1)

    def test_data_loader_features_and_deduplication(self):
        """DataLoader must remove duplicates and exclude product_id & tolerance_deviation_mm."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
            # Write with one duplicate row
            self.sample_data.to_csv(tmp_path, index=False)
            with open(tmp_path, "a") as f:
                # Add duplicate of PART-A
                f.write("PART-A,250.0,100.0,50.0,25.0,0.0,False\n")

        try:
            X, y = load_and_clean_data(tmp_path)
            # Should have deduplicated: 4 unique rows
            self.assertEqual(len(X), 4)
            self.assertEqual(len(y), 4)

            # Features should only be weight_g and length_mm
            self.assertListEqual(list(X.columns), FEATURE_NAMES)
            self.assertNotIn("product_id", X.columns)
            self.assertNotIn("tolerance_deviation_mm", X.columns)
            self.assertNotIn("width_mm", X.columns)
            self.assertNotIn("height_mm", X.columns)

            # Target type should be boolean
            self.assertEqual(y.dtype, bool)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_pipeline_serialization_and_reload(self):
        """Fitted pipeline saved with joblib should reload and predict identically."""
        X = self.sample_data[FEATURE_NAMES]
        y = self.sample_data["defective"].astype(bool)

        pipe = create_pipeline(random_state=42)
        pipe.fit(X, y)
        original_preds = pipe.predict(X)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_model_path = tmp.name

        try:
            joblib.dump(pipe, tmp_model_path)
            reloaded_pipe = joblib.load(tmp_model_path)
            reloaded_preds = reloaded_pipe.predict(X)
            np.testing.assert_array_equal(original_preds, reloaded_preds)
        finally:
            Path(tmp_model_path).unlink(missing_ok=True)

    def test_predict_defect_function(self):
        """predict_defect helper function should return correct structure and handle missing weight."""
        X = self.sample_data[FEATURE_NAMES]
        y = self.sample_data["defective"].astype(bool)

        pipe = create_pipeline(random_state=42)
        pipe.fit(X, y)

        # 1. Normal compliant part
        result_pass = predict_defect(weight_g=250.0, length_mm=100.0, pipeline=pipe)
        self.assertIn("defective", result_pass)
        self.assertIn("status", result_pass)
        self.assertIsInstance(result_pass["defective"], bool)

        # 2. Part with missing weight (None)
        result_missing_weight = predict_defect(weight_g=None, length_mm=100.0, pipeline=pipe)
        self.assertFalse(result_missing_weight["defective"])
        self.assertEqual(result_missing_weight["status"], "PASSED QA")

    def test_predict_defect_edge_cases(self):
        """Test prediction output labels for boundary, defect, and nominal cases."""
        X, y = load_and_clean_data("manufacturing_samples.csv")
        pipe = create_pipeline(random_state=42)
        pipe.fit(X, y)

        # Compliant nominal part
        nominal = predict_defect(250.2, 100.04, pipeline=pipe)
        self.assertFalse(nominal["defective"])
        self.assertEqual(nominal["status"], "PASSED QA")

        # Oversized defective part
        oversized = predict_defect(262.3, 100.41, pipeline=pipe)
        self.assertTrue(oversized["defective"])
        self.assertEqual(oversized["status"], "DEFECTIVE")

        # Undersized defective part
        undersized = predict_defect(239.1, 99.68, pipeline=pipe)
        self.assertTrue(undersized["defective"])
        self.assertEqual(undersized["status"], "DEFECTIVE")

        # Missing sensor weight with compliant length
        missing_w = predict_defect(None, 100.03, pipeline=pipe)
        self.assertFalse(missing_w["defective"])
        self.assertEqual(missing_w["status"], "PASSED QA")


if __name__ == "__main__":
    unittest.main()

