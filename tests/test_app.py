"""
Tests for the Streamlit app: validation and rule logic, safe loading, and each page run headlessly.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.app_logic import (
    REPORT_FILES,
    inspect_part,
    load_metrics,
    load_model_or_error,
    missing_reports,
    training_ranges,
    validate_sensor_inputs,
)
from src.predict import load_model, predict_machine_failure

REPO_ROOT = Path(__file__).resolve().parent.parent
APP = str(REPO_ROOT / "app.py")
VALID_SENSORS = {
    "air_temperature_k": 301.2,
    "process_temperature_k": 311.5,
    "rotational_speed_rpm": 1280,
    "torque_nm": 69.8,
    "tool_wear_min": 195,
}
PART_FIELDS = ["weight_g", "length_mm", "width_mm", "height_mm", "tolerance_deviation_mm"]


def run_page(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    return at.switch_page(page).run()


class TestSensorValidation(unittest.TestCase):

    def test_valid_inputs_pass_unchanged(self):
        values, errors, warnings = validate_sensor_inputs(VALID_SENSORS, training_ranges())
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])
        self.assertEqual(values, {k: float(v) for k, v in VALID_SENSORS.items()})

    def test_missing_non_numeric_and_out_of_bounds_are_rejected(self):
        raw = dict(VALID_SENSORS, process_temperature_k="abc", rotational_speed_rpm=None,
                   torque_nm=-5, tool_wear_min=float("nan"))
        values, errors, _ = validate_sensor_inputs(raw)
        self.assertEqual(len(errors), 4)
        self.assertEqual(set(values), {"air_temperature_k"})
        joined = " ".join(errors)
        for text in ("must be a number", "is missing", "between 0 and 150", "finite"):
            self.assertIn(text, joined)

    def test_out_of_training_range_warns_without_blocking(self):
        values, errors, warnings = validate_sensor_inputs(dict(VALID_SENSORS, rotational_speed_rpm=3500), training_ranges())
        self.assertEqual(errors, [])
        self.assertEqual(values["rotational_speed_rpm"], 3500.0)
        self.assertTrue(any("extrapolating" in w for w in warnings))


class TestPartInspection(unittest.TestCase):

    def test_rule_matches_every_generated_label(self):
        """The app's rule must reproduce the dataset's defective column exactly."""
        df = pd.read_csv(REPO_ROOT / "manufacturing_samples.csv")
        for row in df.itertuples():
            measurements = {k: None if pd.isna(getattr(row, k)) else getattr(row, k) for k in PART_FIELDS}
            result = inspect_part(measurements, machine_failure=bool(row.source_machine_failure))
            self.assertEqual(result["errors"], [], row.product_id)
            self.assertEqual(result["defective"], bool(row.defective), row.product_id)

    def test_limits_are_strict(self):
        """Exactly at a limit passes, just past it fails, matching the generator's `>` comparisons."""
        base = {"weight_g": 250.0, "length_mm": 100.0, "width_mm": 50.0, "height_mm": 25.0}
        self.assertFalse(inspect_part(dict(base, tolerance_deviation_mm=0.12), False)["defective"])
        self.assertTrue(inspect_part(dict(base, tolerance_deviation_mm=-0.1201), False)["defective"])
        self.assertFalse(inspect_part(dict(base, weight_g=257.0, tolerance_deviation_mm=0.0), False)["defective"])
        self.assertTrue(inspect_part(dict(base, weight_g=242.99, tolerance_deviation_mm=0.0), False)["defective"])
        self.assertTrue(inspect_part(dict(base, tolerance_deviation_mm=0.0), True)["defective"])

    def test_missing_and_invalid_measurements(self):
        missing_tol = inspect_part({"weight_g": 250.0}, False)
        self.assertIsNone(missing_tol["defective"])
        self.assertIn("Tolerance deviation (mm) is missing.", missing_tol["errors"])

        missing_weight = inspect_part({"tolerance_deviation_mm": 0.0}, False)
        self.assertFalse(missing_weight["defective"])
        self.assertTrue(any("Weight is missing" in w for w in missing_weight["warnings"]))

        bad = inspect_part({"tolerance_deviation_mm": "x", "weight_g": -3}, False)
        self.assertEqual(len(bad["errors"]), 2)

    def test_inconsistent_length_warns(self):
        result = inspect_part({"tolerance_deviation_mm": 0.05, "length_mm": 100.2}, False)
        self.assertTrue(any("length is always 100 mm" in w for w in result["warnings"]))


class TestArtifactLoading(unittest.TestCase):

    def test_missing_model_returns_message(self):
        model, error = load_model_or_error(Path("/nonexistent/model.joblib"))
        self.assertIsNone(model)
        self.assertIn("not found", error)

    def test_corrupt_model_returns_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "model.joblib"
            bad.write_text("not a model")
            model, error = load_model_or_error(bad)
        self.assertIsNone(model)
        self.assertIn("could not be loaded", error)

    def test_missing_reports_are_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(missing_reports(Path(tmp)), REPORT_FILES)
            self.assertIsNone(load_metrics(Path(tmp)))
        self.assertEqual(missing_reports(), [])


class TestAppPages(unittest.TestCase):

    def test_every_page_runs(self):
        for page in ["app_pages/overview.py", "app_pages/maintenance.py", "app_pages/quality.py",
                     "app_pages/analytics.py", "app_pages/about.py"]:
            at = run_page(page)
            self.assertEqual(len(at.exception), 0, page)
            self.assertTrue(any("synthetic data" in w.value for w in at.warning), page)

    def test_maintenance_matches_prediction_function(self):
        at = run_page("app_pages/maintenance.py")
        for widget, value in zip(at.number_input, VALID_SENSORS.values()):
            widget.set_value(value)
        at.button[0].click().run()
        self.assertEqual(len(at.exception), 0)

        expected = predict_machine_failure(**VALID_SENSORS, pipeline=load_model(str(REPO_ROOT / "models" / "defect_pipeline.joblib")))
        self.assertEqual(at.metric[0].value, f"{expected['failure_probability']:.1%}")
        self.assertTrue(any("Failure predicted" in w.value for w in at.warning) == expected["failure"])

    def test_maintenance_rejects_missing_input(self):
        at = run_page("app_pages/maintenance.py")
        at.button[0].click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("Please fix these inputs" in e.value for e in at.error))
        self.assertEqual(len(at.metric), 0)

    def test_quality_inspection_flags_tolerance(self):
        at = run_page("app_pages/quality.py")
        values = {"weight_g": 250.0, "length_mm": 100.2, "width_mm": 50.1, "height_mm": 25.05, "tolerance_deviation_mm": 0.2}
        for widget, key in zip(at.number_input, PART_FIELDS):
            widget.set_value(values[key])
        at.button[0].click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("Defective (simulated)" in e.value and "Tolerance" in e.value for e in at.error))

    def test_quality_rejects_missing_tolerance(self):
        at = run_page("app_pages/quality.py")
        at.button[0].click().run()
        self.assertTrue(any("Tolerance deviation (mm) is missing." in e.value for e in at.error))

    def test_app_never_trains(self):
        """No page may fit a model: the app must only load the saved pipeline."""
        with mock.patch("sklearn.ensemble.RandomForestClassifier.fit", side_effect=AssertionError("app trained a model")), \
             mock.patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("app trained a pipeline")):
            for page in ["app_pages/maintenance.py", "app_pages/analytics.py"]:
                at = run_page(page)
                if page.endswith("maintenance.py"):
                    for widget, value in zip(at.number_input, VALID_SENSORS.values()):
                        widget.set_value(value)
                    at.button[0].click().run()
                self.assertEqual(len(at.exception), 0, page)


if __name__ == "__main__":
    unittest.main()
