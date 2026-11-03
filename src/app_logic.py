"""
Logic behind the Streamlit app, kept free of Streamlit so it can be unit tested.

- Sensor validation for the predictive-maintenance workflow (Track 1 model)
- The deterministic Track 2 quality rule, using the generator's own constants
- Safe loading of the saved model and report artifacts
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from scripts.adapt_ai4i_dataset import (
    NOMINAL_LENGTH_MM,
    NOMINAL_WEIGHT_G,
    TOLERANCE_LIMIT_MM,
    WEIGHT_LIMIT_G,
)
from src.pipeline import TELEMETRY_FEATURES
from src.predict import load_model

REPORTS_DIR = REPO_ROOT / "reports"
MODEL_PATH = REPO_ROOT / "models" / "defect_pipeline.joblib"
DATA_PATH = REPO_ROOT / "manufacturing_samples.csv"
REPORT_FILES = [
    "eda_sensor_distributions.png",
    "eda_failure_rules.png",
    "confusion_matrix.png",
    "classification_report.txt",
    "feature_importance.png",
    "feature_importance.csv",
    "metrics.json",
]

# Inputs outside these bounds are rejected as implausible or as likely typos. They are not
# failure limits; the model's training range is narrower and produces a warning instead.
SENSOR_SPECS = {
    "air_temperature_k": {"label": "Air temperature", "unit": "K", "min": 250.0, "max": 350.0},
    "process_temperature_k": {"label": "Process temperature", "unit": "K", "min": 250.0, "max": 400.0},
    "rotational_speed_rpm": {"label": "Rotational speed", "unit": "rpm", "min": 1.0, "max": 5000.0},
    "torque_nm": {"label": "Torque", "unit": "N·m", "min": 0.0, "max": 150.0},
    "tool_wear_min": {"label": "Tool wear", "unit": "min", "min": 0.0, "max": 500.0},
}

# Sanity bounds for part measurements, to catch typos. The quality rule itself is the constants above.
PART_SPECS = {
    "weight_g": {"label": "Weight", "unit": "g", "min": 1.0, "max": 1000.0},
    "length_mm": {"label": "Length", "unit": "mm", "min": 50.0, "max": 150.0},
    "width_mm": {"label": "Width", "unit": "mm", "min": 25.0, "max": 75.0},
    "height_mm": {"label": "Height", "unit": "mm", "min": 10.0, "max": 40.0},
    "tolerance_deviation_mm": {"label": "Tolerance deviation", "unit": "mm", "min": -5.0, "max": 5.0},
}
NOMINAL_WIDTH_MM = 50.0
NOMINAL_HEIGHT_MM = 25.0


def _parse_number(value: Any) -> Tuple[Optional[float], Optional[str]]:
    """Return (number, None), or (None, reason) for missing or non-numeric input."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, "is missing"
    if isinstance(value, bool):
        return None, "must be a number"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, "must be a number"
    if not math.isfinite(number):
        return None, "must be a finite number"
    return number, None


def training_ranges(data_path: Path = DATA_PATH) -> Dict[str, Tuple[float, float]]:
    """Min and max of each sensor in the training file, used to warn about extrapolation."""
    df = pd.read_csv(data_path, usecols=TELEMETRY_FEATURES)
    return {col: (float(df[col].min()), float(df[col].max())) for col in TELEMETRY_FEATURES}


def validate_sensor_inputs(raw: Dict[str, Any], ranges: Optional[Dict[str, Tuple[float, float]]] = None):
    """
    Check the five sensor readings. Returns (values, errors, warnings).
    Errors block prediction; warnings do not. Invalid input is never replaced.
    """
    values, errors, warnings = {}, [], []
    for key, spec in SENSOR_SPECS.items():
        name = f"{spec['label']} ({spec['unit']})"
        number, problem = _parse_number(raw.get(key))
        if problem:
            errors.append(f"{name} {problem}.")
            continue
        if not spec["min"] <= number <= spec["max"]:
            errors.append(f"{name} must be between {spec['min']:g} and {spec['max']:g}; got {number:g}.")
            continue
        values[key] = number
        if ranges and key in ranges:
            low, high = ranges[key]
            if not low <= number <= high:
                warnings.append(
                    f"{name} of {number:g} is outside the training data range ({low:g} to {high:g}). "
                    "The model is extrapolating, so treat the result with extra caution."
                )
    if not errors and values["process_temperature_k"] < values["air_temperature_k"]:
        warnings.append("Process temperature is below air temperature, which never happens in the training data.")
    return values, errors, warnings


def inspect_part(measurements: Dict[str, Any], machine_failure: bool) -> Dict[str, Any]:
    """
    Apply the Track 2 label rule from scripts/adapt_ai4i_dataset.py:
    defective = machine failure OR |tolerance deviation| > 0.12 mm OR |weight - 250 g| > 7 g.

    Tolerance deviation is required. A missing weight is allowed and, exactly as in the
    generator, never triggers the weight rule. Length, width and height are not part of the rule.
    """
    errors, warnings, values = [], [], {}
    for key, spec in PART_SPECS.items():
        raw = measurements.get(key)
        number, problem = _parse_number(raw)
        if problem == "is missing" and key != "tolerance_deviation_mm":
            values[key] = None
            continue
        name = f"{spec['label']} ({spec['unit']})"
        if problem:
            errors.append(f"{name} {problem}.")
            continue
        if not spec["min"] <= number <= spec["max"]:
            errors.append(f"{name} must be between {spec['min']:g} and {spec['max']:g}; got {number:g}.")
            continue
        values[key] = number
    if errors:
        return {"errors": errors, "warnings": warnings, "values": values, "defective": None, "checks": []}

    tol, weight = values["tolerance_deviation_mm"], values["weight_g"]
    checks = [
        {
            "rule": "Machine failure during the run",
            "value": "Yes" if machine_failure else "No",
            "limit": "Any failure",
            "ratio": None,
            "triggered": bool(machine_failure),
        },
        {
            "rule": "|Tolerance deviation| > 0.12 mm",
            "value": f"{tol:+.4f} mm",
            "limit": f"±{TOLERANCE_LIMIT_MM:g} mm",
            "ratio": abs(tol) / TOLERANCE_LIMIT_MM,
            "triggered": abs(tol) > TOLERANCE_LIMIT_MM,
        },
        {
            "rule": "|Weight − 250 g| > 7 g",
            "value": "Missing" if weight is None else f"{weight:.2f} g",
            "limit": f"{NOMINAL_WEIGHT_G - WEIGHT_LIMIT_G:g} to {NOMINAL_WEIGHT_G + WEIGHT_LIMIT_G:g} g",
            "ratio": None if weight is None else abs(weight - NOMINAL_WEIGHT_G) / WEIGHT_LIMIT_G,
            "triggered": weight is not None and abs(weight - NOMINAL_WEIGHT_G) > WEIGHT_LIMIT_G,
        },
    ]
    if weight is None:
        warnings.append("Weight is missing, so the weight rule cannot fire. The dataset generator treats missing weights the same way.")

    expected_length = NOMINAL_LENGTH_MM + tol
    if values["length_mm"] is not None and abs(values["length_mm"] - expected_length) > 0.001:
        warnings.append(
            f"In this dataset, length is always 100 mm + tolerance deviation ({expected_length:.4f} mm here). "
            f"The entered length of {values['length_mm']:.4f} mm doesn't match, but length is not part of the rule."
        )
    return {
        "errors": [],
        "warnings": warnings,
        "values": values,
        "defective": any(check["triggered"] for check in checks),
        "checks": checks,
    }


def load_model_or_error(model_path: Path = MODEL_PATH):
    """Return (pipeline, None), or (None, message) if the saved model is missing or can't be loaded."""
    try:
        return load_model(str(model_path)), None
    except FileNotFoundError:
        return None, f"The saved model was not found at {model_path}. Run `python src/train.py` to create it."
    except Exception as exc:  # e.g. a model saved by another scikit-learn version
        return None, f"The saved model at {model_path} could not be loaded ({type(exc).__name__}: {exc})."


def missing_reports(reports_dir: Path = REPORTS_DIR) -> List[str]:
    """Report files that don't exist yet."""
    return [name for name in REPORT_FILES if not (Path(reports_dir) / name).exists()]


def load_metrics(reports_dir: Path = REPORTS_DIR) -> Optional[Dict[str, Any]]:
    """Read reports/metrics.json, or None if it is missing or unreadable."""
    try:
        return json.loads((Path(reports_dir) / "metrics.json").read_text())
    except (OSError, ValueError):
        return None
