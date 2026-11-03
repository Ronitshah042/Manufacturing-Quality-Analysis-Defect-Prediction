"""
Manufacturing Quality & Predictive Maintenance Inference Script
===============================================================
Loads the serialized Scikit-learn pipeline and predicts machine failure
risk from the UCI AI4I 2020 sensor columns (Track 1). Note: AI4I 2020 is a
synthetic benchmark published by UCI, not logs from a real factory.

Usage Examples:
---------------
1. Run built-in demonstration cases:
   python3 src/predict.py

2. Predict custom machine telemetry via CLI:
   python3 src/predict.py --air_temp 298.1 --process_temp 308.6 --rpm 1551 --torque 42.8 --tool_wear 120
   python3 src/predict.py --air_temp 303.5 --process_temp 313.2 --rpm 1380 --torque 68.5 --tool_wear 225
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import numpy as np
import pandas as pd

# The saved model refers to src.pipeline by name, so that import must work however this file is run.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import FAILURE_THRESHOLD, TELEMETRY_FEATURES, failure_probabilities, predict_failure

DEFAULT_MODEL_PATH = "models/defect_pipeline.joblib"

# Example sensor readings for the demo and the app presets. Results come from the model, never from here.
DEMO_SCENARIOS = [
    {"desc": "Nominal milling operation", "air": 298.2, "proc": 308.7, "rpm": 1500, "torque": 40.0, "wear": 40},
    {"desc": "Fresh cutting tool, moderate cut", "air": 299.1, "proc": 309.4, "rpm": 1620, "torque": 35.5, "wear": 12},
    {"desc": "Critical tool wear breakdown (TWF)", "air": 300.4, "proc": 310.8, "rpm": 1410, "torque": 52.0, "wear": 235},
    {"desc": "Overstrain cutting load (OSF)", "air": 301.2, "proc": 311.5, "rpm": 1280, "torque": 69.8, "wear": 195},
    {"desc": "Heat dissipation failure (HDF: ΔT < 8.6K, RPM < 1380)", "air": 302.5, "proc": 310.4, "rpm": 1340, "torque": 55.0, "wear": 180},
]


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load the trained Scikit-learn pipeline from disk.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fitted model not found at '{model_path}'. "
            "Please run 'python3 src/train.py' first to train and serialize the pipeline."
        )
    return joblib.load(path)


def predict_machine_failure(
    air_temperature_k: float,
    process_temperature_k: float,
    rotational_speed_rpm: float,
    torque_nm: float,
    tool_wear_min: float,
    pipeline=None,
    model_path: str = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Predict machine failure risk from the five AI4I 2020 sensor columns.

    Parameters
    ----------
    air_temperature_k : float
        Ambient air temperature in Kelvin.
    process_temperature_k : float
        Cutting process temperature in Kelvin.
    rotational_speed_rpm : float
        Spindle rotational speed in RPM.
    torque_nm : float
        Cutting torque in Newton-meters (N*m).
    tool_wear_min : float
        Cumulative tool wear in minutes.
    pipeline : sklearn.pipeline.Pipeline, optional
        Preloaded pipeline. If None, loaded from `model_path`.
    model_path : str, default='models/defect_pipeline.joblib'
        Path to the saved pipeline artifact.

    Returns
    -------
    dict
        Prediction results including boolean flag, status string, probability, and inputs.
    """
    if pipeline is None:
        pipeline = load_model(model_path)

    def _to_float(val):
        return np.nan if val is None else float(val)

    input_data = pd.DataFrame(
        [{
            "air_temperature_k": _to_float(air_temperature_k),
            "process_temperature_k": _to_float(process_temperature_k),
            "rotational_speed_rpm": _to_float(rotational_speed_rpm),
            "torque_nm": _to_float(torque_nm),
            "tool_wear_min": _to_float(tool_wear_min),
        }],
        columns=TELEMETRY_FEATURES,
    )

    is_failure = bool(predict_failure(pipeline, input_data)[0] == 1)

    prob_failure = None
    if hasattr(pipeline, "predict_proba") and 1 in list(pipeline.classes_):
        prob_failure = float(failure_probabilities(pipeline, input_data)[0])

    return {
        "failure": is_failure,
        "status": "MACHINE FAILURE RISK" if is_failure else "NORMAL OPERATION",
        "failure_probability": prob_failure,
        "input": {
            "air_temperature_k": air_temperature_k,
            "process_temperature_k": process_temperature_k,
            "rotational_speed_rpm": rotational_speed_rpm,
            "torque_nm": torque_nm,
            "tool_wear_min": tool_wear_min,
        },
    }


def predict_defect(*args, **kwargs):
    """
    Deprecated alias kept so old imports do not break.

    Accepts the same telemetry keyword arguments as predict_machine_failure().
    The old (weight_g, length_mm) signature is no longer supported: the saved
    model is trained on sensor columns, and there is no valid way to convert
    part dimensions into sensor readings.
    """
    if args or "weight_g" in kwargs or "length_mm" in kwargs:
        raise TypeError(
            "predict_defect(weight_g, length_mm) is no longer supported. "
            "The model predicts machine failure from sensor telemetry; call "
            "predict_machine_failure(air_temperature_k=..., process_temperature_k=..., "
            "rotational_speed_rpm=..., torque_nm=..., tool_wear_min=...) instead."
        )
    return predict_machine_failure(**kwargs)


def run_demo_predictions(pipeline=None):
    """
    Run demonstration predictions across realistic industrial operating scenarios.
    """
    print("\n" + "=" * 76)
    print(" PREDICTIVE MAINTENANCE INSPECTOR: INDUSTRIAL SCENARIOS RUN")
    print("=" * 76)

    if pipeline is None:
        pipeline = load_model(DEFAULT_MODEL_PATH)

    results = []
    for case in DEMO_SCENARIOS:
        res = predict_machine_failure(
            air_temperature_k=case["air"],
            process_temperature_k=case["proc"],
            rotational_speed_rpm=case["rpm"],
            torque_nm=case["torque"],
            tool_wear_min=case["wear"],
            pipeline=pipeline,
        )
        prob_str = f"{res['failure_probability']*100:.1f}%" if res["failure_probability"] is not None else "N/A"
        results.append({
            "Operating Scenario": case["desc"],
            "RPM": case["rpm"],
            "Torque": f"{case['torque']:.1f} Nm",
            "Tool Wear": f"{case['wear']} min",
            "Status": res["status"],
            "Failure Prob": prob_str,
        })

    demo_df = pd.DataFrame(results)
    print(demo_df.to_string(index=False))
    print("=" * 76)
    print("Model trained on the UCI AI4I 2020 synthetic predictive-maintenance benchmark.")
    print("Run with --help to see how to predict custom sensor readings via CLI.")


def main():
    parser = argparse.ArgumentParser(
        description="Predict machine failure risk from industrial sensor telemetry."
    )
    parser.add_argument("--air_temp", type=float, default=None, help="Air temperature in Kelvin (e.g. 298.5)")
    parser.add_argument("--process_temp", type=float, default=None, help="Process temperature in Kelvin (e.g. 308.7)")
    parser.add_argument("--rpm", type=float, default=None, help="Rotational speed in RPM (e.g. 1500)")
    parser.add_argument("--torque", type=float, default=None, help="Torque in Nm (e.g. 40.5)")
    parser.add_argument("--tool_wear", type=float, default=None, help="Tool wear in minutes (e.g. 120)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help=f"Path to saved model (default: {DEFAULT_MODEL_PATH})")

    args = parser.parse_args()

    # If no arguments provided, run demo mode
    if all(v is None for v in [args.air_temp, args.process_temp, args.rpm, args.torque, args.tool_wear]):
        run_demo_predictions()
        return

    # Default fallback values for omitted arguments.
    # Use `is None` (not `or`) so that a valid 0 (e.g. --tool_wear 0) is kept.
    def _default(value, fallback):
        return fallback if value is None else value

    air = _default(args.air_temp, 300.0)
    proc = _default(args.process_temp, 310.0)
    rpm = _default(args.rpm, 1500.0)
    torque = _default(args.torque, 40.0)
    wear = _default(args.tool_wear, 100.0)

    res = predict_machine_failure(
        air_temperature_k=air,
        process_temperature_k=proc,
        rotational_speed_rpm=rpm,
        torque_nm=torque,
        tool_wear_min=wear,
        model_path=args.model,
    )

    print("\n--- Predictive Maintenance Inspection Result ---")
    print(f"Air Temperature:     {air:.1f} K")
    print(f"Process Temperature: {proc:.1f} K")
    print(f"Rotational Speed:    {rpm:.0f} RPM")
    print(f"Spindle Torque:      {torque:.1f} Nm")
    print(f"Tool Wear Duration:  {wear:.0f} min")
    print(f"Operation Status:    {res['status']}")
    print(f"Failure Flag:        {res['failure']}")
    if res["failure_probability"] is not None:
        print(f"Failure Probability: {res['failure_probability'] * 100:.1f}%")
    print(f"Flagged At or Above: {FAILURE_THRESHOLD * 100:.0f}%")
    print("------------------------------------------------\n")


if __name__ == "__main__":
    main()
