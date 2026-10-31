"""
Manufacturing Defect Prediction Script
======================================
Loads the serialized Scikit-learn pipeline (median imputer + model)
and predicts whether a manufactured part passes or fails quality inspection
based on weight (g) and length (mm).

Usage Examples:
---------------
1. Run built-in demonstration cases:
   python3 src/predict.py

2. Predict for custom measurements via CLI:
   python3 src/predict.py --weight 250.2 --length 100.04
   python3 src/predict.py --weight 262.3 --length 100.41
   python3 src/predict.py --length 100.02  # Missing weight (handled by imputer)
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import numpy as np
import pandas as pd

# Support execution both as script and module
try:
    from src.pipeline import FEATURE_NAMES
except ModuleNotFoundError:
    from pipeline import FEATURE_NAMES

DEFAULT_MODEL_PATH = "models/defect_pipeline.joblib"


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


def predict_defect(
    weight_g: Optional[Union[float, int]],
    length_mm: Union[float, int],
    pipeline=None,
    model_path: str = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Predict defect status for a single manufactured part.

    Parameters
    ----------
    weight_g : float or None
        Measured part weight in grams. If None or NaN, the pipeline's
        SimpleImputer will impute using the training set median.
    length_mm : float
        Measured part length in millimeters.
    pipeline : sklearn.pipeline.Pipeline, optional
        Preloaded pipeline. If None, loaded from `model_path`.
    model_path : str, default='models/defect_pipeline.joblib'
        Path to the saved pipeline artifact.

    Returns
    -------
    dict
        Dictionary containing prediction results:
        - 'defective': bool (True = Defective, False = Passed)
        - 'status': 'DEFECTIVE' or 'PASSED QA'
        - 'defect_probability': float (if predict_proba available, else None)
        - 'input': dict of input features
    """
    if pipeline is None:
        pipeline = load_model(model_path)

    # Prepare single-row DataFrame with explicit feature column names
    input_data = pd.DataFrame(
        [{"weight_g": np.nan if weight_g is None else float(weight_g), "length_mm": float(length_mm)}],
        columns=FEATURE_NAMES,
    )

    pred_raw = pipeline.predict(input_data)[0]
    is_defective = bool(pred_raw)

    # Estimate probability if classifier provides predict_proba
    prob_defect = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(input_data)[0]
        # In scikit-learn, classes_ order determines column indexing
        classes = list(pipeline.classes_)
        if True in classes:
            idx = classes.index(True)
            prob_defect = float(probs[idx])

    return {
        "defective": is_defective,
        "status": "DEFECTIVE" if is_defective else "PASSED QA",
        "defect_probability": prob_defect,
        "input": {
            "weight_g": weight_g,
            "length_mm": length_mm,
        },
    }


def run_demo_predictions(pipeline=None):
    """
    Run demonstration predictions across realistic manufacturing scenarios.
    """
    print("\n" + "=" * 70)
    print(" MANUFACTURING QUALITY INSPECTOR: DEMONSTRATION RUN")
    print("=" * 70)

    test_cases = [
        {"desc": "Nominal compliant part", "weight": 250.2, "length": 100.04},
        {"desc": "In-spec slightly light part", "weight": 249.4, "length": 99.95},
        {"desc": "Defective: oversized length", "weight": 262.3, "length": 100.41},
        {"desc": "Defective: undersized length", "weight": 239.1, "length": 99.68},
        {"desc": "Missing weight sensor reading", "weight": None, "length": 100.03},
        {"desc": "Severe overweight anomaly", "weight": 275.0, "length": 100.35},
    ]

    if pipeline is None:
        pipeline = load_model(DEFAULT_MODEL_PATH)

    results = []
    for case in test_cases:
        res = predict_defect(case["weight"], case["length"], pipeline=pipeline)
        prob_str = f"{res['defect_probability']*100:.0f}%" if res["defect_probability"] is not None else "N/A"
        results.append({
            "Scenario": case["desc"],
            "Weight (g)": f"{case['weight']:.1f}" if case["weight"] is not None else "MISSING (NaN)",
            "Length (mm)": f"{case['length']:.2f}",
            "Status": res["status"],
            "Defect Prob": prob_str,
        })

    demo_df = pd.DataFrame(results)
    print(demo_df.to_string(index=False))
    print("=" * 70)
    print("Note: The 'MISSING (NaN)' weight was automatically imputed using the training median.")
    print("      Run with --help to see how to predict custom measurements via CLI.")


def main():
    parser = argparse.ArgumentParser(
        description="Predict manufacturing defect status from part weight and length."
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=None,
        help="Part weight in grams (e.g. 250.5). If omitted, median imputation will be used.",
    )
    parser.add_argument(
        "--length",
        type=float,
        default=None,
        help="Part length in millimeters (e.g. 100.05).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to saved pipeline (default: {DEFAULT_MODEL_PATH}).",
    )

    args = parser.parse_args()

    # If no arguments provided, run demo mode
    if args.length is None and args.weight is None:
        run_demo_predictions()
        return

    if args.length is None:
        parser.error("The --length argument is required when running single-part predictions.")

    result = predict_defect(
        weight_g=args.weight,
        length_mm=args.length,
        model_path=args.model,
    )

    weight_disp = f"{result['input']['weight_g']} g" if result['input']['weight_g'] is not None else "None (Imputed)"
    print("\n--- Manufacturing Quality Inspection Result ---")
    print(f"Input Weight:       {weight_disp}")
    print(f"Input Length:       {result['input']['length_mm']} mm")
    print(f"Inspection Result:  {result['status']}")
    print(f"Defective Flag:     {result['defective']}")
    if result["defect_probability"] is not None:
        print(f"Defect Probability: {result['defect_probability'] * 100:.1f}%")
    print("------------------------------------------------\n")


if __name__ == "__main__":
    main()
