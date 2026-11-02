"""
Pipeline and Data Loading Module (Track 1: AI4I 2020 Sensor Columns)
====================================================================
Defines the machine learning preprocessing and modeling pipeline
for predicting machine failure from the original sensor columns of the
UCI AI4I 2020 Predictive Maintenance benchmark (a synthetic dataset).

Track 1 Source Features (copied unchanged from UCI):
- air_temperature_k: Factory ambient temperature in Kelvin [295 - 305 K]
- process_temperature_k: Cutting zone temperature in Kelvin [305 - 314 K]
- rotational_speed_rpm: CNC spindle speed in RPM [1168 - 2886 rpm]
- torque_nm: Spindle cutting torque in N*m [3.8 - 76.6 N*m]
- tool_wear_min: Cumulative milling insert wear in minutes [0 - 253 min]

Target:
- source_machine_failure: UCI 'Machine failure' label (1 = Failed, 0 = Normal).
  UCI sets it to 1 when any of five rule-based modes fires: Tool Wear (TWF),
  Heat Dissipation (HDF), Power (PWF), Overstrain (OSF), or Random (RNF).

Note on Data Provenance:
Track 1 avoids the *project-generated* leakage of Track 2 (predicting our own
threshold-generated `defective` label from our own generated dimensions).
It does NOT give real-factory evidence: the UCI label is itself generated from
rules on these sensor columns. See docs/DATASET_CARD.md.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Track 1: Original UCI AI4I 2020 sensor columns (Primary)
TELEMETRY_FEATURES = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
TELEMETRY_TARGET = "source_machine_failure"

# Track 2: Legacy / Digital Twin Workpiece Geometry (Reference)
DIMENSIONAL_FEATURES = ["weight_g", "length_mm"]
DIMENSIONAL_TARGET = "defective"

# Default to Track 1 sensor columns
FEATURE_NAMES = TELEMETRY_FEATURES
TARGET_NAME = TELEMETRY_TARGET

# Flag a failure at this probability: a missed failure is assumed to cost more than an inspection.
FAILURE_THRESHOLD = 0.4

RAW_AI4I_PATH = "data/raw/ai4i2020.csv"
RAW_TO_TELEMETRY = {
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
}


def add_interaction_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add the sensor combinations that UCI's failure rules use (docs/DATASET_CARD.md §2b)."""
    return X.assign(
        temperature_difference_k=X["process_temperature_k"] - X["air_temperature_k"],
        power_w=X["torque_nm"] * X["rotational_speed_rpm"] * 2 * np.pi / 60,
        wear_torque_min_nm=X["tool_wear_min"] * X["torque_nm"],
    )


def failure_probabilities(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return each row's probability of class 1, or zeros if training data had no failures."""
    classes = list(pipeline.classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return pipeline.predict_proba(X)[:, classes.index(1)]


def predict_failure(pipeline, X: pd.DataFrame, threshold: float = FAILURE_THRESHOLD) -> np.ndarray:
    """Return 0/1 failure predictions using the project's decision threshold."""
    if not hasattr(pipeline, "predict_proba"):
        return pipeline.predict(X).astype(int)
    return (failure_probabilities(pipeline, X) >= threshold).astype(int)


def load_unseen_ai4i(
    sample_csv: str = "manufacturing_samples.csv",
    raw_csv: str = RAW_AI4I_PATH,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return raw UCI rows that are not in the training sample, as X and y in the Track 1 schema."""
    raw = pd.read_csv(raw_csv)
    sample_ids = pd.read_csv(sample_csv)["product_id"].str.replace("PART-", "").astype(int)
    missing = sample_ids[~sample_ids.isin(raw["UDI"])]
    if not missing.empty:
        raise ValueError(f"{len(missing)} sample rows are not in {raw_csv}")

    unseen = raw[~raw["UDI"].isin(sample_ids)]
    X = unseen[list(RAW_TO_TELEMETRY)].rename(columns=RAW_TO_TELEMETRY).astype(float)
    y = unseen["Machine failure"].astype(int).rename(TELEMETRY_TARGET)
    return X.reset_index(drop=True), y.reset_index(drop=True)


def load_and_clean_data(
    csv_path: str = "manufacturing_samples.csv",
    feature_set: str = "telemetry",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load manufacturing CSV data, drop duplicate records, and return feature matrix X and target y.

    Parameters
    ----------
    csv_path : str
        Path to the manufacturing samples CSV file.
    feature_set : str, default='telemetry'
        'telemetry' for the original UCI sensor columns (Track 1), or
        'dimensional' for synthetic workpiece geometry features (Track 2).

    Returns
    -------
    X : pd.DataFrame
        Cleaned feature matrix.
    y : pd.Series
        Cleaned binary/boolean target series.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Manufacturing dataset not found at: {csv_path}")

    df = pd.read_csv(path)

    # 1. Deduplication: remove exact duplicate rows before train/test splitting
    # to avoid data leakage between splits.
    df_clean = df.drop_duplicates().copy()

    # 2. Select feature set and target
    if feature_set == "telemetry":
        features = TELEMETRY_FEATURES
        target = TELEMETRY_TARGET
    elif feature_set == "dimensional":
        features = DIMENSIONAL_FEATURES
        target = DIMENSIONAL_TARGET
    else:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Must be 'telemetry' or 'dimensional'.")

    # 3. Validate expected columns
    for col in features + [target]:
        if col not in df_clean.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")

    # 4. Extract features and target
    X = df_clean[features].copy()
    y = df_clean[target].astype(int).copy()

    return X, y


def create_pipeline(
    classifier: Optional[BaseEstimator] = None,
    include_scaler: bool = True,
    random_state: int = 42,
    include_interactions: bool = False,
) -> Pipeline:
    """
    Build a reproducible Scikit-learn Pipeline with median imputation,
    standard scaling, and an estimator.

    Parameters
    ----------
    classifier : BaseEstimator, optional
        Scikit-learn classifier. Defaults to DecisionTreeClassifier(random_state=random_state).
    include_scaler : bool, default=True
        Whether to include StandardScaler for normalizing telemetry features with
        disparate engineering units (RPM vs Nm vs Kelvin vs Minutes).
    random_state : int, default=42
        Random seed for reproducibility.
    include_interactions : bool, default=False
        Whether to derive the Track 1 interaction features. Requires the telemetry columns.
        A missing sensor makes its derived features missing too, and the imputer fills them.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline containing:
        - 'features': add_interaction_features (if include_interactions=True)
        - 'imputer': SimpleImputer(strategy='median')
        - 'scaler': StandardScaler() (if include_scaler=True)
        - 'classifier': Chosen classifier estimator
    """
    if classifier is None:
        classifier = DecisionTreeClassifier(random_state=random_state)

    steps = []
    if include_interactions:
        steps.append(("features", FunctionTransformer(add_interaction_features)))
    steps.append(("imputer", SimpleImputer(strategy="median")))
    if include_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))

    return Pipeline(steps)
