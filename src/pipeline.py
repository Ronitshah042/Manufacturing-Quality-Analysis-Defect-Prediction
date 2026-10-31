"""
Pipeline and Data Loading Module
================================
Defines the machine learning preprocessing and modeling pipeline
for manufacturing defect prediction.

Features used:
- weight_g: Measured part weight in grams (Nominal: 250.0 g)
- length_mm: Measured part length in millimeters (Nominal: 100.00 mm)

Features excluded:
- product_id: An arbitrary tracking identifier (e.g. PART-101) with no
  physical causal relationship to part quality. Including it risks memorization.
- tolerance_deviation_mm: Redundant because it is directly derived from
  length_mm (tolerance_deviation_mm = length_mm - 100.00 mm). Including both
  introduces collinearity.
- width_mm / height_mm: Reserved for potential future multi-feature iterations.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

FEATURE_NAMES = ["weight_g", "length_mm"]
TARGET_NAME = "defective"


def load_and_clean_data(csv_path: str = "manufacturing_samples.csv") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load manufacturing CSV data, drop duplicate records, and return feature matrix X and target y.

    Parameters
    ----------
    csv_path : str
        Path to the manufacturing samples CSV file.

    Returns
    -------
    X : pd.DataFrame
        Cleaned feature matrix containing only ['weight_g', 'length_mm'].
    y : pd.Series
        Cleaned boolean target series ('defective').
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Manufacturing dataset not found at: {csv_path}")

    df = pd.read_csv(path)

    # 1. Deduplication: remove exact duplicate rows before train/test splitting
    # to avoid data leakage between splits (e.g. duplicate PART-107).
    df_clean = df.drop_duplicates().copy()

    # 2. Validate expected columns
    for col in FEATURE_NAMES + [TARGET_NAME]:
        if col not in df_clean.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")

    # 3. Extract features and target
    X = df_clean[FEATURE_NAMES].copy()
    y = df_clean[TARGET_NAME].astype(bool).copy()

    return X, y


def create_pipeline(classifier: Optional[BaseEstimator] = None, random_state: int = 42) -> Pipeline:
    """
    Build a reproducible Scikit-learn Pipeline with median imputation and a classifier.

    Parameters
    ----------
    classifier : BaseEstimator, optional
        Scikit-learn classifier. Defaults to DecisionTreeClassifier(random_state=random_state).
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline containing:
        - 'imputer': SimpleImputer(strategy='median')
        - 'classifier': Chosen classifier estimator
    """
    if classifier is None:
        classifier = DecisionTreeClassifier(random_state=random_state)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", classifier)
    ])

    return pipeline
