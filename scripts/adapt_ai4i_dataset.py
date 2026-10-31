"""
AI4I 2020 Predictive Maintenance Dataset Adapter
================================================
Fetches the open-source AI4I 2020 Predictive Maintenance Dataset
from the UCI Machine Learning Repository and adapts it to the manufacturing
quality inspection schema (part dimensions, mass, tolerance deviation, defects).

Source:
Stephan Matzka, 'Explainable Artificial Intelligence for Predictive Maintenance Applications',
Third International Conference on Artificial Intelligence for Industries (AI4I 2020).
UCI ML Repository: https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv
"""

import io
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

UCI_AI4I_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"


def fetch_raw_ai4i(url: str = UCI_AI4I_URL) -> pd.DataFrame:
    """Download the raw AI4I 2020 CSV dataset from UCI."""
    print(f"Downloading raw dataset from: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode("utf-8")
    return pd.read_csv(io.StringIO(content))


def adapt_to_manufacturing_schema(
    raw_df: pd.DataFrame,
    total_samples: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Adapt machine telemetry to physical fabricated part dimensions and quality labels.
    """
    np.random.seed(random_state)

    # 1. Stratified sampling: 150 failure cases and (total_samples - 150) normal cases
    failures = raw_df[raw_df["Machine failure"] == 1]
    non_failures = raw_df[raw_df["Machine failure"] == 0]

    n_fail = min(150, len(failures))
    n_norm = total_samples - n_fail

    sample_fail = failures.sample(n=n_fail, random_state=random_state)
    sample_norm = non_failures.sample(n=n_norm, random_state=random_state)

    df = pd.concat([sample_norm, sample_fail]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # 2. Extract telemetry variables
    temp_diff = df["Process temperature [K]"] - df["Air temperature [K]"]  # Nominal ~10 K
    tool_wear = df["Tool wear [min]"]                                     # 0 to 250 min
    torque = df["Torque [Nm]"]                                            # ~40 Nm
    rpm = df["Rotational speed [rpm]"]                                    # ~1500 rpm

    # 3. Model Physical Part Length and Tolerance Deviation:
    # Tool wear causes cutting edge recession and deflection.
    # Thermal expansion (temp_diff) causes part expansion.
    noise_len = np.random.normal(0.0, 0.02, len(df))
    tol_dev = ((tool_wear - 100.0) / 250.0 * 0.18) + ((temp_diff - 10.0) / 4.0 * 0.05) + noise_len

    # Machine failure events introduce catastrophic dimensional error
    fail_mask = df["Machine failure"] == 1
    tol_dev[fail_mask] += np.random.choice([-1, 1], size=fail_mask.sum()) * np.random.uniform(0.12, 0.25, size=fail_mask.sum())
    tol_dev = np.round(tol_dev, 4)

    length_mm = np.round(100.00 + tol_dev, 4)

    # 4. Model Part Weight:
    # Torque and RPM variance alter the Material Removal Rate (MRR).
    noise_weight = np.random.normal(0.0, 0.8, len(df))
    weight_g = 250.0 + ((torque - 40.0) / 20.0 * 4.5) - ((rpm - 1500.0) / 300.0 * 2.0) + noise_weight
    weight_g[fail_mask] += np.random.choice([-1, 1], size=fail_mask.sum()) * np.random.uniform(5.0, 12.0, size=fail_mask.sum())
    weight_g = np.round(weight_g, 2)

    # Add realistic sensor missingness (~0.8% NaN values in weight_g)
    nan_indices = np.random.choice(len(df), size=8, replace=False)
    weight_g[nan_indices] = np.nan

    # 5. Model Cross-Sectional Dimensions
    width_mm = np.round(50.00 + tol_dev * 0.5 + np.random.normal(0.0, 0.015, len(df)), 4)
    height_mm = np.round(25.00 + tol_dev * 0.25 + np.random.normal(0.0, 0.01, len(df)), 4)

    # 6. Quality Label (Defective):
    # Part fails QA if machine experienced catastrophic failure OR if physical dimensions
    # breach quality engineering limits (tolerance > 0.12 mm or mass drift > 7.0 g).
    defective = fail_mask | (np.abs(tol_dev) > 0.12) | (np.abs(weight_g - 250.0) > 7.0)

    # 7. Map Failure Type
    failure_type = np.where(df["TWF"] == 1, "Tool Wear",
                   np.where(df["HDF"] == 1, "Heat Dissipation",
                   np.where(df["PWF"] == 1, "Power Failure",
                   np.where(df["OSF"] == 1, "Overstrain",
                   np.where(df["RNF"] == 1, "Random Failure", "Normal Operation")))))

    adapted_df = pd.DataFrame({
        "product_id": [f"PART-{udi:04d}" for udi in df["UDI"]],
        "weight_g": weight_g,
        "length_mm": length_mm,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "tolerance_deviation_mm": tol_dev,
        "defective": defective.astype(bool),
        "air_temperature_k": df["Air temperature [K]"].round(2),
        "process_temperature_k": df["Process temperature [K]"].round(2),
        "rotational_speed_rpm": df["Rotational speed [rpm]"].astype(int),
        "torque_nm": df["Torque [Nm]"].round(2),
        "tool_wear_min": df["Tool wear [min]"].astype(int),
        "source_machine_failure": df["Machine failure"].astype(int),
        "failure_type": failure_type,
    })

    return adapted_df


def main():
    raw_df = fetch_raw_ai4i()
    adapted_df = adapt_to_manufacturing_schema(raw_df, total_samples=1000)

    output_path = Path("manufacturing_samples.csv")
    adapted_df.to_csv(output_path, index=False)
    print(f"Adapted dataset written to {output_path} ({len(adapted_df)} records)")
    print(f"Class distribution: {adapted_df['defective'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
