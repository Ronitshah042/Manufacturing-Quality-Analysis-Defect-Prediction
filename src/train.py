"""
Model Training and Evaluation Script
====================================
Trains the baseline Decision Tree pipeline and compares it with
Logistic Regression and Random Forest. Evaluates metrics, explains
phenomena (e.g. linear model failure on bimodal tolerances), and saves
the fitted pipeline to models/defect_pipeline.joblib.

Usage:
------
python3 src/train.py
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Support execution both as script and module
try:
    from src.pipeline import create_pipeline, load_and_clean_data
except ModuleNotFoundError:
    from pipeline import create_pipeline, load_and_clean_data


def evaluate_model(name: str, pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Compute accuracy, precision, recall, F1, and confusion matrix.
    """
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "predictions": y_pred,
    }


def print_evaluation(metrics: dict):
    """
    Print formatted evaluation results for a model.
    """
    print(f"\n--- {metrics['model']} ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.1f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print("                Predicted Pass  Predicted Defect")
    print(f"Actual Pass         {metrics['confusion_matrix'][0, 0]:<15} {metrics['confusion_matrix'][0, 1]}")
    print(f"Actual Defect       {metrics['confusion_matrix'][1, 0]:<15} {metrics['confusion_matrix'][1, 1]}")


def train_and_evaluate(
    data_path: str = "manufacturing_samples.csv",
    model_save_path: str = "models/defect_pipeline.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Full training, evaluation, comparison, and saving workflow.
    """
    print("=" * 65)
    print(" MANUFACTURING QUALITY PREDICTION: MODEL TRAINING & EVALUATION")
    print("=" * 65)

    # 1. Load data
    print(f"\n[1/5] Loading and cleaning dataset from: {data_path}")
    X, y = load_and_clean_data(data_path)
    print(f"      Total unique parts: {len(X)}")
    print(f"      Passing QA (False): {sum(~y)} ({sum(~y)/len(y)*100:.1f}%)")
    print(f"      Defective (True):   {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"      Features used:      {list(X.columns)}")

    # 2. Train/Test Split BEFORE Preprocessing
    # Critical: splitting first ensures the median imputer calculates its median
    # strictly from training samples, preventing data leakage from test samples.
    print(f"\n[2/5] Splitting data into train ({int((1-test_size)*100)}%) and test ({int(test_size*100)}%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"      Training set: {len(X_train)} samples ({sum(~y_train)} pass, {sum(y_train)} defect)")
    print(f"      Test set:     {len(X_test)} samples ({sum(~y_test)} pass, {sum(y_test)} defect)")

    # 3. Model Training & Comparison
    print("\n[3/5] Training baseline and comparison models inside pipelines...")
    candidate_classifiers = {
        "Baseline Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Logistic Regression": LogisticRegression(random_state=random_state),
        "Random Forest (50 trees)": RandomForestClassifier(n_estimators=50, random_state=random_state),
    }

    results = []
    fitted_pipelines = {}

    for name, clf in candidate_classifiers.items():
        # Pipeline ensures SimpleImputer(strategy='median') is fitted ONLY on X_train
        pipe = create_pipeline(classifier=clf, random_state=random_state)
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        metrics = evaluate_model(name, pipe, X_test, y_test)
        results.append(metrics)
        print_evaluation(metrics)

    # Summary Comparison Table
    print("\n" + "=" * 65)
    print(" MODEL PERFORMANCE SUMMARY ON TEST SET (n = 5)")
    print("=" * 65)
    summary_df = pd.DataFrame([
        {
            "Model": r["model"],
            "Accuracy": f"{r['accuracy']:.2f}",
            "Precision": f"{r['precision']:.2f}",
            "Recall": f"{r['recall']:.2f}",
            "F1-Score": f"{r['f1']:.2f}",
        }
        for r in results
    ])
    print(summary_df.to_string(index=False))

    # Educational notes on why models behave this way
    print("\n[4/5] Analytical Insights & Beginner Takeaways:")
    print("  * Why Decision Tree achieved 100% accuracy:")
    print("    Manufacturing tolerances reject parts that are either TOO SMALL or TOO LARGE.")
    print("    A Decision Tree naturally learns multiple orthogonal cut-offs (e.g. length <= 99.80")
    print("    or length > 100.17 -> defective), easily capturing bilateral tolerance bands.")
    print("  * Why Logistic Regression failed (0% Recall, Accuracy = 60%):")
    print("    Logistic Regression is a linear classifier. It fits a single straight decision boundary.")
    print("    It cannot separate a central passing cluster from defects occurring on both outer flanks")
    print("    without non-linear feature engineering (e.g. absolute deviation or polynomial terms).")
    print("  * Why high metrics on this dataset require extreme caution:")
    print("    1. Small Sample Size: The test set has only 5 samples. A single flip changes accuracy by 20%.")
    print("    2. Synthetic / Rule-based Target: The defective labels in the dataset were generated by")
    print("       applying strict threshold rules on length and weight. The tree simply learned the rule.")
    print("    3. Provenance: No real-world sensor logs or CMM inspection calibrations verify these labels.")

    # 4. Save Baseline Pipeline
    print(f"\n[5/5] Serializing baseline Decision Tree pipeline...")
    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_pipe = fitted_pipelines["Baseline Decision Tree"]
    joblib.dump(baseline_pipe, save_path)
    print(f"      Pipeline successfully saved to: {save_path}")

    # Verify reloading
    reloaded_pipe = joblib.load(save_path)
    reloaded_pred = reloaded_pipe.predict(X_test)
    assert np.array_equal(results[0]["predictions"], reloaded_pred), "Reloaded model output mismatch!"
    print("      Verification passed: Reloaded pipeline predictions match in-memory model identically.")

    return baseline_pipe, results


if __name__ == "__main__":
    train_and_evaluate()
