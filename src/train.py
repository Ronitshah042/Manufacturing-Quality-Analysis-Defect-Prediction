"""
Model Training and Evaluation Script (Track 1: AI4I 2020 Sensor Columns)
========================================================================
Trains Decision Tree, Logistic Regression, and Random Forest models on the
original UCI AI4I 2020 sensor columns plus three interaction features, to
predict source_machine_failure. AI4I 2020 is itself a synthetic benchmark whose
failure label is generated from rules on these sensors (see docs/DATASET_CARD.md).

Evaluates failure recall, precision, F1-score, and confusion matrices on a test
split, with repeated cross-validation, and on raw UCI rows outside the training
sample. Saves the fitted Random Forest pipeline.

Usage:
------
python3 src/train.py
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier

# The saved model refers to src.pipeline by name, so that import must work however this file is run.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import (
    FAILURE_THRESHOLD,
    TELEMETRY_TARGET,
    create_pipeline,
    load_and_clean_data,
    load_unseen_ai4i,
    predict_failure,
)


def evaluate_model(
    name: str,
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame = None,
    y_train: pd.Series = None,
    threshold: float = FAILURE_THRESHOLD,
) -> dict:
    """
    Compute training accuracy (if provided), test accuracy, precision, recall, F1, and confusion matrix.
    """
    # 1. Training metrics (measures learning fit)
    train_acc = None
    if X_train is not None and y_train is not None:
        y_train_pred = predict_failure(pipeline, X_train, threshold)
        train_acc = accuracy_score(y_train, y_train_pred)

    # 2. Test metrics (measures generalization to unseen data)
    y_pred = predict_failure(pipeline, X_test, threshold)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": name,
        "train_accuracy": train_acc,
        "accuracy": acc,
        "test_accuracy": acc,
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
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    print(f"\n--- {metrics['model']} ---")
    if metrics.get("train_accuracy") is not None:
        print(f"Train Accuracy:   {metrics['train_accuracy']:.4f} ({metrics['train_accuracy'] * 100:.1f}%)")
    print(f"Test Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.1f}%)")
    print(f"Failure Precision:{metrics['precision']:.4f} ({metrics['precision'] * 100:.1f}%)")
    print(f"Failure Recall:   {metrics['recall']:.4f} ({metrics['recall'] * 100:.1f}%)")
    print(f"F1-Score:         {metrics['f1']:.4f}")
    print(f"Confusion Matrix (Test Set, n = {int(cm.sum())}):")
    print("                  Predicted Normal  Predicted Failure")
    print(f"Actual Normal         {tn:<17} {fp:<10} (False Alarms: {fp})")
    print(f"Actual Failure        {fn:<17} {tp:<10} (Missed Failures: {fn})")


def cross_validate_failure_metrics(
    pipeline,
    X,
    y,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
    threshold: float = FAILURE_THRESHOLD,
) -> dict:
    """
    Score an unfitted pipeline on repeated stratified folds, refitting it per fold.
    Returns per-fold arrays for precision, recall, and F1 at the given threshold.
    """
    def scorer(estimator, X_fold, y_fold):
        y_pred = predict_failure(estimator, X_fold, threshold)
        return {
            "precision": precision_score(y_fold, y_pred, zero_division=0),
            "recall": recall_score(y_fold, y_pred, zero_division=0),
            "f1": f1_score(y_fold, y_pred, zero_division=0),
        }

    folds = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    scores = cross_validate(pipeline, X, y, cv=folds, scoring=scorer)
    return {name: scores[f"test_{name}"] for name in ("precision", "recall", "f1")}


def train_and_evaluate(
    data_path: str = "manufacturing_samples.csv",
    model_save_path: str = "models/defect_pipeline.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Full training, evaluation, comparison, and saving workflow for Track 1.
    """
    print("=" * 72)
    print(" MANUFACTURING QUALITY: TRACK 1 SENSOR TELEMETRY PREDICTIVE MAINTENANCE")
    print("=" * 72)

    # 1. Load data
    print(f"\n[1/5] Loading UCI AI4I 2020 sensor columns from: {data_path}")
    X, y = load_and_clean_data(data_path, feature_set="telemetry")
    n_total = len(X)
    n_norm = sum(y == 0)
    n_fail = sum(y == 1)

    print(f"      Total machine runs: {n_total}")
    print(f"      Normal runs (0):    {n_norm} ({n_norm / n_total * 100:.1f}%)")
    print(f"      Machine failure (1):{n_fail} ({n_fail / n_total * 100:.1f}%)")
    print(f"      Sensor features:    {list(X.columns)}")
    print(f"      Target:             {TELEMETRY_TARGET}")

    # 2. Train/Test Split BEFORE Preprocessing
    print(f"\n[2/5] Splitting data into train ({int((1-test_size)*100)}%) and test ({int(test_size*100)}%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"      Training set: {len(X_train)} samples ({sum(y_train == 0)} normal, {sum(y_train == 1)} failures)")
    print(f"      Test set:     {len(X_test)} samples ({sum(y_test == 0)} normal, {sum(y_test == 1)} failures)")

    # 3. Model Training & Comparison
    print("\n[3/5] Training baseline and comparison models inside pipelines...")
    candidate_classifiers = {
        "Baseline Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
        "Random Forest (300 trees)": RandomForestClassifier(n_estimators=300, random_state=random_state),
    }

    results = []
    fitted_pipelines = {}

    for name, clf in candidate_classifiers.items():
        # Pipeline ensures SimpleImputer and StandardScaler are fitted ONLY on X_train
        pipe = create_pipeline(classifier=clf, include_scaler=True, random_state=random_state, include_interactions=True)
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        metrics = evaluate_model(name, pipe, X_test, y_test, X_train=X_train, y_train=y_train)
        results.append(metrics)
        print_evaluation(metrics)

    # Summary Comparison Table
    print("\n" + "=" * 72)
    print(f" MODEL PERFORMANCE SUMMARY: TRAIN (n = {len(X_train)}) vs. TEST (n = {len(X_test)})")
    print("=" * 72)
    summary_df = pd.DataFrame([
        {
            "Model": r["model"],
            "Train Acc": f"{r['train_accuracy']:.4f}",
            "Test Acc": f"{r['accuracy']:.4f}",
            "Precision": f"{r['precision']:.4f}",
            "Recall": f"{r['recall']:.4f}",
            "F1-Score": f"{r['f1']:.4f}",
        }
        for r in results
    ])
    print(summary_df.to_string(index=False))

    # Cross-validation on all rows, since one test split has only 30 failures
    print(f"\nFailure threshold: {FAILURE_THRESHOLD}")
    print(f"5-FOLD CROSS-VALIDATION REPEATED 5 TIMES ON ALL {n_total} ROWS (mean ± std across folds)")
    cv_candidates = [
        (name, clone(clf), True, FAILURE_THRESHOLD) for name, clf in candidate_classifiers.items()
    ] + [
        ("Old: RF (50), sensors, t=0.5", RandomForestClassifier(n_estimators=50, random_state=random_state), False, 0.5),
    ]
    for name, clf, interactions, threshold in cv_candidates:
        cv = cross_validate_failure_metrics(
            create_pipeline(classifier=clf, include_scaler=True, random_state=random_state, include_interactions=interactions),
            X, y, random_state=random_state, threshold=threshold,
        )
        print(
            f"  {name:<29} "
            f"Precision {cv['precision'].mean():.3f} ± {cv['precision'].std():.3f}  "
            f"Recall {cv['recall'].mean():.3f} ± {cv['recall'].std():.3f}  "
            f"F1 {cv['f1'].mean():.3f} ± {cv['f1'].std():.3f}"
        )

    # Analytical Insights
    print("\n[4/5] Analytical Insights & Engineering Takeaways:")
    print("  * Interpretation caveat:")
    print("    - AI4I 2020 is a synthetic benchmark; its failure label is generated by rules on")
    print("      these same sensors (plus product Type and random TWF/RNF events). Scores measure how")
    print("      well each model approximates those rules, not real-factory performance.")
    print("    - The test set uses an oversampled 15% failure rate (source rate is 3.39%).")
    print("  * Failure recall per model:")
    print(f"      * Decision Tree Recall:     {results[0]['recall'] * 100:.1f}% ({results[0]['confusion_matrix'][1, 1]}/{sum(y_test == 1)} failures detected)")
    print(f"      * Logistic Regression Recall:{results[1]['recall'] * 100:.1f}% ({results[1]['confusion_matrix'][1, 1]}/{sum(y_test == 1)} failures detected)")
    print(f"      * Random Forest Recall:      {results[2]['recall'] * 100:.1f}% ({results[2]['confusion_matrix'][1, 1]}/{sum(y_test == 1)} failures detected)")
    print("  * Random Forest:")
    print(f"    - F1-score {results[2]['f1']:.4f}, precision {results[2]['precision'] * 100:.1f}% on a single")
    print("      200-row test split; expect noticeable variation with a different random split.")

    # 4. Save Top Performing Pipeline (Random Forest for Track 1)
    print(f"\n[5/5] Serializing fitted Random Forest pipeline to: {model_save_path}")
    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_pipe = fitted_pipelines["Random Forest (300 trees)"]
    joblib.dump(best_pipe, save_path, compress=3)
    print("      Pipeline successfully serialized.")

    # Reporting only: choosing models or thresholds on these rows would make this check optimistic.
    X_unseen, y_unseen = load_unseen_ai4i(data_path)
    unseen_metrics = evaluate_model("Saved model on unseen UCI rows", best_pipe, X_unseen, y_unseen)
    print(f"\n      {len(X_unseen)} raw UCI rows outside the sample, failure rate {y_unseen.mean() * 100:.2f}%:")
    print_evaluation(unseen_metrics)

    # Verify reloading
    reloaded_pipe = joblib.load(save_path)
    reloaded_pred = predict_failure(reloaded_pipe, X_test)
    assert np.array_equal(results[2]["predictions"], reloaded_pred), "Reloaded model output mismatch!"
    print("      Verification passed: Reloaded pipeline predictions match in-memory model identically.")

    return best_pipe, results


if __name__ == "__main__":
    train_and_evaluate()
