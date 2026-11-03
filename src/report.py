"""
EDA and Evaluation Report (Track 1)
===================================
Writes figures and text for the saved model to reports/:

- eda_sensor_distributions.png: each model input, normal runs vs. failures
- eda_failure_rules.png: two UCI failure rules drawn over the data
- confusion_matrix.png: test split and unseen UCI rows, at FAILURE_THRESHOLD
- classification_report.txt: per-class precision, recall and F1 for both sets
- feature_importance.png / feature_importance.csv: permutation and impurity importance
- metrics.json: precision, recall, F1, accuracy and confusion matrix for the train split, test split and unseen rows

docs/EVALUATION.md explains how to read each one.

Usage:
------
python3 src/report.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # writes files only, so it also works without a display (CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# The saved model refers to src.pipeline by name, so that import must work however this file is run.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import (
    FAILURE_THRESHOLD,
    TELEMETRY_FEATURES,
    add_interaction_features,
    load_and_clean_data,
    load_unseen_ai4i,
    predict_failure,
)
from src.predict import DEFAULT_MODEL_PATH, load_model

# Reference palette: categorical slots 1-2 (validated colorblind-safe), text and surface tokens
NORMAL_COLOR = "#2a78d6"
FAILURE_COLOR = "#eb6834"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
CM_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

FEATURE_LABELS = {
    "air_temperature_k": "Air temperature (K)",
    "process_temperature_k": "Process temperature (K)",
    "rotational_speed_rpm": "Rotational speed (rpm)",
    "torque_nm": "Torque (Nm)",
    "tool_wear_min": "Tool wear (min)",
    "temperature_difference_k": "Temperature difference (K)",
    "power_w": "Power (W)",
    "wear_torque_min_nm": "Tool wear × torque (min·Nm)",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": TEXT,
    "axes.labelcolor": TEXT_MUTED,
    "axes.edgecolor": GRID,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 11,
    "font.size": 9,
    "legend.frameon": False,
})


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path


def plot_sensor_distributions(X: pd.DataFrame, y: pd.Series, path: Path) -> Path:
    """Histogram of each model input, scaled per class so 850 normal runs and 150 failures compare."""
    features = add_interaction_features(X)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))
    for ax, column in zip(axes.flat, FEATURE_LABELS):
        values = features[column].dropna()
        bins = np.histogram_bin_edges(values, bins=30)
        for label, color, name in ((0, NORMAL_COLOR, "Normal run"), (1, FAILURE_COLOR, "Machine failure")):
            ax.hist(values[y.loc[values.index] == label], bins=bins, density=True,
                    histtype="step", linewidth=2, color=color, label=name)
        ax.set_title(FEATURE_LABELS[column], loc="left")
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    axes.flat[0].legend(loc="upper left")
    fig.suptitle("Model inputs: normal runs vs. machine failures (each class scaled to equal area)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


def plot_failure_rules(X: pd.DataFrame, y: pd.Series, path: Path) -> Path:
    """Draw the UCI power and overstrain rules (docs/DATASET_CARD.md §2b) over the data."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2))
    normal, failure = X[y == 0], X[y == 1]
    marker_style = dict(s=36, edgecolors=SURFACE, linewidths=0.8)

    for ax, x_col, y_col in ((left, "rotational_speed_rpm", "torque_nm"), (right, "tool_wear_min", "torque_nm")):
        ax.scatter(normal[x_col], normal[y_col], color=NORMAL_COLOR, alpha=0.55, marker="o", label="Normal run", **marker_style)
        ax.scatter(failure[x_col], failure[y_col], color=FAILURE_COLOR, marker="^", label="Machine failure", **marker_style)
        ax.set_xlabel(FEATURE_LABELS[x_col])
        ax.set_ylabel(FEATURE_LABELS[y_col])

    rpm = np.linspace(X["rotational_speed_rpm"].min(), X["rotational_speed_rpm"].max(), 200)
    for watts in (3500, 9000):
        left.plot(rpm, watts / (rpm * 2 * np.pi / 60), color=TEXT_MUTED, linewidth=1.5)
        left.annotate(f"{watts:,} W", (rpm[-1], watts / (rpm[-1] * 2 * np.pi / 60)),
                      xytext=(4, 0), textcoords="offset points", va="center", color=TEXT_MUTED)
    left.set_title("Power rule: failure when power is below 3,500 W or above 9,000 W", loc="left")
    left.set_ylim(0, X["torque_nm"].max() * 1.05)

    wear = np.linspace(60, X["tool_wear_min"].max(), 200)
    right.plot(wear, 11000 / wear, color=TEXT_MUTED, linewidth=1.5)
    right.annotate("11,000 min·Nm", (wear[-1], 11000 / wear[-1]), xytext=(4, 0),
                   textcoords="offset points", va="center", color=TEXT_MUTED)
    right.set_title("Overstrain rule: failure above 11,000-13,000 min·Nm, by product type", loc="left")
    right.set_ylim(0, X["torque_nm"].max() * 1.05)
    right.legend(loc="lower left")
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion_matrices(results, path: Path) -> Path:
    """One matrix per evaluation set, shaded by the share of each actual class."""
    fig, axes = plt.subplots(1, len(results), figsize=(5.4 * len(results), 4.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", CM_RAMP)
    for ax, (title, cm) in zip(np.atleast_1d(axes), results):
        share = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.1%} of row", ha="center", va="center",
                        color="white" if share[i, j] > 0.5 else TEXT, fontsize=10)
        ax.set_xticks([0, 1], ["Predicted normal", "Predicted failure"])
        ax.set_yticks([0, 1], ["Actual normal", "Actual failure"])
        ax.grid(False)
        ax.set_title(title, loc="left")
    fig.suptitle(f"Confusion matrices for the saved model (failure flagged at probability ≥ {FAILURE_THRESHOLD})",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


def plot_feature_importance(permutation: pd.DataFrame, impurity: pd.Series, path: Path) -> Path:
    """Two views: what each sensor is worth on unseen rows, and how the forest split on each input."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8))
    perm = permutation.sort_values("mean")
    left.barh([FEATURE_LABELS[f] for f in perm.index], perm["mean"], xerr=perm["std"],
              color=NORMAL_COLOR, height=0.6, error_kw=dict(ecolor=TEXT_MUTED, lw=1, capsize=3))
    left.set_xlabel("Drop in average precision when the sensor is shuffled")
    left.set_title("Permutation importance of each sensor (unseen UCI rows)", loc="left")

    imp = impurity.sort_values()
    right.barh([FEATURE_LABELS[f] for f in imp.index], imp.values, color=NORMAL_COLOR, height=0.6)
    right.set_xlabel("Share of impurity reduction (sums to 1)")
    right.set_title("Random Forest impurity importance of each model input", loc="left")
    for ax in (left, right):
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, path)


def write_reports(
    out_dir: str = "reports",
    data_path: str = "manufacturing_samples.csv",
    model_path: str = DEFAULT_MODEL_PATH,
    n_repeats: int = 10,
    random_state: int = 42,
) -> list:
    """Write every EDA and evaluation artifact for the saved model and return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipe = load_model(model_path)

    X, y = load_and_clean_data(data_path)
    # Same split as src/train.py, so the test rows are the ones the saved model never saw.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    X_unseen, y_unseen = load_unseen_ai4i(data_path)

    metrics = {"threshold": FAILURE_THRESHOLD, "model_path": str(model_path), "sets": {}}
    for key, X_eval, y_eval in (("train", X_train, y_train), ("test", X_test, y_test), ("unseen", X_unseen, y_unseen)):
        y_pred = predict_failure(pipe, X_eval)
        metrics["sets"][key] = {
            "rows": int(len(y_eval)),
            "failures": int(y_eval.sum()),
            "precision": round(float(precision_score(y_eval, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_eval, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_eval, y_pred, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y_eval, y_pred)), 4),
            "confusion_matrix": confusion_matrix(y_eval, y_pred, labels=[0, 1]).tolist(),
        }
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    written = [
        plot_sensor_distributions(X, y, out / "eda_sensor_distributions.png"),
        plot_failure_rules(X, y, out / "eda_failure_rules.png"),
    ]

    evaluations = [
        (f"Test split (n = {len(X_test):,}, {int(y_test.sum())} failures)", X_test, y_test),
        (f"Unseen UCI rows (n = {len(X_unseen):,}, {int(y_unseen.sum())} failures)", X_unseen, y_unseen),
    ]
    matrices, report_lines = [], [
        f"Saved model: {model_path}",
        f"Failure flagged at probability >= {FAILURE_THRESHOLD}",
        "AI4I 2020 is synthetic; these scores do not show real-factory performance.",
        "",
    ]
    for title, X_eval, y_eval in evaluations:
        y_pred = predict_failure(pipe, X_eval)
        matrices.append((title, confusion_matrix(y_eval, y_pred, labels=[0, 1])))
        report_lines += [title, classification_report(
            y_eval, y_pred, labels=[0, 1], target_names=["normal", "failure"], digits=3, zero_division=0)]
    written.append(plot_confusion_matrices(matrices, out / "confusion_matrix.png"))
    report_path = out / "classification_report.txt"
    report_path.write_text("\n".join(report_lines))
    written.append(report_path)

    # Shuffling a raw sensor also changes the features derived from it, so this measures each sensor's full value.
    perm = permutation_importance(pipe, X_unseen, y_unseen, scoring="average_precision",
                                  n_repeats=n_repeats, random_state=random_state)
    permutation = pd.DataFrame({"mean": perm.importances_mean, "std": perm.importances_std}, index=TELEMETRY_FEATURES)
    model_inputs = list(pipe.named_steps["imputer"].feature_names_in_)
    impurity = pd.Series(pipe.named_steps["classifier"].feature_importances_, index=model_inputs)

    table = pd.concat([
        permutation.rename(columns={"mean": "permutation_ap_drop_mean", "std": "permutation_ap_drop_std"}),
        impurity.rename("impurity_importance"),
    ], axis=1).loc[model_inputs]
    table.index.name = "feature"
    csv_path = out / "feature_importance.csv"
    table.round(4).to_csv(csv_path)
    written += [plot_feature_importance(permutation, impurity, out / "feature_importance.png"), csv_path, metrics_path]
    return written


if __name__ == "__main__":
    for written_path in write_reports():
        print(f"Wrote {written_path}")
