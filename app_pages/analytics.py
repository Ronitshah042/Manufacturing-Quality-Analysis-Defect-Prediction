import pandas as pd
import streamlit as st

from app_pages.common import DOCS_URL, get_feature_importance, get_metrics, page_header
from src.app_logic import REPO_ROOT, REPORTS_DIR, missing_reports

page_header(
    "Evaluation artifacts",
    "Analytics & Evaluation",
    "Results, figures and tables for the saved model, read from the files `src/report.py` wrote to `reports/`. "
    "Nothing on this page is recomputed when you interact with it.",
)

missing = missing_reports()
if missing:
    st.error(
        "These report files are missing, so their sections are hidden: "
        + ", ".join(f"`{name}`" for name in missing)
        + ". Run `python src/report.py` from the repository root to generate them.",
        icon=":material/error:",
    )


def show_image(path, caption):
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"`{path.relative_to(REPO_ROOT)}` is missing.")


metrics = get_metrics()
results, matrices, importance, sensors, dimensions = st.tabs(
    ["Model results", "Confusion matrices", "Feature importance", "Sensor EDA", "Dimension EDA (Track 2)"]
)

with results:
    st.markdown(
        "Three sets, never mixed:\n"
        "- **Train split:** the 800 rows the model learned from. Scores here only show fit, not performance.\n"
        "- **Test split:** 200 rows of the same file held out from training (15% failures, as in the file).\n"
        "- **Unseen rows:** the 9,000 UCI rows outside the training file (2.10% failures). They come from the same "
        "synthetic UCI generator, so they're new rows, not independent real-world data."
    )
    if metrics is None:
        st.info("`reports/metrics.json` is missing, so no metrics are shown.")
    else:
        names = {"train": "Train split", "test": "Test split", "unseen": "Unseen UCI rows"}
        table = pd.DataFrame([
            {"Set": names[key], "Rows": s["rows"], "Failures": s["failures"], "Failure precision": s["precision"],
             "Failure recall": s["recall"], "Failure F1": s["f1"], "Accuracy": s["accuracy"]}
            for key, s in metrics["sets"].items()
        ])
        score = st.column_config.NumberColumn(format="%.3f")
        st.dataframe(
            table, hide_index=True, width="stretch",
            column_config={"Rows": st.column_config.NumberColumn(format="localized"), "Failure precision": score,
                           "Failure recall": score, "Failure F1": score, "Accuracy": score},
        )
        st.caption(f"Failure flagged at probability ≥ {metrics['threshold']}. Source: `reports/metrics.json`.")
        st.markdown(
            "- **Precision** is the share of failure alerts that were real failures.\n"
            "- **Recall** is the share of real failures that were caught.\n"
            "- **F1** combines the two.\n"
            "- **Accuracy** isn't useful here: predicting \"normal\" for every row would already score 0.85 on the "
            "test split and 0.98 on the unseen rows."
        )
        st.markdown(
            "For model comparisons, use the repeated cross-validation in the "
            f"[evaluation guide]({DOCS_URL}/EVALUATION.md) and README, not a single split."
        )
    report_path = REPORTS_DIR / "classification_report.txt"
    if report_path.exists():
        with st.expander("Full classification report"):
            st.code(report_path.read_text(), language=None)

with matrices:
    show_image(REPORTS_DIR / "confusion_matrix.png", "Rows: actual outcome. Columns: prediction.")
    st.markdown(
        "- **False negative (missed failure):** a machine fails without warning. In a factory that means "
        "unplanned downtime, scrapped parts and possible tool damage.\n"
        "- **False positive (false alarm):** a healthy machine is stopped for an inspection that finds nothing. "
        "That costs time, and too many false alarms teach people to ignore alerts."
    )
    if metrics is not None:
        unseen = metrics["sets"]["unseen"]
        (tn, fp), (fn, tp) = unseen["confusion_matrix"]
        normals = tn + fp
        st.markdown(
            f"**Why false alarms dominate on unseen rows:** only {unseen['failures']} of {unseen['rows']:,} runs fail. "
            f"A false-alarm rate of just {fp / normals:.1%} on {normals:,} normal runs still gives {fp} false alarms, "
            f"more than the {tp} failures caught. Class imbalance turns a small rate into a large count."
        )

with importance:
    show_image(REPORTS_DIR / "feature_importance.png", "Global importance for the saved model.")
    table = get_feature_importance()
    if table is not None:
        st.dataframe(table, width="stretch")
    st.markdown(
        "- **Permutation importance (left):** how much the model's ranking of failures worsens on unseen rows "
        "when one sensor is shuffled. Shuffling a sensor also breaks the features derived from it.\n"
        "- **Impurity importance (right):** how much each of the eight model inputs was used in the forest's "
        "splits during training. It splits credit between related inputs like torque and power.\n\n"
        "Both are global: neither explains why one particular reading got its prediction."
    )

with sensors:
    show_image(REPORTS_DIR / "eda_sensor_distributions.png",
               "Each model input, normal runs vs. failures, each class scaled to equal area.")
    st.write("Look for places where the failure line sits away from the normal one: low speed, high torque, "
             "tool wear past 200 minutes, temperature difference below 8.6 K, and power at either extreme.")
    show_image(REPORTS_DIR / "eda_failure_rules.png", "Two UCI failure rules drawn over the data.")
    st.write("Every run outside the 3,500–9,000 W power curves is a failure. Failures above the overstrain curve "
             "cluster at high tool wear. The derived power and strain features let the model cut along these curves.")

with dimensions:
    st.write("These plot the dimensions this project generated, with the label rule's limits: "
             "0.12 mm tolerance deviation and 243–257 g weight.")
    show_image(REPO_ROOT / "manufacturing_scatter_plot.png", "Weight vs. absolute tolerance deviation.")
    st.write("Defective parts inside both limits are defective only through the machine-failure part of the rule.")
    show_image(REPO_ROOT / "manufacturing_dimension_boxplots.png", "Dimension distributions, passed vs. defective.")

st.markdown(f"Full explanations of every figure: [docs/EVALUATION.md]({DOCS_URL}/EVALUATION.md).")
