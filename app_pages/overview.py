import streamlit as st

from app_pages.common import get_metrics, page_header

page_header(
    "Manufacturing analytics portfolio project",
    "Manufacturing Quality & Predictive Maintenance",
    "Predict machine failure from five sensor readings with a trained scikit-learn model, "
    "check simulated parts against a documented quality rule, and review how the model was evaluated.",
)

left, right = st.columns(2)
with left, st.container(border=True):
    st.subheader(":material/build: Predictive Maintenance")
    st.write(
        "Enter air and process temperature, rotational speed, torque and tool wear. A Random Forest "
        "pipeline trained on the UCI AI4I 2020 sensor columns returns a failure prediction and probability."
    )
    st.caption("Machine-learning model · Track 1")
    st.page_link("app_pages/maintenance.py", label="Open Predictive Maintenance", icon=":material/arrow_forward:")
with right, st.container(border=True):
    st.subheader(":material/straighten: Part Quality Inspection")
    st.write(
        "Enter part weight and tolerance deviation. The app applies the exact rule this project used to "
        "label simulated parts, and shows which limit decided the result."
    )
    st.caption("Deterministic rule, no model · Track 2")
    st.page_link("app_pages/quality.py", label="Open Part Quality Inspection", icon=":material/arrow_forward:")

st.subheader("The model")
metrics = get_metrics()
if metrics is None:
    st.info("Evaluation metrics are not available yet. Run `python src/report.py` to generate them.")
else:
    unseen = metrics["sets"]["unseen"]
    st.caption(
        f"Saved model on the {unseen['rows']:,} UCI rows it never saw ({unseen['failures']} failures), "
        f"flagging failure at probability ≥ {metrics['threshold']}. From `reports/metrics.json`."
    )
    cols = st.columns(4)
    cols[0].metric("Failure recall", f"{unseen['recall']:.1%}", help="Share of real failures the model caught")
    cols[1].metric("Failure precision", f"{unseen['precision']:.1%}", help="Share of failure alerts that were real failures")
    cols[2].metric("Failure F1", f"{unseen['f1']:.3f}")
    cols[3].metric("Missed failures", f"{unseen['confusion_matrix'][1][0]}")

st.markdown(
    "- **Inputs:** five sensors, plus three features derived inside the pipeline: temperature difference, power and tool wear × torque.\n"
    "- **Pipeline:** feature derivation → median imputation → standard scaling → 300-tree Random Forest, all fitted on the training split only.\n"
    "- **Stack:** Python, pandas, NumPy, scikit-learn, Matplotlib, joblib and Streamlit."
)
st.page_link("app_pages/analytics.py", label="See Analytics & Evaluation", icon=":material/insights:")
