import altair as alt
import pandas as pd
import streamlit as st

from app_pages.common import page_header
from src.app_logic import DATA_PATH, PART_SPECS, inspect_part

page_header(
    "Track 2 · deterministic rule, no model",
    "Part Quality Inspection",
    "Check a simulated part against the exact rule this project used to label its generated dimensions. "
    "No model is involved: the result follows directly from the limits below.",
)
st.info(
    "**Rule** (from `scripts/adapt_ai4i_dataset.py`): a part is defective if the machine failed during its run, "
    "**or** |tolerance deviation| > 0.12 mm, **or** |weight − 250 g| > 7 g. "
    "Length, width and height are shown for reference but are not part of the rule.",
    icon=":material/rule:",
)


@st.cache_data
def example_parts():
    """One real row from the generated dataset for each way a part can end up."""
    df = pd.read_csv(DATA_PATH)
    tol_bad = df.tolerance_deviation_mm.abs() > 0.12
    weight_bad = (df.weight_g - 250).abs() > 7
    picks = {
        "Passes every limit": df[~df.defective],
        "Defective: tolerance deviation over the limit": df[tol_bad & ~weight_bad],
        "Defective: weight outside the limit": df[weight_bad & ~tol_bad],
        "Defective: machine failure only": df[df.defective & ~tol_bad & ~weight_bad & df.weight_g.notna()],
    }
    return {f"Example {rows.iloc[0].product_id}: {name}": rows.iloc[0].to_dict() for name, rows in picks.items()}


examples = example_parts()
choice = st.selectbox("Start from", ["Enter my own values"] + list(examples),
                      help="Examples are real rows from manufacturing_samples.csv.")
example = examples.get(choice)

with st.form("part_form"):
    cols = st.columns(5)
    raw = {}
    for col, (key, spec) in zip(cols, PART_SPECS.items()):
        raw[key] = col.number_input(
            f"{spec['label']} ({spec['unit']})",
            value=None if example is None else float(example[key]),
            step=0.01 if key == "weight_g" else 0.0001,
            format="%.2f" if key == "weight_g" else "%.4f",
            placeholder="Required" if key == "tolerance_deviation_mm" else "Optional",
            key=f"{key}_{choice}",
        )
    failure = st.radio("Machine failure recorded during this part's run?", ["No", "Yes"], horizontal=True,
                       index=1 if example is not None and example["source_machine_failure"] == 1 else 0,
                       key=f"failure_{choice}")
    submitted = st.form_submit_button("Inspect part", type="primary")

if submitted:
    result = inspect_part(raw, machine_failure=failure == "Yes")
    if result["errors"]:
        st.error("Please fix these inputs before inspecting:\n\n" + "\n".join(f"- {e}" for e in result["errors"]), icon=":material/error:")
    else:
        for warning in result["warnings"]:
            st.warning(warning, icon=":material/warning:")
        fired = [c["rule"] for c in result["checks"] if c["triggered"]]
        if result["defective"]:
            st.error(f"**Defective (simulated).** Triggered: {'; '.join(fired)}.", icon=":material/cancel:")
        else:
            st.success("**Passes (simulated).** No rule was triggered.", icon=":material/check_circle:")

        table = pd.DataFrame([
            {"Rule": c["rule"], "Value": c["value"], "Limit": c["limit"],
             "Result": "Triggered" if c["triggered"] else "Within limit"}
            for c in result["checks"]
        ])
        st.dataframe(table, hide_index=True, width="stretch")

        bar_names = {"|Tolerance deviation| > 0.12 mm": "Tolerance deviation", "|Weight − 250 g| > 7 g": "Weight"}
        bars = pd.DataFrame([
            {"Measurement": bar_names[c["rule"]], "Share of limit": c["ratio"],
             "Status": "Over the limit" if c["triggered"] else "Within the limit"}
            for c in result["checks"] if c["ratio"] is not None
        ])
        if not bars.empty:
            st.subheader("Measurements relative to their limits")
            chart = (
                alt.Chart(bars).mark_bar(cornerRadiusEnd=4).encode(
                    y=alt.Y("Measurement:N", title=None, sort=["Tolerance deviation", "Weight"]),
                    x=alt.X("Share of limit:Q", axis=alt.Axis(format="%"), title="Distance from nominal, as a share of the allowed limit",
                            scale=alt.Scale(domain=[0, max(1.5, float(bars["Share of limit"].max()) * 1.1)])),
                    color=alt.Color("Status:N", scale=alt.Scale(domain=["Within the limit", "Over the limit"], range=["#22d3ee", "#d03b3b"]),
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=["Measurement", alt.Tooltip("Share of limit:Q", format=".0%"), "Status"],
                )
                + alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(color="#e2e8f0", strokeWidth=2).encode(x="x:Q")
            ).properties(height=alt.Step(40))
            st.altair_chart(chart, width="stretch")
            st.caption("100% marks the limit. Bars past the line break the rule.")

st.caption(
    "This is a synthetic, rule-based simulation of the dimensions this project generated from sensor data. "
    "It is not validated industrial inspection, and no measuring equipment stands behind these fields."
)
