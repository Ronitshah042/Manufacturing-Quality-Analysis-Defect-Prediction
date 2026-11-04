import pandas as pd
import streamlit as st

from app_pages.common import get_feature_importance, get_model, get_training_ranges, page_header
from src.app_logic import SENSOR_SPECS, validate_sensor_inputs
from src.pipeline import FAILURE_THRESHOLD, add_interaction_features
from src.predict import DEMO_SCENARIOS, predict_machine_failure

page_header(
    "Track 1 · machine-learning model",
    "Predictive Maintenance",
    "Enter five sensor readings to get the saved model's machine-failure prediction. "
    "The app uses the same pipeline and prediction function as the command-line tool and the tests.",
)

model, model_error = get_model()
if model_error:
    st.error(model_error, icon=":material/error:")
    st.stop()

PRESET_KEYS = {"air_temperature_k": "air", "process_temperature_k": "proc", "rotational_speed_rpm": "rpm",
               "torque_nm": "torque", "tool_wear_min": "wear"}
presets = {"Enter my own values": None} | {f"Example: {s['desc']}": s for s in DEMO_SCENARIOS}
choice = st.selectbox(
    "Start from", list(presets),
    help="Examples are sample sensor readings only. Their results come from the model when you submit.",
)
preset = presets[choice]
ranges = get_training_ranges()

with st.form("sensor_form"):
    cols = st.columns(5)
    raw = {}
    for col, (key, spec) in zip(cols, SENSOR_SPECS.items()):
        low, high = ranges[key]
        raw[key] = col.number_input(
            f"{spec['label']} ({spec['unit']})",
            value=None if preset is None else float(preset[PRESET_KEYS[key]]),
            step=0.1 if key.endswith("_k") or key == "torque_nm" else 1.0,
            placeholder="Required",
            help=f"Training data range: {low:g} to {high:g} {spec['unit']}",
            key=f"{key}_{choice}",
        )
    submitted = st.form_submit_button("Predict machine failure", type="primary")

if submitted:
    values, errors, warnings = validate_sensor_inputs(raw, ranges)
    if errors:
        st.error("Please fix these inputs before predicting:\n\n" + "\n".join(f"- {e}" for e in errors), icon=":material/error:")
    else:
        result = predict_machine_failure(**values, pipeline=model)
        for warning in warnings:
            st.warning(warning, icon=":material/warning:")

        probability = result["failure_probability"]
        left, right = st.columns([1, 2])
        with left:
            st.metric("Model failure probability", "Not available" if probability is None else f"{probability:.1%}",
                      help=f"A run is flagged when this is at least {FAILURE_THRESHOLD:.0%}.")
        with right:
            if result["failure"]:
                st.warning(
                    f"**Failure predicted.** The model's estimate is at or above the {FAILURE_THRESHOLD:.0%} alert threshold. "
                    "Suggested next step: inspect the tool and check the load and cooling before the next run. "
                    "On unseen data, fewer than half of these alerts were real failures.",
                    icon=":material/build:",
                )
            else:
                st.success(
                    f"**No failure predicted.** The model's estimate is below the {FAILURE_THRESHOLD:.0%} alert threshold. "
                    "Suggested next step: continue normal monitoring. The model misses about 1 in 10 real failures, "
                    "mostly random tool-wear and random failures.",
                    icon=":material/check_circle:",
                )
        st.caption(
            "The probability is the share of the forest's trees voting for failure. It is a model estimate on a "
            "synthetic benchmark, not a calibrated real-world failure risk or a guarantee."
        )

        derived = add_interaction_features(pd.DataFrame([values])).iloc[0]
        summary = pd.DataFrame(
            [(f"{s['label']} ({s['unit']})", f"{values[k]:g}", f"{ranges[k][0]:g} to {ranges[k][1]:g}") for k, s in SENSOR_SPECS.items()]
            + [
                ("Temperature difference (K)", f"{derived['temperature_difference_k']:.1f}", "derived"),
                ("Power (W)", f"{derived['power_w']:,.0f}", "derived"),
                ("Tool wear × torque (min·N·m)", f"{derived['wear_torque_min_nm']:,.0f}", "derived"),
            ],
            columns=["Input", "Value", "Training range"],
        )
        st.subheader("Submitted readings")
        st.dataframe(summary, hide_index=True, width="stretch")

st.subheader("What the model relies on overall")
importance = get_feature_importance()
if importance is None:
    st.info("Feature importance is not available yet. Run `python src/report.py` to generate `reports/feature_importance.csv`.")
else:
    st.write(
        "This is **global** importance for the saved model across many runs. It does not explain any single "
        "prediction above. Shuffling torque hurts the model most, because it feeds both power and tool wear × torque."
    )
    perm = importance["permutation_ap_drop_mean"].dropna().sort_values(ascending=False)
    st.dataframe(
        pd.DataFrame({"Sensor": [SENSOR_SPECS[k]["label"] for k in perm.index], "Importance": perm.values}),
        hide_index=True, width="stretch",
        column_config={"Importance": st.column_config.NumberColumn(format="%.3f")},
    )
    st.caption(
        "Importance is the drop in average precision when that sensor is shuffled: permutation importance on "
        "the 9,000 unseen UCI rows, from `reports/feature_importance.csv`."
    )
