# CLAUDE.md

A scikit-learn learning and portfolio project that predicts machine failure from the UCI AI4I 2020 benchmark. Older NumPy, pandas and Matplotlib exercises sit in the repo root.

## Commands

Run everything from the repo root, because data and model paths are relative to the working directory.

Use `.venv/bin/python` (Python 3.15, pinned dependencies). A bare `python3` has no dependencies installed. If `.venv` is missing, create it: `python3.15 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

```bash
.venv/bin/python -m unittest discover -s tests -v        # 34 tests, about 10 s; pytest is not installed
.venv/bin/python -m unittest tests.test_pipeline.TestManufacturingPipeline.test_pipeline_structure  # one test
.venv/bin/streamlit run app.py                           # Streamlit app at http://localhost:8501
.venv/bin/python src/predict.py                          # demo scenarios
.venv/bin/python src/predict.py --rpm 1280 --torque 69.8 --tool_wear 195  # omitted flags use defaults
.venv/bin/python src/train.py                            # overwrites the saved model
.venv/bin/python scripts/adapt_ai4i_dataset.py           # uses data/raw/ai4i2020.csv, overwrites manufacturing_samples.csv
```

The last two reproduce the tracked files exactly (checked 2 Nov 2026), so any diff after running them means something changed.

`.github/workflows/tests.yml` runs the tests on Linux for every push and pull request. macOS was dropped because hosted Mac runners kept being unavailable, so run the suite locally before pushing. `.github/workflows/keep-awake.yml` runs `scripts/keep_awake.py` every 6 hours, which opens the live app in a browser and fails if it doesn't load.

Live app: https://manufacturing-quality-analysis-defect-prediction-mkbvjywxtqkhq.streamlit.app. It deploys automatically from `main`. There is no linter or formatter.

## Layout

- `src/pipeline.py`: data loading, `create_pipeline`, the derived features, `FAILURE_THRESHOLD` (0.4) and `predict_failure`, plus the feature and target constants.
- `src/train.py`: trains and compares three models, then saves the 300-tree Random Forest to `models/defect_pipeline.joblib`.
- `src/report.py`: writes the EDA and evaluation figures and tables for the saved model to `reports/`. `docs/EVALUATION.md` explains them, and its numbers must be updated whenever the model or threshold changes. Run `.venv/bin/python src/report.py` after `src/train.py`.
- `app.py` and `app_pages/`: the Streamlit app. Pages only load `models/defect_pipeline.joblib` (cached with `st.cache_resource`) and read `reports/`. They never train. Keep logic in `src/app_logic.py`, which has no Streamlit import, so it stays unit-testable.
- `src/app_logic.py`: sensor validation (rejects missing, non-numeric and implausible values; warns outside the training range), the Track 2 rule `inspect_part`, and safe model and report loading. The rule constants come from `scripts/adapt_ai4i_dataset.py`, so the app and the generator can't drift apart.
- `src/predict.py`: `predict_machine_failure` and the CLI. `predict_defect` is a deprecated alias that raises `TypeError` on the old `(weight_g, length_mm)` arguments.
- `data/raw/ai4i2020.csv`: unmodified UCI source file (CC BY 4.0). Don't edit it.
- `scripts/adapt_ai4i_dataset.py`: generates `manufacturing_samples.csv` from the raw file. It checks the SHA-256 and downloads only if the file is missing.
- `tests/test_app.py`: app tests. They include running every page headlessly with `streamlit.testing.v1.AppTest`, and a check that the app never calls `fit`.
- `tests/test_pipeline.py`: `unittest` suite. Most tests use small in-memory data, and others check the saved model and that the generator reproduces the tracked CSV. No test needs network access.
- `docs/DATASET_CARD.md`: column provenance, generator formulas and UCI failure rules. Read it before changing features, labels or the generator.
- Root `*.py` scripts: finished learning exercises. They run on execution (no `main` guard) and some write files, for example `quality_analyser.py` overwrites `quality_report.csv` and the visualization scripts overwrite PNGs. Keep them rather than deleting or refactoring them. `manufacturing_visualisation.py` and `manufacturing_visualizations.py` are separate scripts, despite the similar names.

## Two tracks

| | Features | Target | Status |
|---|---|---|---|
| Track 1 (`feature_set="telemetry"`, default) | `air_temperature_k`, `process_temperature_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`, plus three features derived in the pipeline | `source_machine_failure` | Used by `train.py`, `predict.py` and the saved model |
| Track 2 (`feature_set="dimensional"`) | `weight_g`, `length_mm` | `defective` | Kept only to demonstrate leakage |

## Rules for this project

- **Be honest about the data.** All of it is synthetic, including AI4I 2020, whose label comes from rules on these same sensors. Don't describe scores as real-factory performance, or the generated dimensions as measurements.
- **Don't feed project-generated columns into Track 1.** `weight_g`, `length_mm`, `width_mm`, `height_mm`, `tolerance_deviation_mm` and `defective` are derived from the failure label. `failure_type` is also off limits, as a feature or a target, because it encodes the outcome and simplifies multi-mode failures.
- **Prevent preprocessing leakage.** Split first, and keep imputation and scaling inside the `Pipeline` from `create_pipeline`, so they are fitted only on training data.
- **Judge models on failure recall, precision and F1, not accuracy.** Predicting "normal" for every row already scores 0.85. The file oversamples failures to 15% (the UCI rate is 3.39%), which flatters precision. Compare models with the repeated cross-validation that `train.py` prints, not the single split, which has only 30 failures.
- **Keep the unseen rows for reporting.** `load_unseen_ai4i` returns the 9,000 raw rows outside the sample. Don't choose models or thresholds on them, or that check stops being honest.
- **Use `predict_failure`, not `pipeline.predict`,** because `pipeline.predict` ignores `FAILURE_THRESHOLD`.
- **Report only numbers you got from a run.** If retraining changes the metrics, update the cross-validation, unseen-row and single-split results in `README.md`, `docs/DATASET_CARD.md` §4 and `docs/PROJECT_LOG.md` §4.2, then rerun `src/report.py` (which also refreshes `reports/metrics.json` for the app) and update `docs/EVALUATION.md`.

## Gotchas

- `models/defect_pipeline.joblib` is tracked in git. Despite its name, it predicts machine failure, not part defects. Retrain it if you change the features or the pipeline, because `predict.py` depends on them.
- `requirements.txt` pins exact versions because the saved model only works reliably with the scikit-learn that built it. A file built with 1.6.1 crashed under 1.9.1 with `AttributeError`. If you change the pins, retrain and run the tests.
- `src/train.py`, `src/predict.py`, `src/report.py` and `src/app_logic.py` add the repo root to `sys.path` before importing `src.pipeline`. Keep this: the saved model refers to `src.pipeline.add_interaction_features` by name, so it only loads if that import works.
- `manufacturing_samples.csv` has 8 deliberately missing `weight_g` values. This doesn't affect Track 1.
- Streamlit doesn't re-import helper modules like `app_pages/common.py` or `src/` while the server runs. Restart `streamlit run` after changing them.
