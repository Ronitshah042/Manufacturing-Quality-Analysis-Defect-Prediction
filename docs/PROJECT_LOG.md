# Project Log: Manufacturing Quality Analysis & Defect Prediction

**Repository**: `AI_Learning`  
**Domain**: Manufacturing Quality Assurance / Machine Learning  
**Author / Learner**: Pair Programming Portfolio Project  
**Status**: Completed, Tested, and Verified  

---

## 1. Executive Summary & Objective

The objective of this project is to build a clean, reproducible, and modular machine learning workflow for predicting quality defects in manufactured metal parts, expanding upon introductory Python, NumPy, Pandas, and Matplotlib data-analysis exercises.

Rather than recreating the repository from scratch or deleting prior learning exercises, this project structures an end-to-end Scikit-learn pipeline, implements unit testing, diagnoses subtle data leakage and label provenance issues, and provides an inference interface for new product measurements.

---

## 2. Step 1: Baseline Repository Inspection & Data Audit

### 2.1 File & Environment Inventory
Prior to making changes, the repository was audited:
* **Pre-existing analysis & visualization scripts**: `manufacturing_data_analysis.py`, `manufacturing_visualizations.py`, `manufacturing_visualisation.py`, `quality_analyser.py`, `pandas_basics.py`, `numpy_basics.py`, `read_csv.py`. All preserved without deletion.
* **Datasets**: `manufacturing_samples.csv` (primary ML dataset), `products.csv` (basic exercise data), `quality_report.csv`.
* **Environment**: Python 3.9.6, Scikit-learn 1.6.1, Pandas 2.3.3, NumPy 2.0.2, Matplotlib 3.9.4, Joblib 1.5.3.

### 2.2 Dataset Breakdown (`manufacturing_samples.csv`)
* **Raw row count**: 26 rows (including 1 duplicate).
* **Columns**: `product_id`, `weight_g`, `length_mm`, `width_mm`, `height_mm`, `tolerance_deviation_mm`, `defective`.
* **Duplicate records**: `PART-107` appeared twice (rows 6 and 7). Deduplication reduced the dataset to **25 unique parts**.
* **Missing values**: Exactly 1 missing value found in `weight_g` (`PART-114`). All other columns were fully populated.
* **Target balance**:
  * Passing QA (`defective == False`): 17 parts (68.0%)
  * Defective (`defective == True`): 8 parts (32.0%)

### 2.3 Mathematical Collinearity Discovery
Analysis revealed a direct mathematical equivalence between dimensions:
$$\text{tolerance\_deviation\_mm} = \text{length\_mm} - 100.00\text{ mm}$$
The nominal target length for parts is $100.00\text{ mm}$. Feeding both `length_mm` and `tolerance_deviation_mm` into a model introduces collinear, redundant features.

---

## 3. Data Quality, Leakage, and Label Provenance Analysis

A critical engineering responsibility in machine learning is assessing whether reported metrics reflect genuine predictive capability or artifacts of synthetic data generation and data leakage.

### 3.1 Feature Exclusion Decisions
1. **`product_id`**: Excluded. Product IDs (e.g. `PART-101`) are administrative identifiers. They carry no physical causal relationship to defect mechanics. Including identifiers causes models to overfit or memorize specific parts.
2. **`tolerance_deviation_mm`**: Excluded from initial predictive features because it is collinear with `length_mm`.
3. **`width_mm` & `height_mm`**: Excluded from initial baseline to focus strictly on physical weight and primary length variation (`weight_g`, `length_mm`).

### 3.2 Label Leakage & Circular Logic Investigation
Cross-referencing the dataset against the rule tested in `manufacturing_data_analysis.py` revealed:
```python
rule_flagged_defect = (
    (df_clean["abs_tolerance_deviation_mm"] > 0.10)
    | (df_clean["weight_g"] < 248.0)
    | (df_clean["weight_g"] > 252.0)
)
```
This rule matches the dataset's `defective` column with **100.0% agreement (25 out of 25 parts)**:
* Every defective part has `weight_g < 244.2` or `> 257.6` and `abs_tolerance_deviation_mm >= 0.22`.
* Every passing part has `248.7 <= weight_g <= 251.4` and `abs_tolerance_deviation_mm <= 0.09`.

**Why this produces misleadingly high scores**:
In real-world manufacturing, quality defects arise from multi-factorial causes: internal porosity, tooling chatter, thermal stress, or surface pitting. A part might have perfect external dimensions but fail an ultrasonic density or tensile test. 

Here, the labels appear to have been generated synthetically using fixed threshold cut-offs on the exact same measurements. When a decision tree scores 100% on this data, it is not demonstrating mastery of manufacturing physics; it is simply reconstructing the deterministic threshold rule used to generate the synthetic labels.

### 3.3 Label Provenance Statement
> **Notice**: The provenance of the defect labels in `manufacturing_samples.csv` is unverified within this repository. There are no associated calibration logs, Coordinate Measuring Machine (CMM) inspection records, or metallurgical test reports. Consequently, the models developed here serve as an educational and algorithmic prototype, and must **not** be claimed as validated for real factory production.

### 3.4 Limitations of a Tiny Dataset ($n = 25$)
* When splitting $80\%$ training ($20$ parts) and $20\%$ testing ($5$ parts):
  * The test set consists of only **3 passing parts and 2 defective parts**.
  * A single incorrect prediction changes the measured test accuracy by **20 percentage points** ($1 / 5 = 0.20$).
  * Confidence intervals on metrics evaluated on $5$ samples are extremely wide, offering low statistical significance.

---

## 4. Machine Learning Workflow & Pipeline Design

### 4.1 Data Splitting Before Preprocessing
To strictly avoid data leakage:
1. `train_test_split(..., test_size=0.2, random_state=42, stratify=y)` is called **before** fitting any imputer or transformer.
2. The `SimpleImputer(strategy="median")` is placed inside a Scikit-learn `Pipeline`.
3. When `pipeline.fit(X_train, y_train)` executes, the median is calculated solely from the $20$ training samples. The test set remains completely unseen during preprocessing fitting.

### 4.2 Model Performance and Comparison

All models were evaluated on the held-out stratified test set ($n = 5$):

| Model | Accuracy | Precision | Recall | F1-Score | Confusion Matrix (Pass / Defect) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Decision Tree** | **1.00 (100%)** | **1.00** | **1.00** | **1.00** | `[[3, 0], [0, 2]]` |
| **Logistic Regression** | **0.60 (60%)** | **0.00** | **0.00** | **0.00** | `[[3, 0], [2, 0]]` |
| **Random Forest (50 trees)**| **1.00 (100%)** | **1.00** | **1.00** | **1.00** | `[[3, 0], [0, 2]]` |

### 4.3 Why Did Logistic Regression Fail While Trees Succeeded?
This comparison demonstrates a foundational machine learning concept:
1. **The "Sandwich" (Bilateral Tolerance) Problem**:
   * In manufacturing, a part is acceptable only when its dimensions fall within a central band: $[99.80\text{ mm}, 100.17\text{ mm}]$.
   * Defects occur if the part is **too short** ($< 99.80\text{ mm}$) OR **too long** ($> 100.17\text{ mm}$).
2. **Linear Hyperplane Limitation**:
   * Logistic Regression fits a single linear equation: $w_1 \cdot \text{weight} + w_2 \cdot \text{length} + b = 0$.
   * A single straight line cannot isolate a center region bounded on both sides by defects. Under default regularization, it predicted all parts as the majority class (`False`), yielding $0.0$ recall for defects.
3. **Decision Tree Strengths**:
   * A Decision Tree uses axis-aligned orthogonal splits:
     $$\text{length\_mm} \le 99.80 \implies \text{Defective}$$
     $$\text{length\_mm} > 100.17 \implies \text{Defective}$$
     $$\text{otherwise} \implies \text{Passed QA}$$
   * It effortlessly partitions the feature space into two outer defect zones and one inner passing zone.

---

## 5. Codebase Architecture

```
AI_Learning/
├── manufacturing_samples.csv          # Primary CSV dataset (25 unique parts)
├── products.csv                      # Exercise data
├── quality_report.csv                # Output of quality analyser
├── requirements.txt                  # Pinned dependencies
├── numpy_basics.py                   # Preserved Python exercise
├── pandas_basics.py                  # Preserved Pandas exercise
├── read_csv.py                       # Preserved CSV reader exercise
├── quality_analyser.py               # Preserved basic QA script
├── manufacturing_data_analysis.py    # Preserved exploratory analysis script
├── manufacturing_visualizations.py   # Preserved Matplotlib visualization script
├── manufacturing_visualisation.py    # Preserved visualization exercise script
├── src/
│   ├── __init__.py
│   ├── pipeline.py                   # Data loader, deduplication, Pipeline factory
│   ├── train.py                      # Training, model comparison, evaluation, joblib serialization
│   └── predict.py                    # Inference function, CLI interface, and demo runner
├── models/
│   └── defect_pipeline.joblib        # Fitted Scikit-learn pipeline (imputer + decision tree)
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py              # 5 unit tests (imputation, structure, deduplication, etc.)
├── docs/
│   └── PROJECT_LOG.md                # Detailed project documentation and logs
└── README.md                         # Portfolio documentation for recruiters and engineers
```

### 5.1 Module Responsibilities
* **`src/pipeline.py`**:
  * `load_and_clean_data(csv_path)`: Loads data, deduplicates rows, validates column presence, and outputs $(X, y)$.
  * `create_pipeline(classifier, random_state)`: Returns Scikit-learn `Pipeline([('imputer', SimpleImputer(strategy='median')), ('classifier', classifier)])`.
* **`src/train.py`**:
  * Trains baseline Decision Tree and comparison models.
  * Prints evaluation metrics and comparison table.
  * Serializes fitted pipeline to `models/defect_pipeline.joblib`.
  * Verifies that reloaded pipeline produces identical predictions.
* **`src/predict.py`**:
  * `predict_defect(weight_g, length_mm, pipeline)`: Predicts defect status, status string (`PASSED QA` / `DEFECTIVE`), and probability.
  * Handles missing `weight_g` using the trained median imputer.
  * CLI support: `python3 src/predict.py --weight 250.2 --length 100.04`.
  * Demo mode: `python3 src/predict.py` (evaluates 6 realistic test scenarios).
* **`tests/test_pipeline.py`**:
  * Unit test suite using standard library `unittest` (6 tests).

---

## 6. Verification and Command Execution Log

The following commands were executed and verified directly in the environment:

### Command 1: Run Unit Tests
```bash
python3 -m unittest discover -s tests -v
```
**Outcome**:
```
test_data_loader_features_and_deduplication (test_pipeline.TestManufacturingPipeline) ... ok
test_missing_value_imputation (test_pipeline.TestManufacturingPipeline) ... ok
test_pipeline_serialization_and_reload (test_pipeline.TestManufacturingPipeline) ... ok
test_pipeline_structure (test_pipeline.TestManufacturingPipeline) ... ok
test_predict_defect_edge_cases (test_pipeline.TestManufacturingPipeline) ... ok
test_predict_defect_function (test_pipeline.TestManufacturingPipeline) ... ok

----------------------------------------------------------------------
Ran 6 tests in 0.080s

OK
```

### Command 2: Model Training and Evaluation
```bash
python3 src/train.py
```
**Outcome**:
* Successfully loaded 25 unique parts from `manufacturing_samples.csv`.
* Stratified 80/20 train/test split completed.
* Evaluated Decision Tree (100% Acc), Logistic Regression (60% Acc), Random Forest (100% Acc).
* Serialized fitted pipeline to `models/defect_pipeline.joblib`.
* Verified reload consistency against in-memory model predictions.

### Command 3: Demo Predictions
```bash
python3 src/predict.py
```
**Outcome**:
* Nominal compliant part (`250.2 g`, `100.04 mm`) $\implies$ `PASSED QA` (Defect Prob: 0%)
* In-spec slightly light part (`249.4 g`, `99.95 mm`) $\implies$ `PASSED QA` (Defect Prob: 0%)
* Oversized length defect (`262.3 g`, `100.41 mm`) $\implies$ `DEFECTIVE` (Defect Prob: 100%)
* Undersized length defect (`239.1 g`, `99.68 mm`) $\implies$ `DEFECTIVE` (Defect Prob: 100%)
* Missing weight reading (`None`, `100.03 mm`) $\implies$ `PASSED QA` (Imputed weight used, Defect Prob: 0%)
* Overweight anomaly (`275.0 g`, `100.35 mm`) $\implies$ `DEFECTIVE` (Defect Prob: 100%)

### Command 4: Custom Single-Part CLI Predictions
```bash
python3 src/predict.py --weight 250.2 --length 100.04
python3 src/predict.py --weight 262.3 --length 100.41
python3 src/predict.py --length 100.03
```
**Outcome**: All CLI options executed with accurate formatted inspection output.

### Command 5: Verification of Existing Learning Scripts
* `python3 quality_analyser.py`: Succeeded without regression.
* `python3 manufacturing_data_analysis.py`: Succeeded without regression.
* `python3 manufacturing_visualizations.py`: Succeeded without regression.

---

## 7. Next Steps & Recommended Production Path

For transitioning this portfolio prototype toward production-grade quality inspection:
1. **Acquire Empirical Inspection Data**: Collect at least $500$–$5,000$ real parts with genuine CMM measurements and independent destructive/metallurgical testing logs.
2. **Incorporate Multi-Sensor Dimensions**: Add `width_mm`, `height_mm`, surface finish ($R_a$), and machine vibration telemetry.
3. **Continuous Integration**: Add GitHub Actions workflow to run `unittest discover` on pull requests.
4. **Drift Monitoring**: Monitor dimensional distribution shift over time to detect CNC tool wear before defects occur.
