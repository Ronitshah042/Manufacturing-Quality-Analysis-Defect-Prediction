# Manufacturing Quality Analysis & Defect Prediction

A reproducible, modular Python and Machine Learning project analyzing fabrication tolerances and predicting defect classifications in precision manufacturing parts.

Built as an applied portfolio project connecting data analysis, Scikit-learn pipelines, and manufacturing quality control (QA/QC).

---

## 📌 Project Overview

In precision manufacturing (CNC machining, aerospace casting, injection molding), parts must adhere to strict dimensional tolerances and mass specifications. 

This project establishes an end-to-end data science and machine learning workflow:
1. **Exploratory Data Analysis & Visualization**: Inspecting dimension distributions, defect rates, and tolerance deviations using Pandas and Matplotlib.
2. **Data Cleaning & Leakage Prevention**: Eliminating duplicates, imputing missing sensor values, and avoiding collinear feature leakage.
3. **Reproducible ML Pipeline**: Building a Scikit-learn `Pipeline` that isolates preprocessing strictly to training splits.
4. **Model Comparison & Diagnostics**: Evaluating a baseline `DecisionTreeClassifier` against `LogisticRegression` and `RandomForestClassifier`.
5. **Inference & CLI**: Providing a command-line tool and Python interface to inspect new fabrication samples.

---

## 🔬 Dataset & Feature Engineering

The primary dataset (`manufacturing_samples.csv`) consists of physical measurements from fabricated mechanical parts:

| Column | Unit / Type | Description | Included in ML? |
| :--- | :--- | :--- | :---: |
| `product_id` | String | Serial identifier (e.g. `PART-101`) | ❌ **Excluded** (Risk of memorization/overfitting) |
| `weight_g` | Grams (float) | Part mass (Nominal target: 250.0 g) | ✅ **Included** (Initial predictive feature) |
| `length_mm` | Millimeters (float) | Part length (Nominal target: 100.00 mm) | ✅ **Included** (Initial predictive feature) |
| `tolerance_deviation_mm` | Millimeters (float) | Deviation from nominal length | ❌ **Excluded** (Collinear: equals `length_mm - 100.0`) |
| `width_mm`, `height_mm` | Millimeters (float) | Part cross-section dimensions | ⏸️ Reserved for multi-feature iterations |
| `defective` | Boolean | QA status (`False` = Passed, `True` = Defective) | 🎯 **Target Label** |

* **Total unique parts**: 25 (after deduplicating `PART-107`).
* **Missing values**: 1 missing `weight_g` value (`PART-114`), handled via median imputation inside the pipeline.
* **Class distribution**: 17 Passed QA (68%), 8 Defective (32%).

---

## 📊 Model Performance & Insights

All models were evaluated on an unseen stratified test set ($20\%$ holdout, $n = 5$):

| Model Architecture | Accuracy | Precision | Recall | F1-Score | Confusion Matrix (Pass / Defect) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Decision Tree** | **1.00 (100%)** | **1.00** | **1.00** | **1.00** | `[[3, 0], [0, 2]]` |
| **Logistic Regression** | **0.60 (60%)** | **0.00** | **0.00** | **0.00** | `[[3, 0], [2, 0]]` |
| **Random Forest (50 trees)** | **1.00 (100%)** | **1.00** | **1.00** | **1.00** | `[[3, 0], [0, 2]]` |

### 🧠 Key Engineering Takeaways

1. **Why Logistic Regression Failed (0% Recall)**:
   * Manufacturing tolerances reject parts that are **too small** or **too large**. Acceptable parts reside in an intermediate "safe band" ($99.80\text{ mm} \le \text{length} \le 100.17\text{ mm}$).
   * A linear classifier draws a single linear boundary ($w^T x + b = 0$). It cannot isolate a center region surrounded on both sides by defects without non-linear feature transformations.
2. **Why Decision Trees Excel on Bilateral Tolerances**:
   * Decision trees naturally perform orthogonal, multi-threshold splits (`length <= 99.80` or `length > 100.17` $\implies$ Defective), isolating the tolerance window effortlessly.
3. **Data Leakage & Preprocessing Order**:
   * Preprocessing (`SimpleImputer(strategy="median")`) is encapsulated inside a Scikit-learn `Pipeline` and fitted **only** on the training set after splitting. Fitting before splitting would leak summary statistics from test data into training data.

---

## ⚠️ Data Quality & Limitations

* **Label Provenance**: The defect labels in `manufacturing_samples.csv` follow a deterministic threshold rule (`|tolerance_deviation| > 0.10` or `weight < 248` or `weight > 252`). The models reproduce this threshold rule, but have not been validated against physical Coordinate Measuring Machine (CMM) or ultrasonic crack detection data.
* **Tiny Dataset ($n = 25$)**: With only 5 test samples, each part represents a $20\%$ swing in accuracy. High scores must be interpreted as algorithmic verification, not production certification.

For a detailed analysis of label leakage and provenance, see [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md).

---

## 🛠️ Project Structure

```
AI_Learning/
├── manufacturing_samples.csv          # Primary manufacturing dataset
├── products.csv                      # Preserved practice data
├── quality_report.csv                # Preserved analysis output
├── requirements.txt                  # Python dependencies
│
├── src/                              # Modular production code
│   ├── __init__.py
│   ├── pipeline.py                   # Data loader, deduplication, Pipeline factory
│   ├── train.py                      # Train, evaluate, compare models, serialize pipeline
│   └── predict.py                    # Inference module, CLI, and demonstration runner
│
├── models/
│   └── defect_pipeline.joblib        # Fitted Scikit-learn pipeline (imputer + decision tree)
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py              # Unit tests for preprocessing, pipeline, and prediction
│
├── docs/
│   └── PROJECT_LOG.md                # Comprehensive project log, decisions, and leakage analysis
│
├── manufacturing_data_analysis.py    # Preserved exploratory analysis & Pandas exercises
├── manufacturing_visualizations.py   # Preserved box plots and scatter plot generation
├── manufacturing_visualisation.py    # Preserved visualization exercises
├── quality_analyser.py               # Preserved basic QA script
├── numpy_basics.py                   # Preserved NumPy fundamentals
├── pandas_basics.py                  # Preserved Pandas fundamentals
└── read_csv.py                       # Preserved CSV reader exercise
```

---

## 🚀 Quickstart & Usage

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Unit Tests
Execute the automated test suite covering imputation, deduplication, serialization, and predictions:
```bash
python3 -m unittest discover -s tests -v
```

### 3. Train & Evaluate Models
Train the baseline pipeline, compare against Logistic Regression and Random Forest, and save the fitted model:
```bash
python3 src/train.py
```

### 4. Run Defect Predictions
Predict quality status for new parts using the serialized pipeline:

**Run the built-in demonstration suite**:
```bash
python3 src/predict.py
```

**Predict for specific part measurements via CLI**:
```bash
# Nominal compliant part
python3 src/predict.py --weight 250.2 --length 100.04

# Out-of-spec defective part
python3 src/predict.py --weight 262.3 --length 100.41

# Part with missing weight reading (automatically imputed)
python3 src/predict.py --length 100.03
```

---

## 📈 Exploratory Visualizations

To regenerate the dimensional box plots and quality inspection scatter plots:
```bash
python3 manufacturing_visualizations.py
```
* `manufacturing_dimension_boxplots.png`: 4-panel distribution comparison for weight, length, width, and height.
* `manufacturing_scatter_plot.png`: Multi-variable inspection map showing tolerance boundaries and weight limits.

---

## 📋 Tools & Technologies

* **Language**: Python 3.9+
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-learn (Pipelines, Decision Trees, SimpleImputer, Metrics)
* **Visualization**: Matplotlib
* **Serialization**: Joblib
* **Testing**: Python standard library `unittest`