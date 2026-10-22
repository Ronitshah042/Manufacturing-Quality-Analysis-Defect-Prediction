"""
Manufacturing Data Analysis with pandas
=======================================
This script explores quality control measurements from a manufacturing line.

Dataset: manufacturing_samples.csv
Columns:
- product_id: Unique identifier for each fabricated part (e.g., PART-101)
- weight_g: Measured part weight in grams (Nominal target: 250.0 g)
- length_mm: Measured length in mm (Nominal target: 100.00 mm)
- width_mm: Measured width in mm (Nominal target: 50.00 mm)
- height_mm: Measured height in mm (Nominal target: 25.00 mm)
- tolerance_deviation_mm: Drift from nominal dimension specification
- defective: Boolean flag (True = failed QA, False = passed QA)
"""

import pandas as pd

# ==========================================
# 1. Loading and First Inspection
# ==========================================
print("=== 1. Loading Dataset ===")
df = pd.read_csv("manufacturing_samples.csv")

# .head(n) lets us peek at the first n rows
print("\nFirst 5 rows of manufacturing data:")
print(df.head())

# .shape returns (rows, columns)
print(f"\nDataset shape: {df.shape[0]} rows and {df.shape[1]} columns")

# .info() shows column types and non-null counts
print("\nDataFrame summary:")
df.info()


# ==========================================
# 2. Data Cleaning: Missing Values & Duplicates
# ==========================================
print("\n=== 2. Missing Value and Duplicate Checks ===")

# .isna().sum() counts how many NaN / null values are in each column
missing_counts = df.isna().sum()
print("\nMissing values per column:")
print(missing_counts)

# .duplicated().sum() counts identical repeated rows
duplicate_count = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")

# Remove duplicate rows to keep our analysis clean
# keep="first" keeps the first occurrence and drops later duplicates
df_clean = df.drop_duplicates().copy()
print(f"Shape after removing duplicates: {df_clean.shape}")


# ==========================================
# 3. Descriptive Statistics
# ==========================================
print("\n=== 3. Descriptive Statistics ===")

# .describe() computes count, mean, std, min, 25%, 50%, 75%, and max
print("\nSummary statistics for numeric columns:")
print(df_clean.describe())

# We can also compute statistics on individual columns:
avg_weight = df_clean["weight_g"].mean()
std_weight = df_clean["weight_g"].std()
max_deviation = df_clean["tolerance_deviation_mm"].abs().max()

print(f"\nAverage Weight: {avg_weight:.2f} g")
print(f"Weight Standard Deviation: {std_weight:.2f} g")
print(f"Maximum Absolute Tolerance Deviation: {max_deviation:.2f} mm")


# ==========================================
# 4. Conditional Filtering
# ==========================================
print("\n=== 4. Conditional Filtering ===")

# Example A: Filter parts with tolerance deviation magnitude > 0.15 mm
high_deviation = df_clean[df_clean["tolerance_deviation_mm"].abs() > 0.15]
print(f"\nParts with tolerance deviation > 0.15 mm (Total: {len(high_deviation)}):")
print(high_deviation[["product_id", "tolerance_deviation_mm", "defective"]])

# Example B: Multiple conditions using & (AND) or | (OR)
# Parts that are defective AND heavier than 255 g
heavy_defective = df_clean[(df_clean["defective"] == True) & (df_clean["weight_g"] > 255)]
print(f"\nDefective parts weighing > 255 g (Total: {len(heavy_defective)}):")
print(heavy_defective[["product_id", "weight_g", "defective"]])


# ==========================================
# 5. Grouping and Aggregation (groupby)
# ==========================================
print("\n=== 5. Grouping by Quality Status (Defective vs Passed) ===")

# Grouping by 'defective' allows us to compare average dimensions between good and bad parts
# numeric_only=True ensures we only take the mean of numerical measurements
quality_comparison = df_clean.groupby("defective")[["weight_g", "length_mm", "tolerance_deviation_mm"]].mean()
print("\nAverage measurements grouped by defective flag:")
print(quality_comparison)


# =====================================================================
# 6. YOUR PRACTICE EXERCISES (Solve these below!)
# =====================================================================
"""
Exercise 1:
-----------
Write code to calculate the overall defect rate percentage for df_clean.
Formula: (defective_count / total_count) * 100
Hint: Use (df_clean["defective"] == True).sum() or df_clean["defective"].mean() * 100

Exercise 2:
-----------
Find all parts that are NOT defective, but have an absolute tolerance deviation greater than 0.05 mm.
Print their product_id and tolerance_deviation_mm.

Exercise 3:
-----------
Using .groupby("defective"), calculate the minimum, average, and maximum 'weight_g'
for both defective and non-defective parts.
Hint: Look at .agg(['min', 'mean', 'max'])
"""

print("\n=== Your Exercise Solutions Below ===")
# --- Write your solution for Exercise 1 here ---


# --- Write your solution for Exercise 2 here ---


# --- Write your solution for Exercise 3 here ---

defective_count = df_clean["defective"].sum()
total_count = len(df_clean)

defect_rate = (defective_count / total_count) * 100

print(f"Number of defective parts: {defective_count}")
print(f"Total parts: {total_count}")
print(f"Defect rate: {defect_rate:.2f}%")

high_dev_parts = df_clean[df_clean["tolerance_deviation_mm"].abs() > 0.15]
print(high_dev_parts[["product_id", "tolerance_deviation_mm", "defective"]])

print(df_clean.groupby("defective")["weight_g"].agg(["min", "mean", "max"]))
# 1. Create the new absolute tolerance deviation feature
df_clean["abs_tolerance_deviation_mm"] = df_clean["tolerance_deviation_mm"].abs()

# 2. Select columns to compare
feature_cols = [
    "weight_g",
    "length_mm",
    "width_mm",
    "height_mm",
    "tolerance_deviation_mm",
    "abs_tolerance_deviation_mm"
]

# 3. Group by 'defective' and compute mean, std, min, and max
quality_distribution = df_clean.groupby("defective")[feature_cols].agg(["mean", "std", "min", "max"])

# .T (transpose) flips rows and columns to make the multi-column table much easier to read!
print("\n=== Detailed Distribution Comparison (Defective vs Passed) ===")
print(quality_distribution.T)

rule_flagged_defect = (
    (df_clean["abs_tolerance_deviation_mm"] > 0.10)
    | (df_clean["weight_g"] < 248.0)
    | (df_clean["weight_g"] > 252.0)
)

print ("Total cases flagged by rule: ", rule_flagged_defect.sum ())

# 1. Compare rule predictions against the actual defective labels
matches = (rule_flagged_defect == df_clean["defective"]).sum()
total_parts = len(df_clean)
agreement_rate = (matches / total_parts) * 100

print(f"Rule Agreement: {matches} / {total_parts} ({agreement_rate:.1f}%)")

# 2. View a cross-tabulation table (Actual vs. Rule-flagged)
print("\n=== Inspection Agreement Matrix ===")
print(pd.crosstab(
    rule_flagged_defect, 
    df_clean["defective"], 
    rownames=["Rule Flagged"], 
    colnames=["Actual Defect"]
))
