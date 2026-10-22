"""
Manufacturing Data Visualization with Matplotlib
================================================
This script visualizes quality inspection data to help engineers
spot defective patterns, dimensional clustering, and outliers.

Dataset: manufacturing_samples.csv
"""

import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. Load Clean Data
# ==========================================
df = pd.read_csv("manufacturing_samples.csv").drop_duplicates()
df["abs_tolerance_deviation_mm"] = df["tolerance_deviation_mm"].abs()

print(f"Loaded {len(df)} parts for visualization.")

# ==========================================
# 2. Scatter Plot: Weight vs. Absolute Tolerance Deviation
# ==========================================
# Split the dataset into passing and defective groups
passed = df[df["defective"] == False]
defective = df[df["defective"] == True]

# Create a figure and axis
plt.figure(figsize=(9, 6))

# Plot passing parts in blue circles
plt.scatter(
    passed["weight_g"],
    passed["abs_tolerance_deviation_mm"],
    color="blue",
    label="Passed QA",
    alpha=0.7,
    s=70,
    edgecolors="black"
)

# Plot defective parts in red triangles
plt.scatter(
    defective["weight_g"],
    defective["abs_tolerance_deviation_mm"],
    color="red",
    marker="^",
    label="Defective",
    alpha=0.8,
    s=90,
    edgecolors="black"
)

# Add tolerance warning boundary line (e.g. at 0.10 mm)
plt.axhline(
    y=0.10,
    color="darkorange",
    linestyle="--",
    linewidth=1.5,
    label="Tolerance Threshold (0.10 mm)"
)

# Add titles and labels with physical units
plt.title("Manufacturing Quality: Weight vs. Absolute Tolerance Deviation", fontsize=13, fontweight="bold")
plt.xlabel("Weight (g)", fontsize=11)
plt.ylabel("Absolute Tolerance Deviation (mm)", fontsize=11)
plt.legend(loc="upper left")
plt.grid(True, linestyle=":", alpha=0.6)

# Save the plot as an image file
output_image = "manufacturing_quality_scatter.png"
plt.savefig(output_image, dpi=150, bbox_inches="tight")
print(f"Scatter plot successfully saved to {output_image}")
plt.close()

# =====================================================================
# 3. YOUR EXERCISE: Box Plot of Weight Distributions
# =====================================================================
"""
Task:
-----
Create a box plot comparing the distribution of 'weight_g' between
passing parts (defective == False) and defective parts (defective == True).

Why Box Plots in Manufacturing?
- The box shows the middle 50% of parts (Interquartile Range / IQR).
- The horizontal line inside the box marks the median (50th percentile).
- The 'whiskers' extend to minimum and maximum non-outlier limits.
- Any dots beyond the whiskers represent extreme process outliers!

Hint:
plt.figure(figsize=(7, 5))
# Remember to drop NA if any missing values exist in weight_g:
plt.boxplot(
    [passed["weight_g"].dropna(), defective["weight_g"].dropna()],
    tick_labels=["Passed QA", "Defective"]
)
plt.title("Part Weight Distribution: Passed vs. Defective", fontweight="bold")
plt.ylabel("Weight (g)")
plt.grid(True, linestyle=":", alpha=0.5)
plt.savefig("manufacturing_weight_boxplot.png", dpi=150, bbox_inches="tight")
print("Box plot saved to manufacturing_weight_boxplot.png")
"""

# --- Write your solution below ---


