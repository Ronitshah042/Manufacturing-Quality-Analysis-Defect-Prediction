"""
Manufacturing Visualizations with Matplotlib
============================================
Lesson: Visualizing physical dimensions and quality metrics to detect defect patterns.

Dataset: manufacturing_samples.csv
"""

import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. Load and Prepare Cleaned Data
# ==========================================
# Load dataset
df = pd.read_csv("manufacturing_samples.csv")

# Remove duplicate records
df_clean = df.drop_duplicates().copy()

# Feature engineering: absolute tolerance deviation
df_clean["abs_tolerance_deviation_mm"] = df_clean["tolerance_deviation_mm"].abs()

print(f"Data ready for visualization: {len(df_clean)} unique parts.")

# Separate into passing and defective groups for plotting
passed = df_clean[df_clean["defective"] == False]
defective = df_clean[df_clean["defective"] == True]

# ==========================================
# STEP 1: Box Plots for All 4 Physical Dimensions
# ==========================================
# Create a 2x2 grid of subplots (figsize: width=12, height=9)
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Physical Dimension Distributions: Passed vs. Defective", fontsize=15, fontweight="bold")

# Define the 4 dimensions to compare, their titles, and units
metrics = [
    ("weight_g", "Weight Distribution", "Weight (g)", axes[0, 0]),
    ("length_mm", "Length Distribution", "Length (mm)", axes[0, 1]),
    ("width_mm", "Width Distribution", "Width (mm)", axes[1, 0]),
    ("height_mm", "Height Distribution", "Height (mm)", axes[1, 1])
]

for col, title, ylabel, ax in metrics:
    # Prepare data, dropping missing values if any
    data_to_plot = [
        passed[col].dropna(),
        defective[col].dropna()
    ]
    
    # Create the box plot
    bp = ax.boxplot(
        data_to_plot,
        tick_labels=["Passed QA", "Defective"],
        patch_artist=True,  # Allows filling boxes with color
        boxprops=dict(facecolor="#e8f4f8", color="#1f77b4"),
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(marker="o", color="red", alpha=0.6)
    )
    
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()

# Save the box plots figure
boxplot_output = "manufacturing_dimension_boxplots.png"
plt.savefig(boxplot_output, dpi=150, bbox_inches="tight")
print(f"Step 1 Complete: Box plots saved to {boxplot_output}")
plt.close()
# ==========================================
# STEP 2: Multi-Variable Quality Scatter Plot
# ==========================================
# Create a dedicated figure for the scatter plot
plt.figure(figsize=(9, 6))

# 1. Plot passing parts in blue circles
plt.scatter(
    passed["weight_g"],
    passed["abs_tolerance_deviation_mm"],
    color="#1f77b4",
    marker="o",
    s=70,
    alpha=0.75,
    edgecolors="black",
    label="Passed QA"
)

# 2. Plot defective parts in red triangles
plt.scatter(
    defective["weight_g"],
    defective["abs_tolerance_deviation_mm"],
    color="#d62728",
    marker="^",
    s=90,
    alpha=0.85,
    edgecolors="black",
    label="Defective"
)

# 3. Add visual quality threshold limits
# Tolerance deviation limit at 0.10 mm
plt.axhline(
    y=0.10, 
    color="darkorange", 
    linestyle="--", 
    linewidth=1.5, 
    label="Tolerance Boundary (0.10 mm)"
)

# Weight limits at 248.0 g and 252.0 g
plt.axvline(x=248.0, color="gray", linestyle=":", linewidth=1.2, label="Weight Spec Limits (248g / 252g)")
plt.axvline(x=252.0, color="gray", linestyle=":", linewidth=1.2)

# 4. Add labels, units, title, legend, and grid
plt.title("Quality Inspection Map: Weight vs. Absolute Tolerance Deviation", fontsize=13, fontweight="bold")
plt.xlabel("Part Weight (g)", fontsize=11)
plt.ylabel("Absolute Tolerance Deviation (mm)", fontsize=11)
plt.legend(loc="upper left", framealpha=0.9)
plt.grid(True, linestyle=":", alpha=0.6)

# 5. Save the scatter plot
scatter_output = "manufacturing_scatter_plot.png"
plt.savefig(scatter_output, dpi=150, bbox_inches="tight")
print(f"Step 2 Complete: Scatter plot saved to {scatter_output}")
plt.close()
