import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Define directory
cwd = Path.cwd()
directory = cwd / "Project_Data_EE4C12_EPE_PQD/SNR_noiseless"

# --- Step 1: Compute per-file means and stds (ignoring first column) ---
file_feature_means = {}
file_feature_stds = {}

for file in directory.iterdir():
    if file.suffix.lower() == ".csv":
        df = pd.read_csv(file)
        # Drop first column (index)
        df = df.iloc[:, 1:]
        # Compute per-feature stats
        file_feature_means[file.name] = df.mean(numeric_only=True)
        file_feature_stds[file.name] = df.std(numeric_only=True)

# Convert dictionaries to DataFrames
means_df = pd.DataFrame(file_feature_means)
stds_df = pd.DataFrame(file_feature_stds)

# --- Step 2: Compute variability across files ---
mean_variability = means_df.std(axis=1)
std_variability = stds_df.std(axis=1)

comparison = pd.DataFrame({
    'Mean Variability (across files)': mean_variability,
    'Std Variability (across files)': std_variability
})

# --- Step 3: Visualize using heatmaps ---

# Feature means across files
plt.figure(figsize=(12, 6))
sns.heatmap(means_df, cmap="viridis", annot=False)
plt.title("Feature Means Across Files")
plt.xlabel("Files")
plt.ylabel("Features")
plt.tight_layout()
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(fontsize=12)

plt.show()

# Feature stds across files
plt.figure(figsize=(12, 6))
sns.heatmap(stds_df, cmap="magma", annot=False)
plt.title("Feature Standard Deviations Across Files")
plt.xlabel("Files")
plt.ylabel("Features")
plt.tight_layout()
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Variability summary
plt.figure(figsize=(6, 8))
sns.heatmap(comparison, cmap="coolwarm", annot=True, fmt=".3f")
plt.title("Feature Variability Across Files")
plt.xlabel("Metric")
plt.ylabel("Feature")
plt.tight_layout()
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(fontsize=12)
plt.show()

#######

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Define directory
cwd = Path.cwd()
directory = cwd / "Project_Data_EE4C12_EPE_PQD/SNR_noiseless"

# --- Step 1: Compute per-file means and stds (ignoring first column) ---
file_feature_means = {}
file_feature_stds = {}

for file in directory.iterdir():
    if file.suffix.lower() == ".csv":
        df = pd.read_csv(file)

        # Drop first column (index or ID)
        df = df.iloc[:, 1:]

        # Remove unwanted columns (those containing 'min' or 'no. pt near 0')
        drop_cols = [c for c in df.columns if 'min' in c.lower() or 'no. pt near 0' in c.lower()]
        df = df.drop(columns=drop_cols, errors='ignore')

        # Compute per-feature stats
        file_feature_means[file.name] = df.mean(numeric_only=True)
        file_feature_stds[file.name] = df.std(numeric_only=True)

# Convert dictionaries to DataFrames
means_df = pd.DataFrame(file_feature_means)
stds_df = pd.DataFrame(file_feature_stds)

# --- Step 2: Compute variability across files ---
mean_variability = means_df.std(axis=1)
std_variability = stds_df.std(axis=1)

comparison = pd.DataFrame({
    'Mean Variability (across files)': mean_variability,
    'Std Variability (across files)': std_variability
})

# --- Step 3: Visualize using heatmaps ---

# Feature means across files
plt.figure(figsize=(12, 6))
sns.heatmap(means_df, cmap="viridis", annot=False)
plt.title("Feature Means Across Files")
plt.xlabel("Files")
plt.ylabel("Features")
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.show()

# Feature stds across files
plt.figure(figsize=(12, 6))
sns.heatmap(stds_df, cmap="magma", annot=False)
plt.title("Feature Standard Deviations Across Files")
plt.xlabel("Files")
plt.ylabel("Features")
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(rotation=0,fontsize=12)
plt.tight_layout()
plt.show()

# Variability summary
plt.figure(figsize=(6, 8))
sns.heatmap(comparison, cmap="coolwarm", annot=True, fmt=".3f")
plt.title("Feature Variability Across Files")
plt.xlabel("Metric")
plt.ylabel("Feature")
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(rotation=0,fontsize=12)
plt.tight_layout()
plt.show()
