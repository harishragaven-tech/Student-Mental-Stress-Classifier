# Import libraries

import pandas as pd

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "M1" / "processed_student_stress.csv"
OUTPUT_DIR = BASE_DIR / "M2" / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)
# Load dataset

df = pd.read_csv(DATA_FILE)

print("Dataset Loaded Successfully")
# Stress Level Distribution

plt.figure(figsize=(8,5))

sns.countplot(x="Stress_Level", data=df)

plt.title("Stress Level Distribution")

plt.xlabel("Stress Level")

plt.ylabel("Number of Students")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "stress_level_distribution.png")
plt.close()

plt.figure(figsize=(8,5))

sns.boxplot(
    x="Stress_Level",
    y="Anxiety_Score",
    data=df
)

plt.title("Anxiety Score vs Stress Level")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "anxiety_score_vs_stress.png")
plt.close()

# CGPA vs Stress Level

plt.figure(figsize=(8,5))

sns.boxplot(
    x="Stress_Level",
    y="CGPA",
    data=df
)

plt.title("CGPA vs Stress Level")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cgpa_vs_stress.png")
plt.close()
# Financial Stress vs Stress Level

plt.figure(figsize=(8,5))

sns.boxplot(
    x="Stress_Level",
    y="Financial_Stress",
    data=df
)

plt.title("Financial Stress vs Stress Level")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "financial_stress_vs_stress.png")
plt.close()

# Correlation Heatmap

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(),
    annot=True
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.close()
