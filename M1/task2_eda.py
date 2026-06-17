# Import pandas library
import pandas as pd

# Import dataset path library
from pathlib import Path
# Dataset location

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "processed_student_stress.csv"
# Load processed dataset

df = pd.read_csv(DATA_FILE)
print("\n========================================")
print("FIRST 5 ROWS")
print("========================================")

print(df.head())
print("\n========================================")
print("MISSING VALUES")
print("========================================")

print(df.isnull().sum())
print("\n========================================")
print("DUPLICATE ROWS")
print("========================================")

print(df.duplicated().sum())
print("\n========================================")
print("DATASET SHAPE")
print("========================================")

print(df.shape)

print("\n========================================")
print("DATA TYPES")
print("========================================")

print(df.dtypes)

print("\n========================================")
print("STATISTICAL SUMMARY")
print("========================================")

print(df.describe())

print("\n========================================")
print("STRESS LEVEL COUNTS")
print("========================================")

print(df["Stress_Level"].value_counts())

print("\n========================================")
print("GENDER COUNTS")
print("========================================")

print(df["Gender"].value_counts())

print("\n========================================")
print("RESIDENCE TYPE COUNTS")
print("========================================")

print(df["Residence_Type"].value_counts())

print("\n========================================")
print("GROUPED ANALYSIS BY STRESS LEVEL")
print("========================================")

print(
    df.groupby("Stress_Level")[
        [
            "CGPA",
            "Anxiety_Score",
            "Depression_Score",
            "Financial_Stress"
        ]
    ].mean()
)