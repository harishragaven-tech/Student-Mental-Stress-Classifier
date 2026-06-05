# Import required libraries

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
# Define dataset paths

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "M1" / "students_mental_health_survey.csv"

OUTPUT_FILE = BASE_DIR / "M1" / "processed_student_stress.csv"
# Load dataset

df = pd.read_csv(INPUT_FILE)

print("\nDATASET LOADED SUCCESSFULLY")
print("\n========================================")
print("MISSING VALUES BEFORE CLEANING")
print("========================================")

print(df.isnull().sum())
df["CGPA"] = df["CGPA"].fillna(
    df["CGPA"].mean()
)
df["Substance_Use"] = df["Substance_Use"].fillna(
    "Unknown"
)
duplicates = df.duplicated().sum()

print("\nDuplicate Rows Found:", duplicates)

df.drop_duplicates(inplace=True)
print("\n========================================")
print("MISSING VALUES AFTER CLEANING")
print("========================================")

print(df.isnull().sum())
categorical_columns = [
    "Course",
    "Gender",
    "Sleep_Quality",
    "Physical_Activity",
    "Diet_Quality",
    "Social_Support",
    "Relationship_Status",
    "Substance_Use",
    "Counseling_Service_Use",
    "Family_History",
    "Chronic_Illness",
    "Extracurricular_Involvement",
    "Residence_Type"
]

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(
        df[col].astype(str)
    )

print("\n========================================")
print("PROCESSED DATASET")
print("========================================")

print(df.head())

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nProcessed dataset saved successfully.")