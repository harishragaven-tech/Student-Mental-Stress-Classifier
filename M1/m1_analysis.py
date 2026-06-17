import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "students_mental_health_survey.csv"
OUTPUT_FILE = BASE_DIR / "processed_student_stress.csv"


def main():

    # ==========================================
    # TASK 1 - LOAD DATASET
    # ==========================================

    # Load the dataset
    df = pd.read_csv(DATA_FILE)

    print("\n==============================")
    print("FIRST 5 ROWS OF DATASET")
    print("==============================")
    print(df.head())

    print("\n==============================")
print("FIRST 5 ROWS OF DATASET")
print("==============================")
print(df.head())

print("\n==============================")
print("DATASET SHAPE")
print("==============================")
print(df.shape)

print("\n==============================")
print("MISSING VALUES")
print("==============================")
print(df.isnull().sum())

df["CGPA"] = df["CGPA"].fillna(df["CGPA"].mean())
df["Substance_Use"] = df["Substance_Use"].fillna("Unknown")

print("\n==============================")
print("MISSING VALUES")
print("==============================")
print(df.isnull().sum())

print("\n==============================")
print("DATA TYPES")
print("==============================")
print(df.dtypes)

print("\n==============================")
print("DATASET INFO")
print("==============================")
df.info()

# ==========================================
# LABEL ENCODING
# ==========================================
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
        "Residence_Type",
    ]

for col in categorical_columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))

print("\n==============================")
print("ENCODED DATASET (FIRST 5 ROWS)")
print("==============================")
print(df.head())

sns.countplot(x="Sleep_Quality", hue="Stress_Level", data=df)
plt.title("Sleep Quality vs Stress Level")
plt.show()

sns.boxplot(x="Stress_Level", y="Anxiety_Score", data=df)
plt.title("Anxiety Score vs Stress Level")
plt.show()

sns.boxplot(x="Stress_Level", y="CGPA", data=df)
plt.title("CGPA vs Stress Level")
plt.show()

df.to_csv(OUTPUT_FILE, index=False)
print(f"Processed dataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()