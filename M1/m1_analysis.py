import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\ragav\Downloads\students_mental_health_survey.csv")

print(df.head())
print(df.shape)
print(df.isnull().sum())

df["CGPA"] = df["CGPA"].fillna(
    df["CGPA"].mean()
)

df["Substance_Use"] = df["Substance_Use"].fillna(
    "Unknown"
)
print(df.isnull().sum())
print(df.dtypes)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

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
    df[col] = le.fit_transform(df[col])

print(df.head())

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x="Sleep_Quality", hue="Stress_Level", data=df)

plt.title("Sleep Quality vs Stress Level")
plt.show()

sns.boxplot(
    x="Stress_Level",
    y="Anxiety_Score",
    data=df
)

plt.title("Anxiety Score vs Stress Level")
plt.show()

sns.boxplot(
    x="Stress_Level",
    y="CGPA",
    data=df
)

plt.title("CGPA vs Stress Level")
plt.show()

df.to_csv(
    "processed_student_stress.csv",
    index=False
)

print("Processed dataset saved successfully.")

import os
print("file saved successfully!")