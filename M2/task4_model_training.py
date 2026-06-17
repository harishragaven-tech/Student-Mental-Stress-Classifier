# Import required libraries

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "M1" / "processed_student_stress.csv"

df = pd.read_csv(DATA_FILE)

print("Dataset Loaded Successfully")
y = df["Stress_Level"]
X = df.drop("Stress_Level", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed")

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n===================================")
print("MODEL ACCURACY")
print("===================================")

print(f"Accuracy: {accuracy:.2f}")
print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

print(confusion_matrix(y_test, y_pred))

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")

print(
    classification_report(
        y_test,
        y_pred
    )
)