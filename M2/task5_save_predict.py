# Import libraries

import pandas as pd
import pickle

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "M1" / "processed_student_stress.csv"
MODEL_FILE = BASE_DIR / "model.pkl"

# Load dataset

df = pd.read_csv(DATA_FILE)

print("Dataset Loaded Successfully")

# Features

X = df.drop("Stress_Level", axis=1)

# Target

y = df["Stress_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train, y_train)

print("Model Trained Successfully")

with open(MODEL_FILE, "wb") as file:
    pickle.dump(model, file)

print(f"Model saved as {MODEL_FILE.name}")

with open(MODEL_FILE, "rb") as file:
    loaded_model = pickle.load(file)

print("Model Loaded Successfully")

sample_student = X.iloc[[0]].copy()

prediction = loaded_model.predict(sample_student)

print("\nPredicted Stress Level:")
print(prediction[0])