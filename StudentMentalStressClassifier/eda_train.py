import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ---------- STEP 1: Load data ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "students_mental_health_survey_cleaned.csv"))

# ---------- STEP 2: Bin Stress_Level into 3 classes: Low/Moderate/High ----------
def bin_stress(x):
    if x <= 1:
        return 0  # Low
    elif x <= 3:
        return 1  # Moderate
    else:
        return 2  # High

df["Stress_Level"] = df["Stress_Level"].apply(bin_stress)

# ---------- STEP 3: Select ONLY the 8 features used in the 10-question survey ----------
FEATURES = [
    "Sleep_Quality", "Anxiety_Score", "Physical_Activity", "Diet_Quality",
    "Social_Support", "Financial_Stress", "Substance_Use", "Counseling_Service_Use"
]

# ---------- STEP 4: Label Encoding (only categorical features) ----------
categorical_cols = [c for c in FEATURES if df[c].dtype == object or str(df[c].dtype) == "str"]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ---------- STEP 5: Quick pattern check (for your report) ----------
print("Stress_Level value counts:\n", df["Stress_Level"].value_counts())
print("\nAvg Anxiety by Stress_Level:\n", df.groupby("Stress_Level")["Anxiety_Score"].mean())
print("\nAvg Financial Stress by Stress_Level:\n", df.groupby("Stress_Level")["Financial_Stress"].mean())

# ---------- STEP 6: Features & Target ----------
X = df[FEATURES]
y = df["Stress_Level"]

# ---------- STEP 7: Train-test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- STEP 8: Scaling ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- STEP 9: Train SVM ----------
model = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42)
model.fit(X_train_scaled, y_train)

# ---------- STEP 10: Evaluate ----------
y_pred = model.predict(X_test_scaled)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------- STEP 11: Save model, scaler, encoders ----------
os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

with open(os.path.join(BASE_DIR, "model", "svm_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(BASE_DIR, "model", "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(BASE_DIR, "model", "encoders.pkl"), "wb") as f:
    pickle.dump(encoders, f)

with open(os.path.join(BASE_DIR, "model", "columns.pkl"), "wb") as f:
    pickle.dump(FEATURES, f)

print("\nSaved: svm_model.pkl, scaler.pkl, encoders.pkl, columns.pkl in model/")