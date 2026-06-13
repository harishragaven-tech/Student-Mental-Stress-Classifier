import os
import pickle
import sqlite3
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DB_PATH = os.path.join(BASE_DIR, "database.db")

# ---------- Load model artifacts ----------
with open(os.path.join(MODEL_DIR, "svm_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, "encoders.pkl"), "rb") as f:
    encoders = pickle.load(f)
with open(os.path.join(MODEL_DIR, "columns.pkl"), "rb") as f:
    COLUMNS = pickle.load(f)

# ---------- Default values for fields NOT asked in the 10-question form ----------
DEFAULTS = {
    "Age": 23.0,
    "Course": "Medical",
    "Gender": "Male",
    "CGPA": 3.49,
    "Depression_Score": 2.25,
    "Relationship_Status": "Single",
    "Family_History": "No",
    "Chronic_Illness": "No",
    "Extracurricular_Involvement": "Moderate",
    "Semester_Credit_Load": 22.0,
    "Residence_Type": "On-Campus",
}

# ---------- Stress level labels & coping suggestions ----------
LEVEL_LABELS = {0: "Low Stress", 1: "Moderate Stress", 2: "High Stress"}

COPING_TIPS = {
    0: [
        "Great job! Keep maintaining your healthy routine.",
        "Continue regular sleep and exercise habits.",
        "Stay socially connected with friends and family.",
        "Keep a journal to track your mood over time."
    ],
    1: [
        "Try relaxation techniques like deep breathing or meditation.",
        "Maintain a consistent sleep schedule (7-8 hours).",
        "Take short breaks during study sessions.",
        "Talk to a friend, mentor, or counselor about your stress.",
        "Engage in light physical activity like walking or yoga."
    ],
    2: [
        "Consider speaking to a counselor or mental health professional.",
        "Prioritize sleep — aim for 7-8 hours every night.",
        "Break large tasks into smaller, manageable steps.",
        "Reduce caffeine/substance intake which can worsen anxiety.",
        "Reach out to your support network — you don't have to handle this alone.",
        "Practice mindfulness or guided meditation daily."
    ],
}

# ---------- Database setup ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sleep_quality TEXT,
            anxiety_score INTEGER,
            physical_activity TEXT,
            diet_quality TEXT,
            social_support TEXT,
            financial_stress INTEGER,
            substance_use TEXT,
            counseling_service_use TEXT,
            prediction TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/survey")
def survey():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT prediction, COUNT(*) FROM responses GROUP BY prediction")
    rows = c.fetchall()
    c.execute("SELECT COUNT(*) FROM responses")
    total = c.fetchone()[0]
    conn.close()

    counts = {"Low Stress": 0, "Moderate Stress": 0, "High Stress": 0}
    for label, cnt in rows:
        if label in counts:
            counts[label] = cnt

    return render_template("dashboard.html", counts=counts, total=total)

@app.route("/history")
def history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, sleep_quality, anxiety_score, physical_activity, diet_quality,
               social_support, financial_stress, substance_use, counseling_service_use,
               prediction, created_at
        FROM responses ORDER BY id DESC
    """)
    rows = c.fetchall()
    conn.close()
    return render_template("history.html", rows=rows)

@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    # ---- Build full feature dict ----
    data = dict(DEFAULTS)  # start with defaults
    data["Sleep_Quality"] = form.get("sleep_quality")
    data["Anxiety_Score"] = float(form.get("anxiety_score", 0))
    data["Physical_Activity"] = form.get("physical_activity")
    data["Diet_Quality"] = form.get("diet_quality")
    data["Social_Support"] = form.get("social_support")
    data["Financial_Stress"] = float(form.get("financial_stress", 0))
    data["Substance_Use"] = form.get("substance_use")
    data["Counseling_Service_Use"] = form.get("counseling_service_use")

    # ---- Encode categorical values using saved LabelEncoders ----
    row = []
    for col in COLUMNS:
        val = data[col]
        if col in encoders:
            le = encoders[col]
            try:
                val = le.transform([val])[0]
            except ValueError:
                val = 0
        row.append(val)

    # ---- Scale & predict ----
    X_scaled = scaler.transform([row])
    pred = int(model.predict(X_scaled)[0])

    label = LEVEL_LABELS.get(pred, "Unknown")
    tips = COPING_TIPS.get(pred, [])

    # ---- Save to SQLite ----
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO responses
        (sleep_quality, anxiety_score, physical_activity, diet_quality,
         social_support, financial_stress, substance_use, counseling_service_use, prediction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["Sleep_Quality"], data["Anxiety_Score"], data["Physical_Activity"],
        data["Diet_Quality"], data["Social_Support"], data["Financial_Stress"],
        data["Substance_Use"], data["Counseling_Service_Use"], label
    ))
    conn.commit()
    conn.close()

    return render_template("result.html", prediction=label, tips=tips, level=pred)

if __name__ == "__main__":
    app.run(debug=True)