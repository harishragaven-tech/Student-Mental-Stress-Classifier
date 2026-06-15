import os
import pickle
import sqlite3
import pandas as pd
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "mindcheck_secret_key_2026"  # change this to any random string

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
            name TEXT,
            email TEXT,
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

    c.execute("PRAGMA table_info(responses)")
    existing_cols = [row[1] for row in c.fetchall()]
    if "name" not in existing_cols:
        c.execute("ALTER TABLE responses ADD COLUMN name TEXT")
    if "email" not in existing_cols:
        c.execute("ALTER TABLE responses ADD COLUMN email TEXT")

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------- Auth helper ----------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ---------- Global login gate ----------
@app.before_request
def require_login():
    allowed_routes = ["login", "register", "static"]
    if request.endpoint not in allowed_routes and "user_id" not in session:
        return redirect(url_for("login"))
    
# ---------- Auth Routes ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        password_hash = generate_password_hash(password)

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, password_hash)
            )
            conn.commit()
            conn.close()
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            conn.close()
            flash("Username or email already exists.", "error")
            return render_template("register.html")

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session["user_id"] = user[0]
            session["username"] = user[1]
            flash(f"Welcome back, {user[1]}!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password.", "error")
            return render_template("login.html")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))

# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/survey")
@login_required
def survey():
    return render_template("index.html")

@app.route("/dashboard")
@login_required
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
@login_required
def history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, name, email, sleep_quality, anxiety_score, physical_activity, diet_quality,
               social_support, financial_stress, substance_use, counseling_service_use,
               prediction, created_at
        FROM responses ORDER BY id DESC
    """)
    rows = c.fetchall()
    conn.close()
    return render_template("history.html", rows=rows)

@app.route("/analytics")
def analytics():
    df = pd.read_csv(os.path.join(BASE_DIR, "students_mental_health_survey_cleaned.csv"))

    sleep_anxiety = df.groupby("Sleep_Quality")["Anxiety_Score"].mean().round(2)
    sleep_cgpa = df.groupby("Sleep_Quality")["CGPA"].mean().round(2)
    substance_counts = df["Substance_Use"].value_counts()
    activity_anxiety = df.groupby("Physical_Activity")["Anxiety_Score"].mean().round(2)

    return render_template(
        "analytics.html",
        sleep_labels=list(sleep_anxiety.index),
        sleep_anxiety=[float(v) for v in sleep_anxiety.values],
        sleep_cgpa=[float(v) for v in sleep_cgpa.values],
        substance_labels=list(substance_counts.index),
        substance_values=[int(v) for v in substance_counts.values],
        activity_labels=list(activity_anxiety.index),
        activity_anxiety=[float(v) for v in activity_anxiety.values],
    )

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        name = request.form.get("name")
        message = request.form.get("message")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("INSERT INTO feedback (name, message) VALUES (?, ?)", (name, message))
        conn.commit()
        conn.close()

        return render_template("feedback.html", submitted=True)

    return render_template("feedback.html", submitted=False)

@app.route("/model-info")
def model_info():
    return render_template("model_info.html", columns=COLUMNS)

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    form = request.form

    user_name = form.get("user_name", "")
    user_email = form.get("user_email", "")

    data = {
        "Sleep_Quality": form.get("sleep_quality"),
        "Anxiety_Score": float(form.get("anxiety_score", 0)),
        "Physical_Activity": form.get("physical_activity"),
        "Diet_Quality": form.get("diet_quality"),
        "Social_Support": form.get("social_support"),
        "Financial_Stress": float(form.get("financial_stress", 0)),
        "Substance_Use": form.get("substance_use"),
        "Counseling_Service_Use": form.get("counseling_service_use"),
    }

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
    if data["Anxiety_Score"] >= 4 or data["Financial_Stress"] >= 4:
        pred = 2
    elif data["Anxiety_Score"] >= 2 or data["Financial_Stress"] >= 2:
        pred = 1
    else:
        pred = 0

    label = LEVEL_LABELS.get(pred, "Unknown")
    tips = COPING_TIPS.get(pred, [])

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO responses
        (name, email, sleep_quality, anxiety_score, physical_activity, diet_quality,
         social_support, financial_stress, substance_use, counseling_service_use, prediction, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_name, user_email,
        data["Sleep_Quality"], data["Anxiety_Score"], data["Physical_Activity"],
        data["Diet_Quality"], data["Social_Support"], data["Financial_Stress"],
        data["Substance_Use"], data["Counseling_Service_Use"], label,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

    return render_template("result.html", prediction=label, tips=tips, level=pred)

if __name__ == "__main__":
    app.run(debug=True)