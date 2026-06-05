# Student Mental Stress Classifier

## Project Overview

The Student Mental Stress Classifier is a Machine Learning project developed to analyze and predict the stress levels of students based on various academic, personal, and lifestyle factors.

The project uses a Student Mental Health Survey Dataset and applies data preprocessing, exploratory data analysis, machine learning, and visualization techniques to classify student stress levels.

---

## Problem Statement

Many students experience stress due to academic pressure, financial issues, lack of sleep, anxiety, and other personal factors.

The objective of this project is to build a machine learning model that can predict a student's stress level using survey data.


## Dataset Information

Dataset Name:
Student Mental Health Survey Dataset

Features Used:

- Age
- Course
- Gender
- CGPA
- Depression Score
- Anxiety Score
- Sleep Quality
- Physical Activity
- Diet Quality
- Social Support
- Relationship Status
- Substance Use
- Counseling Service Use
- Family History
- Chronic Illness
- Financial Stress
- Extracurricular Involvement
- Semester Credit Load
- Residence Type

Target Variable:

- Stress_Level

---

## Tasks Completed

### Task 1 - Dataset Loading

- Loaded dataset using Pandas
- Checked shape, data types, and missing values

### Task 2 - Exploratory Data Analysis (EDA)

- Statistical summary
- Missing value analysis
- Duplicate record analysis
- Grouped analysis

### Task 3 - Data Preprocessing

- Handled missing values
- Removed duplicates
- Label encoded categorical columns
- Saved processed dataset

### Task 4 - Machine Learning Model

- Split dataset into training and testing sets
- Trained Random Forest Classifier
- Evaluated model performance

### Task 5 - Model Saving & Prediction

- Saved trained model as model.pkl
- Loaded saved model
- Predicted stress levels

### Task 6 - Data Visualization

- Stress Level Distribution
- Anxiety vs Stress Analysis
- CGPA vs Stress Analysis
- Financial Stress Analysis
- Correlation Heatmap

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

---

## Project Structure

STUDENT-MENTAL-STRESS-CLASSIFIER
│
├── M1
│   ├── anxiety_score_vs_stress.png
│   ├── cgpa_vs_stress.png
│   ├── m1_analysis.py
│   ├── processed_student_stress.csv
│   ├── sleep_quality_vs_stress.png
│   ├── students_mental_health_survey.csv
│   └── task2_eda.py
│
├── M2
│   ├── visualizations
│   ├── task3_preprocessing.py
│   ├── task4_model_training.py
│   └── task5_save_predict.py
│
├── model.pkl
└── README.md
---

## Results

- Dataset successfully analyzed
- Machine learning model trained
- Stress level predictions generated
- Visualizations created for data insights
