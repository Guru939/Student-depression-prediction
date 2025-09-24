import streamlit as st
import pandas as pd

st.set_page_config(page_title="Student Depression Data Info", layout="wide")

# --- Page Background with Soft Purple Gradient + Global Table Styling ---
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #ede7f6, #d1c4e9); /* light purple gradient */
    background-size: cover;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.85);
}

/* Main heading style */
.main-heading {
    background: linear-gradient(90deg, #f48fb1, #ce93d8);
    color: white;
    padding: 18px;
    border-radius: 15px;
    font-weight: bold;
    font-size: 30px;
    text-align: center;
    margin-bottom: 20px;
    text-shadow: 2px 2px 5px #880e4f;
}

/* Section headings style */
.section-heading {
    background: linear-gradient(90deg, #9575cd, #7e57c2);
    color: white;
    padding: 12px;
    border-radius: 10px;
    font-weight: bold;
    font-size: 22px;
    text-align: center;
    margin-bottom: 15px;
    text-shadow: 1px 1px 3px #311b92;
}

/* Global dataframe style */
[data-testid="stDataFrame"] table {
    background-color: #f3e5f5;  /* light pastel purple */
    color: #4a148c;            /* dark purple text */
    font-size: 14px;
}

/* Header style */
[data-testid="stDataFrame"] th {
    background-color: #7e57c2;
    color: white;
    font-weight: bold;
    font-size: 16px;
    text-align: center !important;
}

/* Cell alignment */
[data-testid="stDataFrame"] td {
    text-align: center !important;
    vertical-align: middle !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --- Main Title ---
st.markdown('<div class="main-heading">📂 Data & Feature Information</div>', unsafe_allow_html=True)

# --- Load Dataset ---
df = pd.read_csv("student_depression_dataset.csv")
st.session_state["data"] = df

# --- Section: Preview of Dataset ---
st.markdown('<div class="section-heading">🔎 Preview of Dataset</div>', unsafe_allow_html=True)
st.dataframe(df.head(), use_container_width=True)   # ✅ interactive preview

# --- Feature Explanations Dictionary ---
feature_explanations = {
    "id": "Unique identifier for each student.",
    "Gender": "Student's gender (Male/Female/Other).",
    "Age": "Age of the student in years.",
    "City": "City where the student lives.",
    "Profession": "Student’s field of study or occupation.",
    "Academic Pressure": "Self-reported academic workload/pressure (scale-based).",
    "Work Pressure": "Workload pressure if employed alongside studies (scale-based).",
    "CGPA": "Cumulative Grade Point Average, indicator of academic performance.",
    "Study Satisfaction": "How satisfied the student is with their studies (scale-based).",
    "Job Satisfaction": "Satisfaction level with job (if employed).",
    "Sleep Duration": "Average sleep hours per day.",
    "Dietary Habits": "Type of diet (Healthy/Unhealthy/Moderate).",
    "Degree": "Level of education (Bachelor’s, Master’s, etc.).",
    "Have you ever had suicidal thoughts ?": "History of suicidal thoughts (Yes/No).",
    "Work/Study Hours": "Average hours spent on work/study daily.",
    "Financial Stress": "Level of financial stress (scale-based).",
    "Family History of Mental Illness": "Whether there is a family history of mental illness (Yes/No).",
    "Depression": "Target variable (1 = Depressed, 0 = Not Depressed)."
}

# --- Build Dataset Info Table ---
info_data = []
for col in df.columns:
    dtype = str(df[col].dtype)
    non_nulls = df[col].notnull().sum()
    explanation = feature_explanations.get(col, "No explanation available.")
    
    if pd.api.types.is_numeric_dtype(df[col]):
        mean_val = round(df[col].mean(), 2)
        min_val = round(df[col].min(), 2)
        max_val = round(df[col].max(), 2)
    else:
        mean_val = min_val = max_val = "-"
    
    info_data.append([col, dtype, non_nulls, mean_val, min_val, max_val, explanation])

info_df = pd.DataFrame(
    info_data, 
    columns=["Feature", "Data Type", "Non-Null Count", "Mean", "Min", "Max", "Explanation"]
)

# --- Section: Dataset Info ---
st.markdown('<div class="section-heading">📊 Dataset Info with Explanations & Descriptive Stats</div>', unsafe_allow_html=True)
st.dataframe(info_df, use_container_width=True)   # ✅ aligned and styled
