# ==============================
# Student Depression Prediction Page Only
# ==============================
import streamlit as st
import pandas as pd

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Student Depression Predictor",
    page_icon="🧑‍🎓",
    layout="wide"
)

# ---------------- Background Gradient ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #f3e5f5, #ede7f6);
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("""
<div style="
    background: linear-gradient(90deg, #7e57c2, #673ab7); 
    padding:25px; 
    border-radius:15px; 
    text-align:center; 
    color:white;
    box-shadow: 4px 4px 15px rgba(0,0,0,0.2);
">
    <h1>🧠 Student Depression Predictor</h1>
    <h3>Enter student details to predict depression risk</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- Get Model and Features from Session ----------------
model = st.session_state.get("model", None)
features = st.session_state.get("features", None)
X_train = st.session_state.get("X_train", None)  # To get default values

# Example options for selectboxes
gender_options = ["Male", "Female", "Other"]
degree_options = ["B.Tech", "B.Sc", "M.Sc"]
profession_options = ["Not Working", "Intern", "Part-Time Job"]
dietary_options = ["Vegetarian", "Non-Vegetarian", "Vegan"]
sleep_options = ["<5 hours", "5-6 hours", "6-7 hours", "7-8 hours", ">8 hours"]
suicidal_options = ["Yes", "No"]
family_history_options = ["Yes", "No"]

if model is None or features is None or X_train is None:
    st.warning("⚠️ Please train a model first on the training page.")
else:
    # ---------------- Input Entries ----------------
    st.header("📝 Enter Student Details")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 10, 60, 20)
        gender = st.selectbox("Gender", gender_options)
        degree = st.selectbox("Degree", degree_options)
        cgpa = st.number_input("CGPA", 0.0, 10.0, 7.0)
        academic_pressure = st.number_input("Academic Pressure (0-5)", 0, 5, 3)
        study_satisfaction = st.number_input("Study Satisfaction (0-5)", 0, 5, 3)

    with col2:
        working_part_time = st.checkbox("Working Part-Time?")
        if working_part_time:
            profession = st.selectbox("Profession", profession_options)
            job_satisfaction = st.number_input("Job Satisfaction (0-5)", 0, 5, 3)
        else:
            profession = "Not Working"
            job_satisfaction = 0

        work_study_hours = st.number_input("Work/Study Hours per Day", 0, 16, 6)
        work_pressure = st.number_input("Work Pressure (0-5)", 0, 5, 3)
        financial_stress = st.number_input("Financial Stress (0-5)", 0, 5, 3)
        dietary_habits = st.selectbox("Dietary Habits", dietary_options)
        sleep_duration = st.selectbox("Sleep Duration", sleep_options)
        suicidal_thoughts = st.selectbox("Suicidal Thoughts", suicidal_options)
        family_history = st.selectbox("Family History of Mental Illness", family_history_options)

    st.markdown("---")

    # ---------------- Prediction ----------------
    if st.button("Predict Depression"):
        input_data = pd.DataFrame([{
            "Age": age,
            "CGPA": cgpa,
            "Academic Pressure": academic_pressure,
            "Study Satisfaction": study_satisfaction,
            "Job Satisfaction": job_satisfaction,
            "Work/Study Hours": work_study_hours,
            "Work Pressure": work_pressure,
            "Financial Stress": financial_stress,
            "Gender": gender,
            "Degree": degree,
            "Profession": profession,
            "Dietary Habits": dietary_habits,
            "Sleep Duration": sleep_duration,
            "Have you ever had suicidal thoughts ?": suicidal_thoughts,
            "Family History of Mental Illness": family_history
        }])

        # Predict using session-state model
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

        # ---------------- Highlighted Prediction Box ----------------
        if prediction == 1:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #ff8a80, #d500f9); 
                padding:25px; 
                border-radius:20px; 
                color:white; 
                text-align:center;
                font-size:28px;
                font-weight:bold;
                box-shadow: 4px 4px 15px rgba(0,0,0,0.3);">
                Depressed<br>
                <span style="font-size:18px; font-weight:normal;">
                Probability: {probability:.2f} — Immediate support is recommended
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #42a5f5, #7e57c2); 
                padding:25px; 
                border-radius:20px; 
                color:white; 
                text-align:center;
                font-size:28px;
                font-weight:bold;
                box-shadow: 4px 4px 15px rgba(0,0,0,0.3);">
                Not Depressed<br>
                <span style="font-size:18px; font-weight:normal;">
                Probability: {probability:.2f} — Continue maintaining wellbeing
                </span>
            </div>
            """, unsafe_allow_html=True)
