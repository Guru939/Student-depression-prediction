# ==============================
# Student Depression Prediction App
# ==============================
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# ---------------- Streamlit Page Config ----------------
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
<div style="background: linear-gradient(90deg, #7e57c2, #673ab7); 
            padding:20px; border-radius:10px; text-align:center; color:white;">
    <h1>🧠Student Depression Predictor</h1>
    <h3>Enter student details to predict depression risk</h3>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ---------------- Load Dataset ----------------
df = pd.read_csv("student_depression_dataset.csv")

# ---------------- Define Features ----------------
numeric_features = ["Age", "CGPA", "Academic Pressure", "Study Satisfaction",
                    "Job Satisfaction", "Work/Study Hours", "Work Pressure",
                    "Financial Stress"]

categorical_features = ["Gender", "Degree", "Profession",
                        "Dietary Habits", "Sleep Duration",
                        "Have you ever had suicidal thoughts ?",
                        "Family History of Mental Illness"]

df[numeric_features] = df[numeric_features].apply(pd.to_numeric, errors='coerce')
df[categorical_features] = df[categorical_features].astype(str)

X = df[numeric_features + categorical_features]
y = df['Depression']

# ---------------- Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42, stratify=y
)

# ---------------- Preprocessing ----------------
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

ct = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

X_train_transformed = ct.fit_transform(X_train)
X_test_transformed = ct.transform(X_test)

# ---------------- Train Model ----------------
lr = LogisticRegression(
    penalty="l2",
    solver="liblinear",
    C=0.03,
    random_state=42,
    class_weight="balanced"
)
lr.fit(X_train_transformed, y_train)

# ---------------- Evaluate Model ----------------
y_pred = lr.predict(X_test_transformed)
f1 = f1_score(y_test, y_pred)
st.success(f"Model F1 Score: {f1:.2f}")

# ---------------- Input Entries ----------------
st.markdown("---")
st.header("📝Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 10, 60, 20)
    gender = st.selectbox("Gender", df["Gender"].unique())
    degree = st.selectbox("Degree", df["Degree"].unique())
    cgpa = st.number_input("CGPA", 0.0, 10.0, 7.0)
    academic_pressure = st.number_input("Academic Pressure (0-5)", 0, 5, 3)
    study_satisfaction = st.number_input("Study Satisfaction (0-5)", 0, 5, 3)

with col2:
    working_part_time = st.checkbox("Working Part-Time?")
    if working_part_time:
        profession = st.selectbox("Profession", df["Profession"].unique())
        job_satisfaction = st.number_input("Job Satisfaction (0-5)", 0, 5, 3)
    else:
        profession = "Not Working"
        job_satisfaction = 0

    work_study_hours = st.number_input("Work/Study Hours per Day", 0, 16, 6)
    work_pressure = st.number_input("Work Pressure (0-5)", 0, 5, 3)
    financial_stress = st.number_input("Financial Stress (0-5)", 0, 5, 3)
    dietary_habits = st.selectbox("Dietary Habits", df["Dietary Habits"].unique())
    sleep_duration = st.selectbox("Sleep Duration", df["Sleep Duration"].unique())
    suicidal_thoughts = st.selectbox("Suicidal Thoughts", df["Have you ever had suicidal thoughts ?"].unique())
    family_history = st.selectbox("Family History of Mental Illness", df["Family History of Mental Illness"].unique())

st.markdown("---")

# ---------------- Prediction ----------------
if st.button("Predict Depression"):
    query_point = pd.DataFrame([{
        "Age": age, "CGPA": cgpa, "Academic Pressure": academic_pressure,
        "Study Satisfaction": study_satisfaction, "Job Satisfaction": job_satisfaction,
        "Work/Study Hours": work_study_hours, "Work Pressure": work_pressure,
        "Financial Stress": financial_stress, "Gender": gender, "Degree": degree,
        "Profession": profession, "Dietary Habits": dietary_habits,
        "Sleep Duration": sleep_duration,
        "Have you ever had suicidal thoughts ?": suicidal_thoughts,
        "Family History of Mental Illness": family_history
    }])

    query_point[numeric_features] = query_point[numeric_features].astype(float)
    query_point[categorical_features] = query_point[categorical_features].astype(str)
    query_transformed = ct.transform(query_point)

    prediction = lr.predict(query_transformed)[0]
    probability = lr.predict_proba(query_transformed)[0][1]

    st.subheader("Prediction Result")

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

    # ---------------- Recommendations ----------------
    st.markdown("---")
    st.header("💡Personalized Recommendations")

    if prediction == 1:  # Depressed
        with st.expander("Study-Life Balance"):
            st.markdown("""
            <div style="background:#d1c4e9; padding:10px; border-radius:10px;">
            1. Plan your day: helps manage time efficiently. <em>(Like a daily cheat sheet for life!)</em><br>
            2. Take short breaks: prevents mental fatigue. <em>(Stretch, walk, or dance a little!)</em><br>
            3. Prioritize tasks: makes tasks manageable. <em>(Big scary homework = tiny bites!)</em><br>
            4. Avoid overcommitment: reduces stress triggers. <em>(Say “no” without guilt!)</em><br>
            5. Track mood/journal: monitor improvement. <em>(Your brain will thank you!)</em><br>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Exercise & Healthy Diet"):
            st.markdown("""
            <div style="background:#ce93d8; padding:10px; border-radius:10px;">
            1. Exercise daily: improves mood. <em>(Even a 10-min dance counts!)</em><br>
            2. Eat balanced meals: supports brain health. <em>(Veggies = brain fuel!)</em><br>
            3. Stay hydrated: prevents fatigue. <em>(Water > soda, your brain will love it!)</em><br>
            4. Reduce junk/sugar: avoids mood swings. <em>(Candy is fun, too much = cranky!)</em><br>
            5. Include protein/vitamins: sustains energy. <em>(Eggs, nuts = superpowers!)</em><br>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Sleep & Rest"):
            st.markdown("""
            <div style="background:#ba68c8; padding:10px; border-radius:10px; color:white;">
            1. Sleep 7–8 hours: supports mental health. <em>(Sleep = recharge button!)</em><br>
            2. Keep consistent sleep times: regulates rhythm. <em>(Body loves routines!)</em><br>
            3. Reduce screen time before bed: improves sleep. <em>(No phone zombies!)</em><br>
            4. Comfortable sleep environment: reduces disturbances. <em>(Cozy bed = happy mind!)</em><br>
            5. Relax before sleep: helps fall asleep. <em>(Read, chill, meditate!)</em><br>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Social Motivation"):
            st.markdown("""
            <div style="background:#9c27b0; padding:10px; border-radius:10px; color:white;">
            1. Connect with friends/family: social support. <em>(Call or chat = mood boost!)</em><br>
            2. Participate in hobbies/groups: boosts belonging. <em>(Fun + friends = happy vibes!)</em><br>
            3. Seek professional help: therapy if needed. <em>(Smart, not weak!)</em><br>
            4. Engage in laughter/light activities: reduces stress. <em>(Laugh like nobody's watching!)</em><br>
            5. Avoid isolation: maintain interactions. <em>(Don’t ghost yourself!)</em><br>
            </div>
            """, unsafe_allow_html=True)

    else:  # Not Depressed
        with st.expander("Study-Life Balance"):
            st.markdown("""
            <div style="background:#bbdefb; padding:10px; border-radius:10px;">
            1. Keep balanced daily schedule: mental stability. <em>(Like pizza slices: everything gets some time!)</em><br>
            2. Take regular breaks: sustains energy. <em>(Even superheroes rest!)</em><br>
            3. Set achievable academic goals: prevents stress. <em>(Small wins = big smiles!)</em><br>
            4. Avoid procrastination: reduces last-minute panic. <em>(Beat the panic monster!)</em><br>
            5. Stay organized: maintain control. <em>(Clutter-free = brain happy!)</em><br>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Exercise & Healthy Diet"):
            st.markdown("""
            <div style="background:#b39ddb; padding:10px; border-radius:10px;">
            1. Regular physical activity: keeps mind & body healthy. <em>(Move it or lose it!)</em><br>
            2. Eat healthy, balanced meals: brain + energy. <em>(Fuel your brain like a boss!)</em><br>
            3. Stay hydrated: focus + prevents fatigue. <em>(Water = brain juice!)</em><br>
            4. Limit processed/sugar: prevents mood swings. <em>(Too much candy = grumpy alert!)</em><br>
            5. Diet for energy & immunity: long-term wellbeing. <em>(Strong body = happy life!)</em><br>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Sleep & Rest"):
            st.markdown("""
            <div style="background:#9575cd; padding:10px; border-radius:10px; color:white;">
            1. Sleep 7–8 hours: mental & physical health. <em>(Recharge your brain!)</em><br>
            2. Consistent sleep/wake times: steady energy. <em>(Routine = secret power!)</em><br>
            3. Reduce screens 30–60 min before bed: sleep quality. <em>(Bye-bye phone zombies!)</em><br>
            4. Calm, comfy sleep environment: restful sleep. <em>(Cozy bed = happy mind!)</em><br>
            5. Relaxing bedtime routine: prepare mind for sleep. <em>(Read, chill, meditate!)</em><br>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Social Motivation"):
            st.markdown("""
            <div style="background:#7e57c2; padding:10px; border-radius:10px; color:white;">
            1. Stay connected: social support. <em>(Talking keeps spirits high!)</em><br>
            2. Social hobbies/clubs: sense of belonging. <em>(New friends = new energy!)</em><br>
            3. Share feelings openly: emotional resilience. <em>(Better out than bottled up!)</em><br>
            4. Light-hearted activities: reduces stress. <em>(Laugh like nobody's watching!)</em><br>
            5. Participate in community/group learning: engagement & wellbeing. <em>(Fun multiplies with friends!)</em><br>
            </div>
            """, unsafe_allow_html=True)
