import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ---------------- PAGE CONFIG & BACKGROUND ----------------
st.set_page_config(page_title="Student Depression ML Pipeline", layout="wide")

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f4f0fa;  /* light lavender background */
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.95);
}
.heading-box {
    color: white;
    font-weight: 900;
    font-size: 28px;
    padding: 15px 20px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 20px;
}
.train-test {
    background-color: #7e57c2;  /* moderate purple */
}
.logistic {
    background-color: #f06292;  /* soft pink */
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------- STEP 1: Train/Test Split ----------------
st.markdown('<div class="heading-box train-test">Data Split</div>', unsafe_allow_html=True)

# Load dataset
df = st.session_state.get("data", None)
if df is None:
    try:
        df = pd.read_csv("student_depression_dataset.csv")
        st.session_state["data"] = df
        st.success("Dataset loaded successfully")
    except FileNotFoundError:
        st.error("'student_depression_dataset.csv' not found.")
        st.stop()

# ---------------- Drop unnecessary columns ----------------
columns_to_drop = ["id", "City", "Depression"]
X = df.drop(columns=columns_to_drop, errors="ignore")
y = df["Depression"]

features = X.columns.tolist()

if st.button("Split Data"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
    st.session_state["features"] = features
    st.success("Data split completed")

# ---------------- STEP 2: Logistic Regression ----------------
st.markdown('<div class="heading-box logistic">Logistic Regression</div>', unsafe_allow_html=True)

X_train = st.session_state.get("X_train", None)
y_train = st.session_state.get("y_train", None)
X_test = st.session_state.get("X_test", None)
y_test = st.session_state.get("y_test", None)
features = st.session_state.get("features", None)

if X_train is None or y_train is None:
    st.warning("Please complete Data Split first")
else:
    if st.button("Train Model"):
        numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])

        lr = LogisticRegression(penalty='l2', solver='liblinear', C=0.030733777087956198)
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", lr)
        ])

        model.fit(X_train, y_train)
        st.session_state["model"] = model
        st.success("Logistic Regression trained successfully")

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Save metrics in session_state
        st.session_state["train_acc"] = accuracy_score(y_train, y_train_pred)
        st.session_state["test_acc"] = accuracy_score(y_test, y_test_pred)
        st.session_state["train_f1"] = f1_score(y_train, y_train_pred)
        st.session_state["test_f1"] = f1_score(y_test, y_test_pred)

# ---------------- Display metrics side by side ----------------
if "train_acc" in st.session_state:
    # Heading styled same as Data Split
    st.markdown('<div class="heading-box train-test" style="text-align:center;">Model Performance Metrics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    # Single color for metrics
    metric_color = "#4a148c"

    # Accuracy
    col1.markdown(f"**<span style='color:{metric_color}'>Training Accuracy: {st.session_state['train_acc']*100:.2f}%</span>**", unsafe_allow_html=True)
    col2.markdown(f"**<span style='color:{metric_color}'>Test Accuracy: {st.session_state['test_acc']*100:.2f}%</span>**", unsafe_allow_html=True)

    # F1-Score
    col1.markdown(f"**<span style='color:{metric_color}'>Training F1-Score: {st.session_state['train_f1']:.3f}</span>**", unsafe_allow_html=True)
    col2.markdown(f"**<span style='color:{metric_color}'>Test F1-Score: {st.session_state['test_f1']:.3f}</span>**", unsafe_allow_html=True)
