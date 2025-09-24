# ML/pages/EDA_Student.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="EDA - Student Depression", layout="wide")

# ---------- Page Background ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #f3e5f5, #ede7f6); /* lavender gradient */
}
h1, h2, h3 {
    font-family: 'Arial', sans-serif;
}
.section-header {
    background: linear-gradient(90deg, #7e57c2, #673ab7);
    padding: 15px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown("""
<div class="section-header">
    <h1>📊 Student Depression Dataset - EDA</h1>
</div>
""", unsafe_allow_html=True)

# ---------- Find CSV ----------
def find_csv(filename="student_depression_dataset.csv"):
    candidates = [
        Path(__file__).parent / filename,
        Path(__file__).parent.parent / filename,
        Path.cwd() / filename,
        Path.cwd() / "app" / filename,
        Path.cwd() / "ML" / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())
    return None

# ---------- Load dataset ----------
@st.cache_data
def load_data(path: str):
    return pd.read_csv(path)

csv_path = find_csv("student_depression_dataset.csv")
df = None
if csv_path:
    df = load_data(csv_path)
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)

if df is None:
    st.warning("No dataset found. Place student_depression_dataset.csv in your project or upload it.")
    st.stop()

# ---------- Column identification ----------
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

st.sidebar.success(f"✅ Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")

# ---------- Sidebar selection ----------
section = st.sidebar.radio("Choose analysis:", 
                          ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])

# ---------- Univariate ----------
if section == "Univariate Analysis":
    st.markdown('<div class="section-header"><h2>📈 Univariate Analysis</h2></div>', unsafe_allow_html=True)
    col = st.selectbox("Select column", df.columns)
    if col in numeric_cols:
        fig = px.histogram(df, x=col, marginal="box", nbins=30, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
        st.write(df[col].describe())
    else:
        vc = df[col].value_counts().reset_index()
        vc.columns = [col, "count"]
        fig = px.bar(vc, x=col, y="count", title=f"Counts of {col}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(vc)

# ---------- Bivariate ----------
elif section == "Bivariate Analysis":
    st.markdown('<div class="section-header"><h2>🔗 Bivariate Analysis</h2></div>', unsafe_allow_html=True)
    if len(numeric_cols) >= 2:
        x_col = st.selectbox("X (numeric)", numeric_cols, index=0)
        y_col = st.selectbox("Y (numeric)", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        color_col = st.selectbox("Optional color (categorical)", [None] + categorical_cols)
        fig = px.scatter(df, x=x_col, y=y_col, 
                         color=color_col if color_col else None,
                         trendline="ols", title=f"{y_col} vs {x_col}")
        st.plotly_chart(fig, use_container_width=True)
        st.write("Correlation:", df[[x_col, y_col]].corr().iloc[0,1].round(3))
    else:
        st.info("Need at least two numeric columns for bivariate plots.")

# ---------- Multivariate ----------
elif section == "Multivariate Analysis":
    st.markdown('<div class="section-header"><h2>🌐 Multivariate Analysis</h2></div>', unsafe_allow_html=True)
    cols = st.multiselect("Select numeric columns", numeric_cols, default=numeric_cols[:4])
    if len(cols) >= 2:
        st.markdown("**Scatter matrix**")
        fig = px.scatter_matrix(df, dimensions=cols, title="Scatter matrix")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Correlation heatmap**")
        corr = df[cols].corr()
        fig2 = px.imshow(corr, text_auto=True, title="Correlation heatmap")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Select at least 2 numeric columns for multivariate analysis.")
