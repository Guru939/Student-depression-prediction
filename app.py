import streamlit as st

# ✅ Page Config (must be first Streamlit command)
st.set_page_config(page_title="Student Depression ML Pipeline", layout="wide")

# ✅ Background styling
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://img.freepik.com/free-vector/abstract-purple-fluid-background_53876-99561.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.8);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ✅ Highlighted Title
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #7b1fa2, #d500f9, #f48fb1);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 4px 4px 15px rgba(0,0,0,0.4);
        margin-bottom: 25px;
    ">
        <h1 style="color:#ffffff; font-size:42px; font-weight:bold; margin-bottom:10px; text-shadow:2px 2px 6px #4a148c;">
            🧠 Student Depression Prediction
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ✅ Problem Statement
st.markdown(
    """
    <div style="background-color:rgba(248,187,208,0.9); padding:20px; border-radius:15px; margin-bottom:20px;">
        <h3 style="color:#880e4f;">📌 Problem Statement</h3>
        <p style="color:#4a148c; font-size:16px;">
        Depression among students is a growing concern that negatively impacts academic performance, 
        social interactions, and overall well-being. Early detection of depressive symptoms 
        can help provide timely support and reduce long-term risks.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ✅ Objective
st.markdown(
    """
    <div style="background-color:rgba(209,196,233,0.9); padding:20px; border-radius:15px; margin-bottom:20px;">
        <h3 style="color:#311b92;">🎯 Objective</h3>
        <p style="color:#1a237e; font-size:16px;">
        The main objective of this project is to develop a <b>Machine Learning pipeline</b> 
        that predicts the likelihood of depression in students based on academic, lifestyle, 
        and personal factors. This can assist educators, counselors, and institutions in identifying 
        students who may need psychological or emotional support.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ✅ Applications
st.markdown(
    """
    <div style="background-color:rgba(232,234,246,0.9); padding:20px; border-radius:15px;">
        <h3 style="color:#283593;">💡 Applications</h3>
        <ul style="color:#212121; font-size:16px;">
            <li><b>Educational Institutions</b>: Identify at-risk students early and provide counseling.</li>
            <li><b>Mental Health Professionals</b>: Use predictions to support clinical assessments.</li>
            <li><b>Researchers</b>: Analyze trends in student mental health and contributing factors.</li>
            <li><b>Parents & Guardians</b>: Gain insights into their child's mental well-being.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
