import streamlit as st
import numpy as np
import pickle
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    font-family: 'Segoe UI', sans-serif;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #cfd8dc;
    margin-bottom: 30px;
}

.glass {
    background: rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 25px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    backdrop-filter: blur(12px);
}

.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    padding: 0.6em 2em;
    font-size: 16px;
    font-weight: bold;
    border: none;
}

.success {
    color: #2ecc71;
    font-size: 22px;
    font-weight: bold;
}

.error {
    color: #e74c3c;
    font-size: 22px;
    font-weight: bold;
}

section[data-testid="stSidebar"] {
    background: #0b1c2d;
}

.footer {
    text-align: center;
    color: #cfd8dc;
    font-size: 14px;
    margin-top: 40px;
    padding: 15px 0;
    border-top: 1px solid rgba(255,255,255,0.2);
}
.footer span {
    color: #00c6ff;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ===================== TITLE =====================
st.markdown('<div class="main-title">🏦 Loan Approval Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Loan Eligibility System</div>', unsafe_allow_html=True)

# ===================== LOAD MODELS =====================
log_model = pickle.load(open("model/logistic_model.pkl", "rb"))
rf_model = pickle.load(open("model/rf_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
features = pickle.load(open("model/features.pkl", "rb"))

# ===================== LOAD DATA =====================
df = pd.read_csv("data/loan_data.csv")

df.columns = df.columns.str.strip()
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

le = LabelEncoder()
for col in ["education", "self_employed"]:
    df[col] = le.fit_transform(df[col])

df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})

X = df.drop(["loan_status", "loan_id"], axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_scaled = scaler.transform(X_train)

# ===================== SIDEBAR =====================
st.sidebar.header("⚙️ Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Random Forest"]
)

# ===================== INPUT FORM =====================
st.markdown('<div class="glass">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    dependents = st.number_input(
        "👨‍👩‍👧 Number of Dependents",
        min_value=0,
        max_value=10,
        step=1,
        help="Total number of financial dependents"
    )

    education = st.selectbox(
        "🎓 Education Qualification",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "💼 Employment Type",
        ["Salaried", "Self-Employed"]
    )

    income = st.number_input(
        "💰 Annual Income (₹)",
        min_value=0,
        max_value=5_000_000,
        step=10_000,
        help="Gross annual income in INR"
    )

    loan_amount = st.number_input(
        "🏦 Requested Loan Amount (₹)",
        min_value=50_000,
        max_value=10_000_000,
        step=50_000,
        help="Loan amount you wish to apply for"
    )

    loan_term = st.number_input(
        "⏳ Loan Tenure (Years)",
        min_value=1,
        max_value=30,
        step=1,
        help="Repayment duration in years"
    )

with col2:
    cibil = st.number_input(
        "📊 CIBIL Credit Score",
        min_value=300,
        max_value=900,
        step=1,
        help="Credit score between 300 and 900"
    )

    res_asset = st.number_input(
        "🏠 Residential Property Value (₹)",
        min_value=0,
        max_value=50_000_000,
        step=100_000,
        help="Market value of residential assets"
    )

    com_asset = st.number_input(
        "🏢 Commercial Property Value (₹)",
        min_value=0,
        max_value=100_000_000,
        step=100_000,
        help="Market value of commercial assets"
    )

    lux_asset = st.number_input(
        "💎 Luxury Assets Value (₹)",
        min_value=0,
        max_value=20_000_000,
        step=50_000,
        help="Value of vehicles, jewelry, etc."
    )

    bank_asset = st.number_input(
        "🏧 Bank Balance & Investments (₹)",
        min_value=0,
        max_value=20_000_000,
        step=50_000,
        help="Savings, FD, mutual funds, etc."
    )


st.markdown('</div>', unsafe_allow_html=True)

# ===================== INPUT PROCESSING =====================
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

input_data = [[
    dependents, education, self_employed, income,
    loan_amount, loan_term, cibil,
    res_asset, com_asset, lux_asset, bank_asset
]]

input_df = pd.DataFrame(input_data, columns=features)
input_scaled = scaler.transform(input_df)

# ===================== PREDICTION =====================
if st.button("🔍 Predict Loan Status"):

    if model_choice == "Logistic Regression":
        prediction = log_model.predict(input_scaled)
        y_pred_train = log_model.predict(X_train_scaled)
    else:
        prediction = rf_model.predict(input_scaled)
        y_pred_train = rf_model.predict(X_train_scaled)

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction[0] == 1:
        st.markdown(
            '<div class="success">✅ Loan Approved<br>'
            'We are pleased to inform you that your loan application meets all eligibility criteria and has been approved.</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            '<div class="error">❌ Loan Not Approved<br>'
            'After careful evaluation of your application, we are unable to approve your loan request at this time.</div>',
            unsafe_allow_html=True
        )

    # ===================== PERFORMANCE =====================
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📊 Model Performance")

    acc = accuracy_score(y_train, y_pred_train)
    cm = confusion_matrix(y_train, y_pred_train)

    st.metric("Accuracy", f"{acc*100:.2f}%")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    Developed by <span>Nikita Mane</span>| Data Scientiest
</div>
""", unsafe_allow_html=True)
