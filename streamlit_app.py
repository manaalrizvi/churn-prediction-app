"""
streamlit_app.py — Streamlit UI that calls the FastAPI backend.

Run: streamlit run streamlit_app.py
(Make sure app.py / uvicorn is already running on port 8000)
"""

import streamlit as st
import requests

API_URL = "http://churn-prediction-app-1tpd.onrender.com/predict"

st.set_page_config(page_title="Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Predictor")
st.caption("Fill in customer details to predict churn risk.")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Has Partner", ["Yes", "No"])
        Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col2:
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, value=800.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    payload = {
        "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner,
        "Dependents": Dependents, "tenure": tenure, "PhoneService": PhoneService,
        "MultipleLines": MultipleLines, "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection, "TechSupport": TechSupport,
        "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies,
        "Contract": Contract, "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=45)
        result = response.json()

        if result["churn_prediction"] == "Yes":
            st.error(f"⚠️ High Churn Risk — Probability: {result['churn_probability']*100:.1f}%")
        else:
            st.success(f"✅ Low Churn Risk — Probability: {result['churn_probability']*100:.1f}%")

    except requests.exceptions.ConnectionError:
        st.error("Could not reach the API. Make sure `uvicorn app:app --port 8000` is running.")

