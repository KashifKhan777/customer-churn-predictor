
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("churn_model.pkl")

# Page settings
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Predictor")
st.write("Enter customer information to predict churn risk.")

# Customer information
st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    tenure = st.number_input(
        "Tenure (months)", min_value=0, max_value=72, value=12
    )

    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col3:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, value=70.0
    )

    total_charges = st.number_input(
        "Total Charges", min_value=0.0, value=840.0
    )

# Additional services
st.header("Additional Services")

col1, col2, col3 = st.columns(3)

with col1:
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

with col3:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

# Prediction
if st.button("🔍 Predict Churn Risk", type="primary"):

    customer = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # Calculate probability of churn
    probability = model.predict_proba(customer)[0][1]

    churn_percentage = probability * 100

    # Determine risk level
    if churn_percentage >= 60:
        risk = "High"
        recommendation = "Prioritize this customer for immediate retention outreach."

    elif churn_percentage >= 30:
        risk = "Medium"
        recommendation = "Monitor this customer and consider a retention offer."

    else:
        risk = "Low"
        recommendation = "Customer currently shows a relatively low churn risk."

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{churn_percentage:.1f}%"
    )

    st.info(f"**Risk Level: {risk}**")

    st.write("### Recommended Action")
    st.write(recommendation)
