import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("churn_model.pkl")

# -----------------------------
# App title
# -----------------------------
st.title("📊 Customer Churn Predictor")

st.write(
    "Use customer information to estimate the likelihood of churn "
    "and identify customers who may need retention attention."
)

st.divider()

# -----------------------------
# Customer information
# -----------------------------
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

# -----------------------------
# Billing information
# -----------------------------
st.subheader("Billing Information")

col3, col4 = st.columns(2)

with col3:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col4:
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
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=500.0
    )

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Churn", use_container_width=True):

    # Create a dataframe containing the customer's information
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
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # Make prediction
    prediction = model.predict(customer)[0]

    # Get churn probability
    probability = model.predict_proba(customer)[0][1]

    st.divider()

    if prediction == "Yes":
        st.error("⚠️ High Churn Risk")
        st.metric(
            "Churn Probability",
            f"{probability * 100:.1f}%"
        )

        st.subheader("Recommended Retention Actions")

        st.write("• Consider offering a contract upgrade incentive.")
        st.write("• Review the customer's monthly charges.")
        st.write("• Contact the customer proactively.")

    else:
        st.success("✅ Low Churn Risk")
        st.metric(
            "Churn Probability",
            f"{probability * 100:.1f}%"
        )

        st.write(
            "This customer is currently predicted to have a lower "
            "likelihood of churning."
        )
