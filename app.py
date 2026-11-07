import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")


model = load_model()


# --------------------------------------------------
# APP TITLE
# --------------------------------------------------

st.title("📊 Customer Churn Predictor")

st.write(
    """
    Predict the likelihood that a telecom customer will leave the company.
    
    Enter the customer's details below and click **Predict Churn Risk**.
    """
)

st.divider()


# --------------------------------------------------
# CUSTOMER INFORMATION
# --------------------------------------------------

st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
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
        max_value=72,
        value=12
    )

with col2:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

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

with col3:
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


# --------------------------------------------------
# BILLING INFORMATION
# --------------------------------------------------

st.header("Contract & Billing Information")

col4, col5 = st.columns(2)

with col4:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
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

with col5:

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=840.0
    )


st.divider()


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

if st.button(
    "🔮 Predict Churn Risk",
    type="primary",
    use_container_width=True
):

    # Create a DataFrame containing the customer's information
    customer_data = pd.DataFrame({
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
    prediction = model.predict(customer_data)[0]

    # Get probability of churn
    probabilities = model.predict_proba(customer_data)[0]

    # Find probability for "Yes"
    churn_probability = probabilities[1] if prediction in ["Yes", 1] else probabilities[
        list(model.classes_).index("Yes")
    ]

    churn_percentage = churn_probability * 100


    # --------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------

    st.subheader("Prediction Result")

    if churn_probability >= 0.70:

        st.error(
            f"🚨 High Churn Risk — {churn_percentage:.1f}%"
        )

        st.write(
            """
            **Recommended action:** Contact this customer with a targeted
            retention offer, loyalty benefit, or contract upgrade.
            """
        )

    elif churn_probability >= 0.40:

        st.warning(
            f"⚠️ Medium Churn Risk — {churn_percentage:.1f}%"
        )

        st.write(
            """
            **Recommended action:** Monitor this customer and consider
            offering improved support, discounts, or service benefits.
            """
        )

    else:

        st.success(
            f"✅ Low Churn Risk — {churn_percentage:.1f}%"
        )

        st.write(
            """
            **Recommended action:** Continue normal customer engagement
            and maintain service quality.
            """
        )


    # --------------------------------------------------
    # SHOW PREDICTION DETAILS
    # --------------------------------------------------

    st.subheader("Prediction Details")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Predicted Churn",
            "Yes" if prediction == "Yes" else "No"
        )

    with result_col2:
        st.metric(
            "Churn Probability",
            f"{churn_percentage:.1f}%"
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Customer Churn Predictor | Built with Python, Scikit-learn and Streamlit"
)
