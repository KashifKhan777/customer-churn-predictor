import streamlit as st
import joblib
import numpy as np

st.title("Telco Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

# User inputs for the dummy features
feature_1 = st.number_input("Feature 1", value=0.0)
feature_2 = st.number_input("Feature 2", value=0.0)
feature_3 = st.number_input("Feature 3", value=0.0)

if st.button("Predict"):
    try:
        # Load model
        model = joblib.load('churn_model.joblib')
        prediction = model.predict(np.array([[feature_1, feature_2, feature_3]]))
        
        if prediction[0] == 1:
            st.error("The customer is likely to churn!")
        else:
            st.success("The customer is likely to stay.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
