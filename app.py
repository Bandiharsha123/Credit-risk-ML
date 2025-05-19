import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("credit_risk_model.pkl")

# Streamlit app UI
st.title("🧮 Harsha Credit Risk Assessment")

st.write("Provide customer details to assess credit risk.")

# Input fields
age = st.slider("Age", 18, 70, 30)
income = st.number_input("Annual Income ($)", min_value=10000, value=50000)
loan_amount = st.number_input("Loan Amount Requested ($)", min_value=500, value=5000)

if st.button("Assess Risk"):
    # Prepare features
    input_features = np.array([[age, income, loan_amount]])
    prediction = model.predict(input_features)[0]
    
    # Display result
    if prediction == 1:
        st.error("⚠️ High Credit Risk")
    else:
        st.success("✅ Low Credit Risk")
