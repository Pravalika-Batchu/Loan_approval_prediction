import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load model and metadata
with open('loan_approval_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)
with open('best_threshold.pkl', 'rb') as f:
    best_threshold = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Loan Prediction System")

# User Input Fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.text_input("Loan Term (in months)", "360")
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Validate and convert loan term
try:
    loan_term = int(loan_term)
except ValueError:
    st.warning("Please enter a valid numerical value for Loan Term in months.")
    loan_term = 360  # Default value

if st.button("Predict Loan Approval"):
    # Prepare input data as DataFrame
    input_data = pd.DataFrame([{
        "Gender": gender, "Married": married, "Dependents": dependents,
        "Education": education, "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history, "Property_Area": property_area
    }])

    # One-Hot Encoding for categorical variables
    input_data = pd.get_dummies(input_data)

    # Ensure all necessary columns exist
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Add missing categorical variables as 0

    # Reorder columns to match training data
    input_data = input_data[feature_columns]

    # Scale numerical features using the saved StandardScaler
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    # Predict probability and raw prediction
    probability = model.predict_proba(input_data)[:, 1][0]
    raw_prediction = model.predict(input_data)[0]
    threshold_prediction = int(probability > best_threshold)

    # Debugging info
    st.write(f"Predicted Probability: {probability:.2f}")
    st.write(f"Using Best Threshold: {best_threshold:.5f}")
    #st.write(f"Threshold-Based Prediction (0/1): {threshold_prediction}")
    #st.write(f"Raw Model Prediction (0/1): {raw_prediction}")

    # Use raw prediction for final decision (to avoid threshold issues)
    if raw_prediction == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Denied")