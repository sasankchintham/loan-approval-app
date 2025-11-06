import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
pipeline = joblib.load("loan_approval_pipeline.pkl")

# App title
st.set_page_config(page_title="Loan Approval Predictor", page_icon="💳", layout="centered")
st.title("💳 Loan Approval Prediction App")
st.markdown("Check your loan approval eligibility using Machine Learning 🧠")

# Collect user input
st.subheader("🧾 Applicant Details")

person_age = st.number_input("Age", 18, 70, 30)
person_gender = st.selectbox("Gender", ["male", "female"])
person_education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "Associate", "Doctorate"])
person_income = st.number_input("Annual Income (₹)", 10000, 2000000, 50000)
person_emp_exp = st.number_input("Work Experience (Years)", 0, 40, 5)
person_home_ownership = st.selectbox("Home Ownership", ["OWN", "RENT", "MORTGAGE", "OTHER"])
loan_amnt = st.number_input("Loan Amount (₹)", 1000, 500000, 10000)
loan_intent = st.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
loan_int_rate = st.number_input("Loan Interest Rate (%)", 0.0, 30.0, 10.5)
loan_percent_income = st.number_input("Loan Percent of Income", 0.01, 1.0, 0.25)
cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", 1, 30, 5)
credit_score = st.number_input("Credit Score", 300, 850, 650)
previous_loan_defaults_on_file = st.selectbox("Previous Loan Default", ["Yes", "No"])

# Prepare DataFrame for prediction
input_data = pd.DataFrame([{
    "person_age": person_age,
    "person_gender": person_gender,
    "person_education": person_education,
    "person_income": person_income,
    "person_emp_exp": person_emp_exp,
    "person_home_ownership": person_home_ownership,
    "loan_amnt": loan_amnt,
    "loan_intent": loan_intent,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "credit_score": credit_score,
    "previous_loan_defaults_on_file": previous_loan_defaults_on_file
}])

# Prediction button
if st.button("🔮 Predict Loan Status"):
    prediction = pipeline.predict(input_data)[0]
    confidence = pipeline.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"✅ Loan Approved! (Confidence: {confidence*100:.2f}%)")
    else:
        st.error(f"❌ Loan Rejected (Confidence: {confidence*100:.2f}%)")
