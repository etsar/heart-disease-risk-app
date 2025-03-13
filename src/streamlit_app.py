import streamlit as st
import numpy as np
from predict import predict_heart_disease
from preprocess import preprocess_input

# st.set_page_config(page_title="Heart Disease Risk Prediction", layout="wide")

# App title
st.title("❤️ Heart Disease Risk Prediction")
st.markdown("#### Enter your details below to check your estimated risk of heart disease:")

# User input fields

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=100, value=50)
    # Symptoms
    # st.subheader("🩺 Symptoms")
    st.markdown("<h4>🩺 Symptoms</h4>", unsafe_allow_html=True)
    fatigue = st.checkbox("Fatigue")
    pain_arms_jaw_back = st.checkbox("Pain in Arms/Jaw/Back")
    shortness_of_breath = st.checkbox("Shortness of Breath")
    cold_sweats_nausea = st.checkbox("Cold Sweats & Nausea")
    palpitations = st.checkbox("Palpitations")
    dizziness = st.checkbox("Dizziness")
    swelling = st.checkbox("Swelling")
    chest_pain = st.checkbox("Chest Pain")

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    # Risk Factors
    # st.subheader("⚕️Medical & Lifestyle Factors")
    st.markdown("<h4 style='margin-top: 15px;'>⚕️ Medical & Lifestyle Factors</h4>", unsafe_allow_html=True)
    sedentary_lifestyle = st.checkbox("Sedentary Lifestyle")
    high_bp = st.checkbox("High Blood Pressure")
    chronic_stress = st.checkbox("Chronic Stress")
    high_cholesterol = st.checkbox("High Cholesterol")
    obesity = st.checkbox("Obesity")
    smoking = st.checkbox("Smoking")
    diabetes = st.checkbox("Diabetes")
    family_history = st.checkbox("Family History of Heart Disease")

# Prediction button
if st.button("Check My Heart Risk"):
    # Convert user input into numerical format
    gender_numeric = 1 if gender == "Male" else 0

    input_data = preprocess_input(
        chest_pain, shortness_of_breath, fatigue, palpitations, 
        dizziness, swelling, pain_arms_jaw_back, cold_sweats_nausea,
        high_bp, high_cholesterol, diabetes, smoking, obesity,
        sedentary_lifestyle, family_history, chronic_stress, 
        gender_numeric, age
    )

    # Get model prediction
    prediction, probability = predict_heart_disease(input_data)

    # Display results
    # risk_label = "🔴 High Risk" if prediction == 1 else "🟢 Low Risk"
    # st.markdown(f"### Prediction: **{risk_label}**")
    # st.markdown(f"**🧮 Probability of High Risk:** {probability:.2%}")
    if prediction == 1:
        risk_label = "<span style='color: red; font-weight: bold;'>🔴 High Risk</span>"
    else:
        risk_label = "<span style='color: green; font-weight: bold;'>🟢 Low Risk</span>"

    st.markdown(f"<h3>Prediction: {risk_label}</h3>", unsafe_allow_html=True)
    st.markdown(
    f"<h4>🧮 Probability of High Risk: <b>{probability:.2%}</b></h4>",
    unsafe_allow_html=True)

# Add disclaimer
st.markdown("---")  # Adds a horizontal line for separation
st.markdown(
    """
    ⚠️ **Disclaimer:**  
    *This tool is for **informational purposes only** and should not be considered medical advice.*  
    *Predictions are based on statistical models and do not replace professional medical evaluation*.  
    *If you have concerns about your heart health, please consult a doctor or healthcare provider*.  
    """,
    unsafe_allow_html=True)