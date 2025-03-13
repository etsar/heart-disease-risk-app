import joblib
import numpy as np
import pandas as pd

# Load the saved scaler
scaler = joblib.load("models/age_scaler.joblib")

# Define feature names
feature_names = ["age"]

def preprocess_input(chest_pain, shortness_of_breath, fatigue, palpitations, 
                     dizziness, swelling, pain_arms_jaw_back, cold_sweats_nausea,
                     high_bp, high_cholesterol, diabetes, smoking, obesity,
                     sedentary_lifestyle, family_history, chronic_stress, gender,
                     age):
    """
    Converts user inputs into a NumPy array for model prediction and scales age using the saved scaler.
    """

    # Convert age to a DataFrame with correct feature name
    age_df = pd.DataFrame([[age]], columns=feature_names)

    # Scale age using the pre-trained scaler
    scaled_age = scaler.transform(age_df)[0, 0]

    # Create the input array
    input_data = np.array([[chest_pain, shortness_of_breath, fatigue, palpitations, 
                     dizziness, swelling, pain_arms_jaw_back, cold_sweats_nausea,
                     high_bp, high_cholesterol, diabetes, smoking, obesity,
                     sedentary_lifestyle, family_history, chronic_stress, gender, 
                     scaled_age]])
    
    return input_data