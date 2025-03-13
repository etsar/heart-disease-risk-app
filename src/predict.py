import joblib
import numpy as np

# Load the trained model
model = joblib.load("models/best_xgb_model.joblib")

# Function to make predictions
def predict_heart_disease(input_data):
    """
    Takes a NumPy array with patient details and returns the heart disease risk prediction.
    """
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]  # Probability of high risk
    return prediction[0], probability[0]