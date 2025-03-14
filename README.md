# ❤️ Heart Disease Risk Prediction App

This is a **machine learning-powered web application** that predicts heart disease risk based on user inputs. It utilizes **XGBoost**, trained on a synthetic dataset, and provides an **interactive interface** using **Streamlit**. You can try the app here: [Heart Disease Risk Prediction App](https://heart-disease-risk-app.streamlit.app/).

## Features
- **User-friendly interface** (Streamlit-powered)
- **Predicts heart disease risk** based on symptoms, medical history, and lifestyle factors
- **Uses trained XGBoost, our best-performing machine learning model**
- **Provides probability scores** for risk assessment
- **Deployed on Streamlit Cloud**

## Project Structure
```
heart-disease-risk-app/
│-- models/                     # Folder for trained ML models
│   ├── best_xgb_model.joblib   # Trained XGBoost model
│   ├── age_scaler.joblib       # Trained StandardScaler for age feature
│-- src/                        # Source code
│   ├── predict.py              # Function for making predictions
│   ├── preprocess.py           # Data preprocessing utilities
│   ├── streamlit_app.py        # Main Streamlit app
│-- requirements.txt            # Required Python packages
│-- README.md                   # Project documentation
```

### Disclaimer
This tool is for informational purposes only and should not be considered medical advice.
Predictions are based on statistical models and do not replace professional medical evaluation.
If you have concerns about your heart health, please consult a doctor or healthcare provider.
