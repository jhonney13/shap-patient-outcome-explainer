import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.pyplot as plt

# Load model and preprocessor
model = joblib.load("ehr_xgb_model.pkl")
preprocessor = joblib.load("ehr_preprocessor.pkl")

# Load raw data (optional – just for column names)
raw_data = pd.read_csv("ehr_outcome_data.csv")
X_raw = raw_data.drop("outcome", axis=1)
y = raw_data["outcome"]

# Preprocess
X_transformed = preprocessor.transform(X_raw)

# SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_transformed)

# Add option for manual data entry
mode = st.radio("Choose input mode", ["Select existing patient", "Enter new patient data"])

if mode == "Select existing patient":
    sample_idx = st.slider("Select a patient index to explain", 0, len(X_raw) - 1, 0)
    input_df = X_raw.iloc[sample_idx:sample_idx+1]
    st.subheader("🧠 SHAP Explanation for Selected Patient")
    st.write("Input features:")
    st.write(input_df)
    # Preprocess
    X_input = preprocessor.transform(input_df)
    # SHAP values
    shap_input_values = explainer(X_input)
    shap_idx = 0
else:
    st.subheader("📝 Enter New Patient Data")
    # Numeric features
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    weight = st.number_input("Weight (kg)", min_value=0.0, max_value=200.0, value=70.0)
    bp_systolic = st.number_input("Systolic BP", min_value=50.0, max_value=250.0, value=120.0)
    bp_diastolic = st.number_input("Diastolic BP", min_value=30.0, max_value=150.0, value=80.0)
    oxygen_level = st.number_input("Oxygen Level (%)", min_value=50.0, max_value=100.0, value=98.0)
    heart_rate = st.number_input("Heart Rate", min_value=30, max_value=250, value=80)
    icu_days = st.number_input("ICU Days", min_value=0, max_value=60, value=1)
    # Categorical features
    gender = st.selectbox("Gender", ["Male", "Female"])
    diagnosis = st.selectbox("Diagnosis", sorted(X_raw["diagnosis"].unique()))
    surgery_type = st.selectbox("Surgery Type", sorted(X_raw["surgery_type"].unique()))
    # Create DataFrame
    input_dict = {
        "age": [age],
        "weight": [weight],
        "gender": [gender],
        "bp_systolic": [bp_systolic],
        "bp_diastolic": [bp_diastolic],
        "oxygen_level": [oxygen_level],
        "heart_rate": [heart_rate],
        "diagnosis": [diagnosis],
        "surgery_type": [surgery_type],
        "icu_days": [icu_days],
    }
    input_df = pd.DataFrame(input_dict)
    st.write("Input features:")
    st.write(input_df)
    # Preprocess
    X_input = preprocessor.transform(input_df)
    # SHAP values
    shap_input_values = explainer(X_input)
    shap_idx = 0

# SHAP force plot
#st.set_option('deprecation.showPyplotGlobalUse', False)
fig = plt.figure()
shap.plots.waterfall(shap_input_values[shap_idx], show=False)
plt.tight_layout()
st.pyplot(fig)