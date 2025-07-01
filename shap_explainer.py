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
    st.subheader(" SHAP Explanation for Selected Patient")
    st.write("Input features:")
    st.write(input_df)
    # Preprocess
    X_input = preprocessor.transform(input_df)
    # SHAP values
    shap_input_values = explainer(X_input)
    shap_idx = 0
else:
    st.subheader(" Enter New Patient Data")
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

# Get transformed feature names from the preprocessor
try:
    transformed_feature_names = preprocessor.get_feature_names_out()
except Exception:
    transformed_feature_names = [f'Feature {i}' for i in range(X_input.shape[1])]

# Use these names for SHAP plots and tables
feature_names = transformed_feature_names

# Try to get the correct feature values (transformed)
try:
    feature_values = shap_input_values.data[0]
except Exception:
    feature_values = [None] * len(shap_input_values.values[0])

shap_vals = shap_input_values.values[0]

# Create a DataFrame for feature contributions
if len(feature_names) == len(shap_vals) == len(feature_values):
    shap_table = pd.DataFrame({
        'Feature': feature_names,
        'Value': feature_values,
        'SHAP Value': shap_vals,
        'Contribution': ['Positive' if v > 0 else 'Negative' for v in shap_vals]
    }).sort_values(by='SHAP Value', key=abs, ascending=False)
else:
    st.warning('Feature names and SHAP values do not align due to preprocessing (e.g., one-hot encoding). Displaying SHAP values for transformed features.')
    shap_table = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_vals,
        'Contribution': ['Positive' if v > 0 else 'Negative' for v in shap_vals]
    }).sort_values(by='SHAP Value', key=abs, ascending=False)

# Show the table
st.markdown("#### Feature Contributions Table")
st.dataframe(shap_table, use_container_width=True)

# Plain-language summary (use available features)
n_top = 3
most_impactful = shap_table.iloc[:n_top]
summ_lines = []
for _, row in most_impactful.iterrows():
    direction = 'increased' if row['SHAP Value'] > 0 else 'decreased'
    val = row['Value'] if 'Value' in row else ''
    summ_lines.append(f"- {row['Feature']} = {val} ({direction} the prediction)")
summary = '\n'.join(summ_lines)
pred_val = shap_input_values.base_values[0] + shap_input_values.values[0].sum()
st.markdown(f"**Summary:** The model predicts a value of `{pred_val:.3f}`. The top factors were:\n{summary}")

# Let user choose plot type
plot_type = st.radio("Choose SHAP plot type", ["Waterfall", "Bar"], horizontal=True)
fig = plt.figure()
if plot_type == "Waterfall":
    shap.plots.waterfall(shap_input_values[0], show=False)
else:
    shap.plots.bar(shap_input_values[0], show=False)
plt.tight_layout()
st.pyplot(fig)

# After the SHAP plot and summary, add this description:
st.markdown("---")
st.markdown("""
#### What does the SHAP graph tell you?

**SHAP (SHapley Additive exPlanations)** is a method to explain individual predictions of machine learning models. The SHAP graph shows how each feature contributed to the model's prediction for the selected patient:

- **Red bars** (or positive SHAP values) indicate features that increased the prediction.
- **Blue bars** (or negative SHAP values) indicate features that decreased the prediction.
- The longer the bar, the greater the impact of that feature on the prediction.
- The graph starts at the model's average prediction (baseline) and moves right (higher risk) or left (lower risk) as each feature is added.

This helps you understand which patient characteristics most influenced the model's decision for this specific case.
""")