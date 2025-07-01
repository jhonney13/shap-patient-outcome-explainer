# Patient Outcome Prediction Using EHR Data - SHAP Explainer App

This Streamlit app allows you to explore and interpret predictions from a machine learning model trained on electronic health record (EHR) data. It uses SHAP (SHapley Additive exPlanations) to explain individual patient predictions, helping you understand which features most influence the model's decisions.

## Features
- **Select an existing patient** from the dataset or **enter new patient data** manually.
- **Visualize SHAP explanations** for individual predictions using waterfall and bar plots.
- **See a table of feature contributions** and a plain-language summary of the top factors.
- **Understand the model's reasoning** for each prediction with a clear SHAP explanation.

## Setup
1. **Clone this repository** and navigate to the project directory.
2. **Install dependencies** (preferably in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure you have the following files in the project directory:**
   - `ehr_outcome_data.csv` (raw EHR data)
   - `ehr_xgb_model.pkl` (trained XGBoost model)
   - `ehr_preprocessor.pkl` (preprocessing pipeline)

## Usage
Run the Streamlit app with:
```bash
streamlit run shap_explainer.py
```

- Use the sidebar to select a patient or enter new data.
- View the SHAP plots, feature contribution table, and summary.
- Read the explanation at the bottom to understand how to interpret the SHAP graph.

## What is SHAP?
SHAP (SHapley Additive exPlanations) is a method to explain individual predictions of machine learning models. It shows how each feature contributed to a specific prediction, making model decisions more transparent and trustworthy.

## License
This project is for educational and research purposes. 