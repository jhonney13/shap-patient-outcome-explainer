import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
data = pd.read_csv("ehr_outcome_data.csv")

# Features and target
X = data.drop("outcome", axis=1)
y = data["outcome"]

# Define columns by type
numeric_cols = ['age', 'weight', 'bp_systolic', 'bp_diastolic', 'oxygen_level', 'heart_rate', 'icu_days']
categorical_cols = ['gender', 'diagnosis', 'surgery_type']

# Preprocessing pipeline
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply transformation
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save for modeling
import joblib
joblib.dump((X_train_processed, X_test_processed, y_train, y_test, preprocessor), "processed_data.pkl")

print("✅ Data preprocessed and saved to 'processed_data.pkl'")
