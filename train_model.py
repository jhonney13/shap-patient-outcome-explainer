import joblib
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load preprocessed data
X_train, X_test, y_train, y_test, preprocessor = joblib.load("processed_data.pkl")

# Train XGBoost model
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Model trained with accuracy: {acc:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and preprocessor
joblib.dump(model, "ehr_xgb_model.pkl")
joblib.dump(preprocessor, "ehr_preprocessor.pkl")
