import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'age': np.random.randint(1, 18, n),
    'weight': np.random.normal(20, 7, n).round(1),
    'gender': np.random.choice(['Male', 'Female'], n),
    'bp_systolic': np.random.normal(100, 15, n).round(),
    'bp_diastolic': np.random.normal(65, 10, n).round(),
    'oxygen_level': np.random.uniform(85, 100, n).round(1),
    'heart_rate': np.random.randint(80, 180, n),
    'diagnosis': np.random.choice(['ASD', 'VSD', 'TOF', 'PDA'], n),
    'surgery_type': np.random.choice(['Open', 'Minimally Invasive'], n),
    'icu_days': np.random.poisson(3, n),
    'outcome': np.random.binomial(1, 0.2, n)  # 20% complication rate
})

data.to_csv("ehr_outcome_data.csv", index=False)
print("✅ Simulated EHR data saved to 'ehr_outcome_data.csv'")
