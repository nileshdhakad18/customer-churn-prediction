import joblib
import numpy as np

model = joblib.load('churn_model.pkl')

# 1. Check for "nuclear" coefficients
print("Top 5 most extreme coefficients:")
for name, coef in sorted(zip(model.feature_names_in_, model.coef_[0]), 
                        key=lambda x: abs(x[1]), reverse=True)[:5]:
    print(f"{name}: {coef:.2f}")

# 2. Verify model's expected input
print("\nSample valid input:")
print("{")
for feat in model.feature_names_in_:
    print(f'    "{feat}": 0.0,  # Change based on feature type')
print("}")