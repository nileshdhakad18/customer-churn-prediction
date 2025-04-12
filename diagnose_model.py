import joblib
import pandas as pd

# Load model
model = joblib.load('churn_model.pkl')

# Print critical model info
print("=== MODEL DIAGNOSIS ===")
print("\n1. Feature Names:")
print(model.feature_names_in_)

print("\n2. Top 5 Most Important Features:")
coefs = pd.Series(model.coef_[0], index=model.feature_names_in_)
print(coefs.abs().sort_values(ascending=False).head(5))

print("\n3. Example Prediction Input:")
print("{")
for feature in model.feature_names_in_[:5]:  # First 5 features as example
    print(f'    "{feature}": 0,')
print("    ...")
print("}")