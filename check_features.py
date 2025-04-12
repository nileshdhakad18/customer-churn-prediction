import joblib

# Load the model
model = joblib.load('churn_model.pkl')

# Print features the model expects
print("Model requires these features:")
print(model.feature_names_in_)

# Print feature count
print(f"\nNumber of features expected: {len(model.feature_names_in_)}")