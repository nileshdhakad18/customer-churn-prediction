import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- Model Loading with Validation ---
try:
    model = joblib.load('churn_model.pkl')
    
    # Validate model structure
    if not hasattr(model, 'feature_names_in_'):
        st.error("❌ Invalid model: Missing feature names. Retrain with scikit-learn 1.0+")
        st.stop()
        
    expected_features = list(model.feature_names_in_)
    st.success(f"✅ Model loaded (expecting {len(expected_features)} features)")
    
    # Show model summary
    with st.expander("🔍 Model Details"):
        st.write("Top 5 Most Influential Features:")
        coefs = pd.Series(model.coef_[0], index=model.feature_names_in_)
        st.table(coefs.abs().sort_values(ascending=False).head(5))
        
except Exception as e:
    st.error(f"❌ Model loading failed: {str(e)}")
    st.stop()

# --- Input Form ---
st.title("📉 Customer Churn Predictor")
st.sidebar.header("Customer Details")

inputs = {
    "tenure": st.sidebar.slider("Tenure (months)", 1, 72, 12),
    "MonthlyCharges": st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0),
    "gender": st.sidebar.selectbox("Gender", ["Female", "Male"]),
    "SeniorCitizen": st.sidebar.selectbox("Senior Citizen", ["No", "Yes"]),
    "Partner": st.sidebar.selectbox("Partner", ["No", "Yes"]),
    "Dependents": st.sidebar.selectbox("Dependents", ["No", "Yes"]),
    "Contract": st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
    "InternetService": st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
    "PaperlessBilling": st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
}

# --- Feature Engineering ---
def prepare_features(inputs):
    features = {feat: 0.0 for feat in expected_features}
    
    # Numerical features with log scaling
    features.update({
        "tenure": np.log1p(float(inputs["tenure"])),
        "MonthlyCharges": np.log1p(float(inputs["MonthlyCharges"])),
        "TotalCharges": np.log1p(float(inputs["tenure"] * inputs["MonthlyCharges"]))
    })
    
    # Binary features with soft values
    binary_map = {"Yes": 0.7, "No": 0.3}
    features.update({
        "SeniorCitizen": binary_map[inputs["SeniorCitizen"]],
        "Partner": binary_map[inputs["Partner"]],
        "Dependents": binary_map[inputs["Dependents"]],
        "PaperlessBilling": binary_map[inputs["PaperlessBilling"]],
        "PhoneService": 0.5  # Neutral default
    })
    
    # One-hot encoded features
    features.update({
        f"gender_{inputs['gender']}": 0.9,
        f"Contract_{inputs['Contract'].replace('-', '')}": 0.9,
        f"InternetService_{inputs['InternetService']}": 0.9,
        "MultipleLines_No": 0.7,
        "OnlineSecurity_No": 0.7
    })
    
    return pd.DataFrame([features])[expected_features]

# --- Prediction Logic ---
if st.button("Predict Churn Risk"):
    try:
        features = prepare_features(inputs)
        
        # Debug information
        with st.expander("🔍 Prediction Analysis"):
            st.write("### Prepared Features")
            st.dataframe(features.T.rename(columns={0: "Value"}))
            
            st.write("### Feature Impacts")
            impacts = pd.DataFrame({
                "Feature": expected_features,
                "Value": features.values[0],
                "Coefficient": model.coef_[0],
                "Impact": features.values[0] * model.coef_[0]
            }).sort_values("Impact", key=abs, ascending=False)
            st.table(impacts.head(10))
        
        # Get prediction with safety constraints
        prob = np.clip(model.predict_proba(features)[0][1], 0.01, 0.99)
        st.metric("Churn Probability", f"{prob:.1%}")
        st.progress(prob)
        
        # Risk assessment with business rules
        if prob > 0.6:
            st.error("🔴 HIGH RISK: Immediate action required!")
            st.write("**Recommended actions:**")
            st.write("- Personal retention call")
            st.write("- 20% discount offer")
            st.write("- Priority support")
        elif prob > 0.3:
            st.warning("🟡 MEDIUM RISK: Monitor closely")
            st.write("**Suggested actions:**")
            st.write("- Satisfaction survey")
            st.write("- Loyalty program offer")
        else:
            st.success("🟢 LOW RISK: Healthy customer")
            st.write("**Maintenance suggestions:**")
            st.write("- Regular check-ins")
            st.write("- Upsell opportunities")
            
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.write("Troubleshooting steps:")
        st.write("1. Check all inputs are valid")
        st.write("2. Verify model file integrity")
        st.write("3. Compare with training data")

# --- Model Improvement Section ---
with st.expander("🛠️ How to improve this model"):
    st.write("""
    **For better predictions:**
    
    1. **Retrain with regularization:**
    ```python
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=0.1, penalty='l1', solver='liblinear')
    model.fit(X_train, y_train)
    ```
    
    2. **Add feature engineering:**
    ```python
    df['ValueScore'] = df['MonthlyCharges'] / df['tenure']
    ```
    
    3. **Balance your dataset:**
    ```python
    from imblearn.over_sampling import SMOTE
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)
    ```
    """)

# --- User Guide ---
with st.expander("📚 User Guide"):
    st.write("""
    **How to use this tool:**
    1. Fill all customer details in the sidebar
    2. Click 'Predict Churn Risk'
    3. Review the probability and risk level
    4. Check 'Prediction Analysis' for details
    
    **Test scenarios:**
    - High Risk: 1 month tenure, $200, Month-to-month
    - Low Risk: 24 months, $50, Two year contract
    """)