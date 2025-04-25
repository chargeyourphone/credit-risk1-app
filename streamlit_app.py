import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("catboost_model.pkl")

# Streamlit app
st.title("🧠 Credit Risk Prediction App")
st.markdown("Predict credit risk using customer information")

# Input fields organized in columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ['male', 'female'])
    job = st.selectbox("Job", [0, 1, 2, 3])
    housing = st.selectbox("Housing", ['own', 'free', 'rent'])
    purpose = st.selectbox("Purpose", [
        'radio/TV', 'education', 'furniture/equipment',
        'car', 'business', 'domestic appliances',
        'repairs', 'vacation/others'
    ])

with col2:
    saving_accounts = st.selectbox("Saving accounts", ['little', 'moderate', 'quite rich', 'rich'])
    checking_account = st.selectbox("Checking account", ['little', 'moderate', 'rich'])
    credit_amount = st.number_input("Credit amount (USD)", min_value=0, value=1000, step=100)
    duration = st.number_input("Duration (months)", min_value=1, value=12, step=1)

# Prepare input data with correct column order
input_data = {
    'Age': age,
    'Sex': sex,
    'Job': job,
    'Housing': housing,
    'Saving accounts': saving_accounts,
    'Checking account': checking_account,
    'Credit amount': credit_amount,
    'Duration': duration,
    'Purpose': purpose
}


def predict_risk(data):
    """Make prediction with proper data formatting"""
    try:
        # Create DataFrame with same column order as training data
        df = pd.DataFrame([data], columns=input_data.keys())

        # Convert categorical columns
        cat_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
        df[cat_cols] = df[cat_cols].astype('category')

        # Make prediction
        prediction = model.predict(df)[0]
        return "✅ Good Credit Risk" if prediction == 1 else "❌ Bad Credit Risk"
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return "Prediction Failed"


# Prediction button
if st.button("Predict Risk"):
    result = predict_risk(input_data)
    st.subheader("Prediction Result")
    st.markdown(f"## {result}")
    st.info(
        "Note: This prediction is based on historical data patterns and should be used as one of multiple decision factors.")