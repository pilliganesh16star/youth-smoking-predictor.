import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

st.title("Youth Predictor App 👨‍🎓")

st.write("This app predicts based on the trained ML model you provided.")

# Example: Replace with your actual input fields
age = st.number_input("Enter Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Select Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["High School", "Undergraduate", "Graduate"])

if st.button("Predict"):
    # Convert categorical to numeric (adjust to your training preprocessing)
    gender_val = 1 if gender == "Male" else 0
    edu_map = {"High School": 0, "Undergraduate": 1, "Graduate": 2}
    edu_val = edu_map[education]

    input_data = np.array([[age, gender_val, edu_val]])

    prediction = model.predict(input_data)
    st.success(f"✅ Model Prediction: {prediction[0]}")
