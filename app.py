import streamlit as st
import numpy as np
import pickle

st.title("❤️ Heart Disease Prediction App")

# Load model and scaler
model = pickle.load(open("heart_disease_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.header("Enter Patient Details")

# User inputs
age = st.number_input("Age", 1, 120, 45)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG results", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 50, 250, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Convert to numeric
sex = 1 if sex == "Male" else 0

# Predict button
if st.button("Predict"):
    # Prepare input
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    scaled_input = scaler.transform(user_input)
    
    # Make prediction
    prediction = model.predict(scaled_input)
    
    if prediction[0] == 1:
        st.error("⚠️ Likely to have heart disease")
    else:
        st.success("✅ Unlikely to have heart disease")







