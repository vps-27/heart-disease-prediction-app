# 🫀 Heart Disease Risk Prediction App

> A machine learning-powered web application that predicts heart disease risk based on key clinical parameters — built with Logistic Regression and deployed via Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heart-disease-prediction-app-ofpjjvdwabsfax24x5g8nj.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🔍 Overview

Cardiovascular disease is one of the leading causes of death globally. Early detection can significantly improve outcomes. This project applies **supervised machine learning** to predict whether a patient is at risk of heart disease, based on inputs such as age, cholesterol levels, resting blood pressure, and more.

The app provides an **instant, interactive risk prediction** through a clean Streamlit interface — no technical knowledge required to use it.

---

## 🚀 Live Demo

👉 **[Try the App Here](https://heart-disease-prediction-app-ofpjjvdwabsfax24x5g8nj.streamlit.app/)**

---

## 🖥️ App Screenshots

**Input Form**

![Input Form](https://github.com/user-attachments/assets/a42ef868-9dd1-4dcc-adb5-2ed4ca3a06dc)

**Prediction Result — At Risk**

![Prediction Result 1](https://github.com/user-attachments/assets/fb4925a8-8396-4383-93af-4e0ba41d2ff9)

**Prediction Result — No Risk**

![Prediction Result 2](https://github.com/user-attachments/assets/35ea0392-00eb-4679-85cd-aabf57753793)

**Additional View**

![App View](https://github.com/user-attachments/assets/8e1ce81d-590c-45bb-834a-d71c1416767a)

---

## ⚙️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Scikit-learn | Model training & evaluation |
| Pandas & NumPy | Data preprocessing & feature engineering |
| Streamlit | Web app deployment |
| Pickle | Model serialization |

---

## 🧠 Model Details

| Parameter | Detail |
|---|---|
| Algorithm | Logistic Regression |
| Task | Binary Classification (Disease / No Disease) |
| Dataset | Heart Disease Dataset (Kaggle / UCI Repository) |
| Preprocessing | StandardScaler for feature normalization |
| Output | Risk prediction with likelihood |

**Input Features:**
- Age, Sex, Chest Pain Type
- Resting Blood Pressure, Serum Cholesterol
- Fasting Blood Sugar, Resting ECG
- Max Heart Rate, Exercise-Induced Angina
- ST Depression, Slope, Major Vessels, Thalassemia

---
---
## 🛠️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/vps-27/heart-disease-prediction-app.git
cd heart-disease-prediction-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

---

## 📌 Key Learnings

- End-to-end ML pipeline: data cleaning → preprocessing → model training → evaluation → deployment
- Applied `StandardScaler` for feature normalization to improve Logistic Regression performance
- Integrated a trained `.pkl` model into a Streamlit web interface for real-time predictions
- Deployed a live ML app on Streamlit Cloud

---

## 👩‍💻 Author

**Vishnu Priya Srivathsan**  
B.E. Electrical and Electronics Engineering — Sri Sai Ram Engineering College, Chennai  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/vishnu-priya-s-6b5349292)

---

## 📄 License

This project is licensed under the MIT License.

## 📁 Project Structure

