
# app.py

import streamlit as st
import pickle
import pandas as pd

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# Now it's safe to call other Streamlit commands
# Load the trained model
with open('model/best_model_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

# Load and apply CSS
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# App Title and description
st.title("💓 Heart Disease Prediction")
st.caption("Mobile-Friendly App | Nyali Children Hospital & Bi-Cross Heart Clinic")

# Form Inputs
with st.form("user_input_form"):
    st.markdown("### 📝 Enter Your Details")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
    serum_cholesterol = st.number_input("Serum Cholesterol (mg/dl)", value=200)
    fasting_blood_sugar = st.radio("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"], horizontal=True)
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
    max_heart_rate = st.number_input("Max Heart Rate Achieved", value=150)
    exercise_induced_angina = st.radio("Exercise Induced Angina", ["Yes", "No"], horizontal=True)
    oldpeak = st.number_input("Oldpeak (ST depression)", value=1.0)
    slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
    num_vessels = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    submit = st.form_submit_button("🔍 Predict")

# Encode Inputs
input_data = {
    "age": age,
    "sex": 1 if sex == "Male" else 0,
    "cp": ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain),
    "trestbps": resting_blood_pressure,
    "chol": serum_cholesterol,
    "fbs": 1 if fasting_blood_sugar == "Yes" else 0,
    "restecg": ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(resting_ecg),
    "thalach": max_heart_rate,
    "exang": 1 if exercise_induced_angina == "Yes" else 0,
    "oldpeak": oldpeak,
    "slope": ["Upsloping", "Flat", "Downsloping"].index(slope),
    "ca": num_vessels,
    "thal": ["Normal", "Fixed Defect", "Reversible Defect"].index(thalassemia)
}

input_df = pd.DataFrame([input_data])

# Prediction
if submit:
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    prob_heart_disease = probabilities[1]
    prob_no_disease = probabilities[0]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("💔 Heart Disease Detected")
        st.markdown(f"🔺 **Model Confidence:** `{prob_heart_disease * 100:.2f}%`")
        st.progress(int(prob_heart_disease * 100))
    else:
        st.success("💚 No Heart Disease Detected")
        st.markdown(f"✅ **Model Confidence:** `{prob_no_disease * 100:.2f}%`")
        st.progress(int(prob_no_disease * 100))

