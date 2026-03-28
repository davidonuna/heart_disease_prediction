
import os
import streamlit as st
import pickle
import pandas as pd

MODEL_PATH = os.getenv('MODEL_PATH', 'model/best_model_pipeline.pkl')
STYLES_PATH = os.getenv('STYLES_PATH', 'assets/styles.css')

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {model_path}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_styles(styles_path):
    try:
        with open(styles_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        st.warning(f"⚠️ Styles file not found: {styles_path}")
        return ""
    except Exception as e:
        st.warning(f"⚠️ Error loading styles: {str(e)}")
        return ""

model = load_model(MODEL_PATH)
styles = load_styles(STYLES_PATH)
if styles:
    st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)

if model is None:
    st.stop()

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=64)
    st.markdown("### 🏥 Nyali Children Hospital")
    st.caption("Bi-Cross Heart Clinic")
    st.divider()
    st.markdown("#### Quick Links")
    st.page_link("https://example.com", label="Learn about Heart Disease", icon="📖")
    st.page_link("https://example.com", label="Contact Hospital", icon="📞")
    st.divider()
    st.caption("💡 Tips: Enter accurate values for better predictions")

tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Model Info", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📝 Patient Information")
        
        with st.form("user_input_form"):
            age = st.number_input("Age", min_value=1, max_value=120, value=30, help="Patient age in years")
            sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
            
            st.markdown("#### Cardiac Metrics")
            chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"], help="Type of chest pain experienced")
            resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", value=120, min_value=80, max_value=200, help="Resting blood pressure in mm Hg")
            serum_cholesterol = st.number_input("Serum Cholesterol (mg/dl)", value=200, min_value=100, max_value=400, help="Serum cholesterol in mg/dl")
            fasting_blood_sugar = st.radio("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"], horizontal=True)
            
            st.markdown("#### ECG & Exercise")
            resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"], help="Resting electrocardiographic results")
            max_heart_rate = st.number_input("Max Heart Rate Achieved", value=150, min_value=60, max_value=220, help="Maximum heart rate achieved during exercise")
            exercise_induced_angina = st.radio("Exercise Induced Angina", ["Yes", "No"], horizontal=True, help="Exercise-induced chest pain")
            
            st.markdown("#### Additional Tests")
            oldpeak = st.number_input("Oldpeak (ST depression)", value=1.0, min_value=0.0, max_value=10.0, step=0.1, help="ST depression induced by exercise")
            slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"], help="Peak exercise ST segment slope")
            num_vessels = st.selectbox("Number of Major Vessels", [0, 1, 2, 3], help="Number of major vessels colored by fluoroscopy")
            thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"], help="Thalassemia type")
            
            c1, c2 = st.columns(2)
            with c1:
                submit = st.form_submit_button("🔍 Predict", use_container_width=True)
            with c2:
                clear = st.form_submit_button("↺ Clear", use_container_width=True)
        
        if clear:
            st.rerun()

    with col2:
        if submit:
            try:
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
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                prob_heart_disease = probabilities[1]
                prob_no_disease = probabilities[0]
                
                st.markdown("### 📊 Prediction Results")
                
                if prediction == 1:
                    st.error("💔 **Heart Disease Detected**")
                    st.progress(int(prob_heart_disease * 100))
                    st.markdown(f"**Confidence:** {prob_heart_disease * 100:.1f}%")
                    
                    st.warning("⚠️ Please consult a cardiologist for further evaluation.")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Heart Disease Risk", f"{prob_heart_disease * 100:.1f}%", delta="High", delta_color="inverse")
                    with col_b:
                        st.metric("Healthy", f"{prob_no_disease * 100:.1f}%")
                else:
                    st.success("💚 **No Heart Disease Detected**")
                    st.progress(int(prob_no_disease * 100))
                    st.markdown(f"**Confidence:** {prob_no_disease * 100:.1f}%")
                    
                    st.info("✅ Continue maintaining a healthy lifestyle!")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Heart Disease Risk", f"{prob_heart_disease * 100:.1f}%")
                    with col_b:
                        st.metric("Healthy", f"{prob_no_disease * 100:.1f}%", delta="Low", delta_color="normal")
                
                with st.expander("📋 View Input Summary"):
                    st.json(input_data)
                    
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
        else:
            st.markdown("""
            <div class="placeholder-box">
                <h3>👈 Enter Patient Details</h3>
                <p>Fill in the form and click <b>Predict</b> to see results</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📈 Risk Factors Overview")
            risk_factors = pd.DataFrame({
                "Factor": ["Age > 45 (M) / 55 (F)", "High Blood Pressure", "High Cholesterol", "Smoking", "Obesity", "Diabetes"],
                "Risk Level": ["Medium", "High", "High", "High", "Medium", "High"]
            })
            st.table(risk_factors)

with tab2:
    st.markdown("### 📊 Model Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Type", "XGBoost Classifier")
        st.metric("Accuracy", "~87%")
    with col2:
        st.metric("Features Used", "13")
        st.metric("Cross-Validation", "5-fold")
    
    st.divider()
    st.markdown("#### 🎯 Key Features Used")
    feature_desc = pd.DataFrame({
        "Feature": ["cp (Chest Pain)", "thalach (Max HR)", "oldpeak", "ca (Vessels)", "thal (Thalassemia)", "exang (Exercise Angina)"],
        "Importance": ["Very High", "High", "High", "Medium", "Medium", "Medium"]
    })
    st.table(feature_desc)

with tab3:
    st.markdown("### ℹ️ About This App")
    
    st.markdown("""
    #### Heart Disease Prediction App
    
    This application uses machine learning to predict the likelihood of heart disease based on clinical parameters.
    
    **Hospital:** Nyali Children Hospital & Bi-Cross Heart Clinic
    
    **Disclaimer:** This is a screening tool only. Please consult a qualified cardiologist for proper diagnosis and treatment.
    
    ---
    
    #### How to Use
    1. Enter patient information in the Prediction tab
    2. Click "Predict" to get results
    3. Review the confidence level
    4. Consult a doctor if heart disease is detected
    """)
    
    st.info("💻 Built with Streamlit | 🤖 Machine Learning Model")
