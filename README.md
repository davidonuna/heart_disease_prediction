# 💓 Heart Disease Prediction App

An interactive and mobile-friendly web application to predict the presence of heart disease based on medical attributes. Built using **Streamlit** and trained using **Optuna-optimized machine learning models**.

### 🚀 Demo
- **Streamlit Cloud**: [https://share.streamlit.io/user/davidonuna](https://share.streamlit.io/user/davidonuna)
- **Render**: [https://heart-disease-predictor-iivj.onrender.com/]

---

## 📦 Features

- Interactive UI for user input (mobile-ready)
- Predicts presence of heart disease using a trained ML model
- Model trained with Optuna-based hyperparameter tuning
- Visual feedback: prediction confidence and progress bar
- Custom styling via CSS

---

## 🧠 Model Training

The model is trained using multiple classifiers (`RandomForest`, `XGBoost`, `SVM`, etc.) and optimized using **Optuna**. The best model is selected and saved as a `.pkl` file for deployment.

> Training script: `train_model.py`

---

## 📁 Project Structure

<!-- heart-disease-prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── Dockerfile
├── model/
│ └── best_model_pipeline.pkl
├── data/
│ └── heart_disease.csv
├── assets/
│ └── styles.css
├── reports/
│ ├── confusion_matrix.png
│ └── classification_report.txt
└── README.md -->


---

## 🚀 Run Locally

1. Clone the repo
```bash
git clone https://github.com/davidonuna/heart_disease_prediction.git
cd heart-disease-prediction


pip install -r requirements.txt


streamlit run app.py
