# 💓 Heart Disease Prediction App

An interactive and mobile-friendly web application to predict the presence of heart disease based on medical attributes. Built using **Streamlit** and trained using **Optuna-optimized machine learning models**.

### 🚀 Demo
- **Streamlit Cloud**: [https://share.streamlit.io/user/davidonuna](https://share.streamlit.io/user/davidonuna)
- **Render**: https://heart-disease-predictor-iivj.onrender.com/

---

## 📦 Features

- Interactive UI for user input (mobile-ready)
- Predicts presence of heart disease using a trained ML model
- Model trained with Optuna-based hyperparameter tuning
- Visual feedback: prediction confidence and progress bar
- Custom styling via CSS
- Docker support for easy deployment
- Tab-based navigation (Prediction, Model Info, About)

---

## 🧠 Model Training

The model is trained using multiple classifiers (`RandomForest`, `XGBoost`, `SVM`, etc.) and optimized using **Optuna**. The best model is selected and saved as a `.pkl` file for deployment.

> Training script: `train_model.py`

---

## 📁 Project Structure

```
heart_disease_prediction/
├── app.py                 # Main Streamlit application
├── train_model.py         # Model training script
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker container definition
├── .dockerignore         # Docker ignore file
├── model/
│   └── best_model_pipeline.pkl  # Trained model
├── assets/
│   └── styles.css        # Custom CSS styling
├── reports/
│   ├── confusion_matrix.png
│   └── classification_report.txt
└── README.md
```

---

## 🚀 Run Locally

### Option 1: Using Docker (Recommended)

```bash
# Build the Docker image
docker build -t heart-disease-prediction .

# Run the container
docker run -p 8501:8501 heart-disease-prediction
```

### Option 2: Without Docker

```bash
# Clone the repo
git clone https://github.com/davidonuna/heart_disease_prediction.git
cd heart_disease_prediction

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## ⚙️ Configuration

The following environment variables can be set:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to model file | `model/best_model_pipeline.pkl` |
| `STYLES_PATH` | Path to CSS file | `assets/styles.css` |

---

## 🛠️ Development

### Retrain the Model

```bash
# Ensure PostgreSQL is running with the heart_disease_data table
python train_model.py
```

### Run Tests

```bash
pytest tests/
```

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

- Dataset: Heart Disease Dataset from UCI ML Repository
- Hospital: Nyali Children Hospital & Bi-Cross Heart Clinic

---

Built with ❤️ using Streamlit and Scikit-learn
