
# train_model.py

import pandas as pd
import pickle
import optuna

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import os          

# Load dataset
df = pd.read_csv("data/heart_disease.csv")
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to get model from name and params
def get_model(name, params):
    if name == 'SVC':
        return SVC(**params, probability=True)
    elif name == 'LogisticRegression':
        return LogisticRegression(**params, max_iter=1000)
    elif name == 'RandomForest':
        return RandomForestClassifier(**params)
    elif name == 'GradientBoosting':
        return GradientBoostingClassifier(**params)
    elif name == 'LightGBM':
        return lgb.LGBMClassifier(**params)
    elif name == 'XGBoost':
        return xgb.XGBClassifier(**params, eval_metric='logloss')
    elif name == 'DecisionTree':
        return DecisionTreeClassifier(**params)

# Optuna objective function
def objective(trial):
    model_name = trial.suggest_categorical('model', [
        'SVC', 'LogisticRegression', 'RandomForest',
        'GradientBoosting', 'LightGBM', 'XGBoost', 'DecisionTree'
    ])

    if model_name == 'SVC':
        params = {
            'C': trial.suggest_float('C', 1e-5, 1e5, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
        }

    elif model_name == 'LogisticRegression':
        params = {
            'C': trial.suggest_float('C', 1e-5, 1e5, log=True),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
        }

    elif model_name == 'RandomForest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        }

    elif model_name == 'GradientBoosting':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
        }

    elif model_name == 'LightGBM':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        }

    elif model_name == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }

    elif model_name == 'DecisionTree':
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        }

    model = get_model(model_name, params)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

    score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy').mean()
    return score

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Get best model and parameters
best_params = study.best_params
best_model_name = best_params.pop('model')
final_model = get_model(best_model_name, best_params)

# Final pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', final_model)
])

pipeline.fit(X_train, y_train)

# Save model
with open('model/best_model_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
    

# Create output directory if it doesn't exist
os.makedirs("reports", exist_ok=True)

# Predict on test set
y_pred = pipeline.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("📊 Confusion Matrix:")
print(cm)

# Save classification report
report = classification_report(y_test, y_pred, target_names=['No Disease (0)', 'Disease (1)'])
print("\n📋 Classification Report:")
print(report)

with open("reports/classification_report.txt", "w") as f:
    f.write("Classification Report\n")
    f.write("=====================\n")
    f.write(report)

# Plot confusion matrix with labels
labels = ['No Disease (0)', 'Disease (1)']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()

# Save confusion matrix plot
plt.savefig("reports/confusion_matrix.png")
plt.show()

   

print(f"✅ Best model '{best_model_name}' saved successfully.")
