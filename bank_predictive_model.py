# bank_predictive_model.py
"""
Predictive Modeling for Bank Marketing Dataset
Predicts whether a client subscribes to a term deposit using Random Forest.
Includes data cleaning, feature importance, and model evaluation.
"""

import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_dataset():
    """Download and load the Bank Marketing Dataset from Kaggle."""
    path = kagglehub.dataset_download("janiobachmann/bank-marketing-dataset")
    csv_path = os.path.join(path, "bank.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"bank.csv not found at {csv_path}")
    return pd.read_csv(csv_path)

def clean_data(df):
    """Clean and organize the dataset."""
    print("Initial Shape:", df.shape)
    
    # Check for missing values
    print("\nMissing Values:\n", df.isnull().sum())
    
    # Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicates. New shape: {df.shape}")
    
    # Cap outliers in 'balance'
    Q1 = df['balance'].quantile(0.25)
    Q3 = df['balance'].quantile(0.75)
    upper_bound = Q3 + 1.5 * (Q3 - Q1)
    df['balance'] = df['balance'].clip(upper=upper_bound)
    print(f"Capped 'balance' at {upper_bound}")
    
    return df

def preprocess_data(df):
    """Preprocess features and target for modeling."""
    X = df.drop('deposit', axis=1)
    y = df['deposit']
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col])
    
    # Encode target
    y = LabelEncoder().fit_transform(y)
    return X, y

def train_and_evaluate(X, y):
    """Train Random Forest model and evaluate performance."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_pred_proba)
    }
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return rf_model, X_test, y_test, y_pred_proba, metrics

def visualize_results(rf_model, X, y_test, y_pred_proba):
    """Visualize feature importance and ROC curve."""
    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\nFeature Importance:\n", feature_importance)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance in Random Forest Model')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load and clean data
    data = load_dataset()
    data = clean_data(data)
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Train and evaluate model
    rf_model, X_test, y_test, y_pred_proba, metrics = train_and_evaluate(X, y)
    
    # Visualize results
    visualize_results(rf_model, X, y_test, y_pred_proba)
