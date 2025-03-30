# Bank Marketing Predictive Model

This project predicts whether a client will subscribe to a term deposit based on the Bank Marketing Dataset from Kaggle. It uses a Random Forest Classifier, includes data cleaning, and provides insights through feature importance and ROC curves.

## Features
- Downloads data directly from Kaggle using `kagglehub`.
- Cleans data by handling duplicates and capping outliers.
- Evaluates model performance with accuracy, precision, recall, and AUC.
- Visualizes feature importance and ROC curve.

## Requirements
See `requirements.txt` for dependencies.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the script: `python bank_predictive_model.py`

## Dataset
- Source: [Bank Marketing Dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)
- Target: Predict `deposit` (yes/no).

## Results
- Accuracy: ~0.80–0.85
- AUC: ~0.85–0.90
- Key Features: `duration`, `balance`, `age`

## Author
Michael Cain - linkedin.com/in/michael-c-b727b6107/
