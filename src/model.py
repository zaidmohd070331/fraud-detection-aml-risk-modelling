# model.py
# Fraud detection model training and evaluation
# Models: Isolation Forest (anomaly detection) + Logistic Regression (classification)

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_score,
                             recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

FEATURE_COLS = [
    'amount', 'hour_of_day', 'day_of_week',
    'is_high_risk_hour', 'is_weekend',
    'is_high_risk_country', 'is_large_amount',
    'is_unknown_merchant'
]
TARGET_COL = 'is_fraud'

def train_isolation_forest(df):
    """
    Unsupervised anomaly detection using Isolation Forest.
    Flags transactions as anomalous without requiring labels.
    Replicates AML screening logic for unknown patterns.
    """
    print("\n=== Isolation Forest — Anomaly Detection ===")
    X = df[FEATURE_COLS]
    model = IsolationForest(contamination=0.3, random_state=42)
    df['anomaly_score'] = model.fit_predict(X)
    df['is_anomaly'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)
    anomaly_rate = df['is_anomaly'].mean() * 100
    print(f"Anomaly detection rate: {anomaly_rate:.1f}%")
    print(df[['transaction_id', 'amount', 'is_anomaly']].to_string())
    return model, df

def train_classifier(df):
    """
    Supervised classification using Logistic Regression.
    SMOTE applied to handle severe class imbalance —
    critical in fraud detection to reduce false negatives.
    """
    print("\n=== Logistic Regression — Fraud Classification ===")
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"After SMOTE — Class distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred,
          target_names=['Legitimate', 'Fraudulent']))
    print(f"ROC-AUC Score : {roc_auc_score(y_test, y_prob):.3f}")
    print(f"Precision     : {precision_score(y_test, y_pred):.3f}")
    print(f"Recall        : {recall_score(y_test, y_pred):.3f}")
    print(f"F1 Score      : {f1_score(y_test, y_pred):.3f}")

    joblib.dump(model, 'fraud_classifier.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\nModel saved as fraud_classifier.pkl")
    return model, scaler

if __name__ == "__main__":
    from preprocess import load_data, engineer_features
    df = load_data("../data/sample_transactions.csv")
    df = engineer_features(df)
    iso_model, df = train_isolation_forest(df)
    clf_model, scaler = train_classifier(df)
