# preprocess.py
# Data preprocessing and feature engineering for fraud detection

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def engineer_features(df):
    """
    Feature engineering for AML/fraud detection.
    Creates risk-relevant features from raw transaction data.
    """
    df = df.copy()

    # High risk hour flag (late night transactions: 0am - 5am)
    df['is_high_risk_hour'] = df['hour_of_day'].apply(
        lambda x: 1 if x <= 5 else 0
    )

    # Weekend flag
    df['is_weekend'] = df['day_of_week'].apply(
        lambda x: 1 if x >= 5 else 0
    )

    # High risk country flag
    high_risk_countries = ['NG', 'RU', 'CN', 'KP', 'IR']
    df['is_high_risk_country'] = df['country'].apply(
        lambda x: 1 if x in high_risk_countries else 0
    )

    # Large amount flag (above 5000)
    df['is_large_amount'] = df['amount'].apply(
        lambda x: 1 if x > 5000 else 0
    )

    # Unknown merchant flag
    df['is_unknown_merchant'] = df['merchant_category'].apply(
        lambda x: 1 if x == 'Unknown' else 0
    )

    return df

def encode_and_scale(df, cat_cols, num_cols):
    """Encode categorical columns and scale numeric columns."""
    df = df.copy()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

if __name__ == "__main__":
    df = load_data("../data/sample_transactions.csv")
    df = engineer_features(df)
    print("Features engineered successfully.")
    print(df[['transaction_id', 'is_high_risk_hour',
              'is_weekend', 'is_high_risk_country',
              'is_large_amount', 'is_unknown_merchant']].to_string())
