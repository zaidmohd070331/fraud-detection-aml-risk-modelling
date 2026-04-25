# risk_scorer.py
# AML risk scoring — replicates watchlist-matching and
# payment screening logic used in Financial Crime Operations

import pandas as pd
import numpy as np

# Risk weights for scoring dimensions
AMOUNT_WEIGHT = 0.35
COUNTRY_WEIGHT = 0.25
HOUR_WEIGHT = 0.20
MERCHANT_WEIGHT = 0.20

# High risk country list (aligned to FATF high-risk jurisdictions)
HIGH_RISK_COUNTRIES = ['NG', 'RU', 'CN', 'KP', 'IR', 'MM', 'PK']

def score_transaction(row):
    """
    Calculate an AML risk score (0-100) for each transaction.
    Higher score = higher suspicion of financial crime activity.
    """
    score = 0

    # Amount component (0-35)
    if row['amount'] > 50000:
        score += 35
    elif row['amount'] > 10000:
        score += 25
    elif row['amount'] > 5000:
        score += 15
    else:
        score += 2

    # Country risk component (0-25)
    if row['country'] in HIGH_RISK_COUNTRIES:
        score += 25
    else:
        score += 0

    # Hour risk component (0-20)
    if row['hour_of_day'] <= 5:
        score += 20
    elif row['hour_of_day'] <= 8:
        score += 10
    else:
        score += 0

    # Merchant category component (0-20)
    if row['merchant_category'] in ['Unknown', 'Crypto']:
        score += 20
    elif row['merchant_category'] == 'Wire Transfer':
        score += 10
    else:
        score += 0

    return min(score, 100)

def assign_risk_label(score):
    """Map numeric score to AML risk label."""
    if score >= 80:
        return 'Critical'
    elif score >= 60:
        return 'High'
    elif score >= 40:
        return 'Medium'
    else:
        return 'Low'

def flag_watchlist(row):
    """
    Simulate watchlist-matching logic.
    Flags transactions from high-risk countries above threshold.
    """
    if row['country'] in HIGH_RISK_COUNTRIES and row['amount'] > 5000:
        return 1
    return 0

def score_dataframe(df):
    df = df.copy()
    df['aml_risk_score'] = df.apply(score_transaction, axis=1)
    df['aml_risk_label'] = df['aml_risk_score'].apply(assign_risk_label)
    df['watchlist_flag'] = df.apply(flag_watchlist, axis=1)
    return df.sort_values('aml_risk_score', ascending=False)

if __name__ == "__main__":
    df = pd.read_csv("../data/sample_transactions.csv")
    result = score_dataframe(df)
    print("\n=== AML Risk Scoring Results ===")
    print(result[['transaction_id', 'amount', 'country',
                  'aml_risk_score', 'aml_risk_label',
                  'watchlist_flag']].to_string())
