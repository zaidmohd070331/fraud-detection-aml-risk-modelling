# 💳 Transaction Fraud Detection & AML Risk Modelling

> End-to-end fraud detection and AML risk scoring on transactional data — replicating Financial Crime Operations and Controls frameworks used in banking institutions.

---

## 📌 Business Problem

Financial institutions process millions of transactions daily. Identifying fraudulent activity and AML risks manually is impossible at scale. This project builds an automated detection and risk scoring pipeline that mirrors real-world watchlist-matching, payment screening, and AML triage workflows.

---

## 🎯 Objectives

- Detect suspicious transactions using unsupervised anomaly detection
- Classify fraudulent transactions using supervised ML with SMOTE
- Score each transaction on AML risk dimensions (amount, country, timing, merchant)
- Flag high-risk transactions for escalation — replicating watchlist-matching logic
- Produce MI-ready outputs for compliance and senior stakeholder review

---

## 🏗️ Project Architecture

```
Raw Transactions → Feature Engineering → Isolation Forest (Anomaly Detection)
                                       → Logistic Regression + SMOTE (Classification)
                                       → AML Risk Scorer → MI Report Output
```

---

## 🔬 Methodology

### 1. Data
- Synthetic transactional dataset (100K+ records structure)
- Fields: transaction_id, account_id, amount, transaction_type, merchant_category, hour_of_day, country, is_fraud

### 2. Feature Engineering
- High-risk hour flag (00:00–05:00 transactions)
- High-risk country flag (FATF high-risk jurisdictions)
- Large amount flag (>£5,000 threshold)
- Unknown/crypto merchant flag
- Weekend transaction flag

### 3. Isolation Forest — Anomaly Detection
- Unsupervised model flagging statistically anomalous transactions
- No labels required — replicates real-world AML screening on unlabelled data
- Contamination parameter tuned to expected fraud rate

### 4. Logistic Regression — Classification
- Supervised classifier trained on labelled fraud data
- SMOTE applied to address severe class imbalance
- Recall improved by ~22% over baseline — critical for reducing false negatives in financial crime detection

### 5. AML Risk Scorer
- Composite risk score (0–100) across 4 dimensions: amount, country risk, transaction timing, merchant category
- Labels: Critical / High / Medium / Low
- Watchlist-matching flag for high-risk country + high-value transactions

---

## 📊 Results

| Metric | Score |
|---|---|
| Classification Accuracy | 84% |
| Recall (Fraud Class) | 88% |
| ROC-AUC Score | 0.91 |
| SMOTE Recall Improvement | ~22% over baseline |

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10 |
| Anomaly Detection | Scikit-learn Isolation Forest |
| Classification | Logistic Regression + SMOTE (imbalanced-learn) |
| Feature Engineering | Pandas, NumPy |
| Evaluation | Precision, Recall, ROC-AUC, Confusion Matrix |
| Visualisation | Matplotlib, Seaborn, Power BI |

---

## 📂 Repository Structure

```
├── data/
│   ├── sample_transactions.csv       # Synthetic transaction data
│   └── aml_risk_labels.csv           # AML risk labels and scores
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   └── 02_model_training_evaluation.ipynb
├── src/
│   ├── preprocess.py                 # Feature engineering pipeline
│   ├── model.py                      # Isolation Forest + LR classifier
│   └── risk_scorer.py                # AML risk scoring logic
├── reports/                          # MI reports and dashboard screenshots
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
git clone https://github.com/zaidmohd070331/fraud-detection-aml-risk-modelling.git
cd fraud-detection-aml-risk-modelling
pip install -r requirements.txt
python src/preprocess.py
python src/model.py
python src/risk_scorer.py
```

---

## 💡 Key Takeaways

- SMOTE is essential in fraud detection — raw class imbalance causes models to ignore minority (fraud) class
- Isolation Forest is powerful for AML screening where labelled data is unavailable
- Risk scoring dimensions mirror real CCO/Financial Crime triage frameworks
- All outputs are structured for MI reporting to compliance stakeholders

---

