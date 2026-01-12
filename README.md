# Credit Card Fraud Detection

End-to-end Machine Learning project for detecting fraudulent credit card transactions using classical ML models, with a strong focus on handling highly imbalanced data.

This project is based on the well-known Kaggle Credit Card Fraud Detection dataset and was built as a learning-oriented yet production-style ML pipeline suitable for portfolio and CV usage.

---

## 📌 Project Overview

- Binary classification problem (Fraud vs Non-Fraud)
- Extremely imbalanced dataset
- Full ML pipeline including:
  - Data preparation
  - Preprocessing & feature handling
  - Model training and validation
  - Threshold tuning
  - Model evaluation on a held-out test set
  - Model persistence and logging

---

## 📂 Dataset

- Source: Kaggle – Credit Card Fraud Detection  
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

- The dataset is used **as-is**
- All preprocessing, splitting, and feature engineering are handled **inside the code**
- The `data/` directory in this repository is kept minimal; the dataset should be downloaded manually from Kaggle

---

## 🧠 Models Used

- Logistic Regression
- Random Forest
- Soft Voting Classifier (Ensemble)

All trained models are saved to disk and reused during testing.

---

## ⚖️ Handling Imbalanced Data

- Under-sampling strategy
- Stratified Train / Validation / Test split
- K-Fold strategy integrated into the training pipeline
- Custom **probability threshold tuning** instead of relying on the default 0.5 threshold

---

## 📊 Evaluation Metrics

Models are evaluated using metrics appropriate for imbalanced classification problems:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

### ✅ Test Set Results (Summary)

Best-performing model on the test set:

- **Model:** Random Forest  
- **Best Threshold:** 0.56  
- **F1-score:** 0.80  
- **ROC-AUC:** 0.95  

(Other models are evaluated and logged for comparison.)

---

## 🧪 Data Splitting Strategy

- Stratified splitting to preserve class distribution
- Three-way split:
  - Train
  - Validation
  - Test

Approximate proportions:
- Train: 64%
- Validation: 16%
- Test: 20%

---

## 🗂️ Project Structure
```bash
.
├── data/
│   ├── data.csv
│   └── .gitkeep
├── models/
│   ├── lr.joblib
│   ├── rf.joblib
│   └── voting.joblib
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── __init__.py
│   ├── config.yaml
│   ├── Data_utils.py
│   ├── training.py
│   ├── eval.py
│   ├── helper_fun.py
│   └── main.py
├── training.log
├── requirements.txt
├── README.md
└── .gitignore
```





## ▶️ How to Run

All commands should be executed from the **project root directory**.

> ⚠️ The project is designed to be executed as a Python module to ensure correct imports and configuration paths.

### Train a model
```bash
python -m src.main --model logistic_regression --mode train
```
### test a model
```bash
python -m src.main --model logistic_regression --mode test
```
### Available Models

- `logistic_regression`
- `random_forest`
- `voting`

---

## 📜 Logging

- Centralized logging using Python’s `logging` module
- Logged information includes:
  - Model name
  - Best threshold
  - Evaluation metrics
- Logs are stored in:
```bash
training.log
```
---

## 🎯 Project Goal

This is a **learning-oriented end-to-end machine learning project** designed to demonstrate:

- Proper ML pipeline design  
- Handling of real-world imbalanced datasets  
- Clean project organization  
- Reproducible training and evaluation  
- Practical use of threshold tuning and logging  

The project is intended for inclusion in a professional CV or portfolio.

---

## 📦 Requirements

All required dependencies are listed in:

```bash
requirements.txt
```
## 👤 Author

**Amir**








