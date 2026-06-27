# Email Spam Detection

A machine learning pipeline that classifies emails as spam or not spam using Logistic Regression, evaluated with a comprehensive set of classification metrics.

---

## Overview

This notebook trains a binary text classification model on a labelled email dataset. It handles preprocessing, feature scaling, model training, and evaluation — including a confusion matrix and ROC curve.

---

## Workflow

```
Load Dataset (CSV)
      ↓
Drop Non-Numeric Columns
      ↓
Train/Test Split (80/20)
      ↓
Feature Scaling (StandardScaler)
      ↓
Model Training (Logistic Regression)
      ↓
Prediction & Evaluation
      ↓
Confusion Matrix + ROC Curve
```

---

## Dataset

The model expects a CSV file with the following structure:

| Column | Description |
|--------|-------------|
| `Email No.` | Identifier column — dropped before training |
| `Prediction` | Target label (`1` = spam, `0` = not spam) |
| All other columns | Numeric word-frequency features |

> Update the file path in the notebook to point to your local copy of `emails.csv`.

---

## Model

**Algorithm:** Logistic Regression (`sklearn.linear_model.LogisticRegression`)

**Key configuration:**
- `class_weight='balanced'` — adjusts for class imbalance between spam and non-spam emails
- Features are standardised using `StandardScaler` before training

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision | Of emails flagged as spam, how many actually were |
| Recall | Of all actual spam emails, how many were caught |
| F1-Score | Harmonic mean of precision and recall |
| AUC-ROC | Area under the ROC curve; measures discriminative ability |

Both a **confusion matrix heatmap** and a **ROC curve** are plotted as output.

---

## Requirements

```bash
pip install pandas scikit-learn seaborn matplotlib
```

---

## Usage

1. Place your `emails.csv` dataset in an accessible path
2. Update the file path in the notebook:
   ```python
   data = pd.read_csv("path/to/your/emails.csv")
   ```
3. Run all cells top to bottom

---

## File

| File | Description |
|------|-------------|
| `email_spam_detection.ipynb` | Main Jupyter notebook |
