# Customer Churn Prediction — End-to-End ML Project

## 1. Problem Statement
Predict whether a telecom customer will churn (leave the company) based on their
account information, services subscribed, and billing details. This is a binary
classification problem (`Churn`: Yes/No), and the business goal is to identify
at-risk customers *before* they leave so retention teams can act.

## 2. Dataset
- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** ~7,043 rows, 21 columns
- **Target variable:** `Churn` (Yes/No → encoded as 1/0)
- **Class balance:** ~73% No Churn, ~27% Churn (imbalanced)

## 3. Data Cleaning
| Issue | Fix |
|---|---|
| `TotalCharges` stored as string with 11 blank entries | Converted to numeric with `pd.to_numeric(..., errors='coerce')`, filled NaN with 0 (these were new customers with `tenure = 0`) |
| `customerID` column | Dropped — unique identifier, no predictive value |
| Categorical columns | One-hot encoded with `pd.get_dummies(drop_first=True)` |
| Target column | Mapped `Yes/No` → `1/0` |

## 4. Exploratory Data Analysis (EDA)
- Univariate analysis of numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`)
- Correlation heatmap across all encoded features
- Class balance check on `Churn` (73:27 split, confirmed imbalance)
- Key insight: `tenure` (-0.35) and `Contract_Two year` (-0.30) showed the
  strongest negative correlation with churn — longer-tenured, longer-contract
  customers are less likely to churn.

## 5. Feature Selection & Data Reduction
**Method 1 — Correlation with target:** Ranked all features by absolute
correlation with `Churn`.

**Method 2 — Multicollinearity check:** `tenure` and `TotalCharges` were
highly correlated (0.83). Since `TotalCharges ≈ tenure × MonthlyCharges`, it
carried redundant information — `TotalCharges` was dropped, `tenure` retained
(stronger correlation with target: -0.35 vs -0.20).

**Method 3 — Random Forest feature importance (embedded method):** Confirmed
`tenure` (0.247) and `MonthlyCharges` (0.237) as the dominant predictors.
Bottom 3 weakest features by both correlation and importance
(`PhoneService_Yes`, `PaymentMethod_Mailed check`,
`PaymentMethod_Credit card (automatic)`) were dropped.

**Result:** Reduced from 21 raw columns to a leaner, non-redundant feature set
without materially losing signal.

## 6. Handling Class Imbalance
Applied **SMOTE** (Synthetic Minority Oversampling) — but only on the training
set, *after* the train-test split, to avoid data leakage into evaluation.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)
```

## 7. Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Random Forest | 0.776 | 0.576 | 0.588 | 0.582 |
| Decision Tree | 0.715 | 0.467 | 0.529 | 0.496 |
| **Logistic Regression** | 0.738 | 0.504 | **0.765** | **0.608** |

**Chosen metric:** Recall / F1-Score, not Accuracy — because a missed churner
(False Negative) is far more costly to the business than a false alarm
(False Positive). Accuracy is misleading on imbalanced data (a model that
always predicts "No Churn" would score ~73% accuracy while being useless).

## 8. Final Model: Logistic Regression
**Confusion Matrix (test set, n=1409):**

| | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** | 754 (TN) | 281 (FP) |
| **Actual: Churn** | 88 (FN) | 286 (TP) |

- **Recall = 76.5%** — the model correctly flags 3 out of 4 customers who
  will actually churn.
- Trade-off: more false alarms (281) in exchange for catching more real
  churners (only 88 missed) — an intentional choice given the business cost
  of losing a customer silently.

## 9. Tech Stack
- **Data processing:** pandas, numpy
- **Visualization:** seaborn, matplotlib
- **Modeling:** scikit-learn (Logistic Regression, Random Forest, Decision Tree, KNN)
- **Imbalance handling:** imbalanced-learn (SMOTE)
- **Serving:** FastAPI (REST API)
- **UI:** Streamlit (interactive frontend)

## 10. Project Structure
```
churn_project/
├── PROJECT_REPORT.md      # this file
├── churn_prediction_complete.ipynb  # full notebook: cleaning → EDA → selection → models → evaluation
├── train_model.py         # same pipeline as a script (final model → save .pkl)
├── app.py                 # FastAPI backend serving predictions
├── streamlit_app.py       # Streamlit frontend UI
├── churn_model.pkl        # saved trained model
├── scaler.pkl             # saved StandardScaler
├── model_columns.pkl      # saved column order (for consistent encoding)
└── requirements.txt
```

## 11. How to Run
```bash
# 1. Train and save the model (run once)
python train_model.py

# 2. Start the FastAPI backend
uvicorn app:app --reload --port 8000

# 3. Start the Streamlit frontend (in a new terminal)
streamlit run streamlit_app.py
```


