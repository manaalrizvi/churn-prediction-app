"""
train_model.py
Full training pipeline: load -> clean -> encode -> feature select ->
split -> scale -> balance -> train -> save artifacts.

Run once: python train_model.py
Produces: churn_model.pkl, scaler.pkl, model_columns.pkl
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# ---------- 1. Load ----------
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"  # change to your file path
df = pd.read_csv(DATA_PATH)

# ---------- 2. Clean ----------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)
df = df.drop("customerID", axis=1)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ---------- 3. Encode ----------
df = pd.get_dummies(df, drop_first=True)

# ---------- 4. Feature selection / reduction ----------
# TotalCharges dropped: highly correlated with tenure (0.83), tenure is the
# stronger predictor of the two, so it is kept and TotalCharges is dropped.
weak_features = [
    "TotalCharges",
    "PhoneService_Yes",
    "PaymentMethod_Mailed check",
    "PaymentMethod_Credit card (automatic)",
]
df = df.drop(columns=[c for c in weak_features if c in df.columns])

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Save the exact column order/names used for training —
# the API needs to reproduce this exact structure at prediction time.
joblib.dump(list(X.columns), "model_columns.pkl")

# ---------- 5. Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 6. Scale ----------
numeric_cols = ["tenure", "MonthlyCharges"]
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
joblib.dump(scaler, "scaler.pkl")

# ---------- 7. Balance (train only) ----------
X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)

# ---------- 8. Train ----------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_bal, y_train_bal)
joblib.dump(model, "churn_model.pkl")

# ---------- 9. Quick sanity check ----------
from sklearn.metrics import classification_report
print(classification_report(y_test, model.predict(X_test)))
print("\nArtifacts saved: churn_model.pkl, scaler.pkl, model_columns.pkl")
