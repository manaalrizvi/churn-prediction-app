"""
train_model.py - full training pipeline.
load -> clean -> encode -> feature selection -> split -> scale -> SMOTE -> train -> save

Run:  python train_model.py                      (uses DATA_PATH below)
      python train_model.py path/to/your.csv     (or pass the path)
Produces: churn_model.pkl, scaler.pkl, model_columns.pkl
"""
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# 1. Load
df = pd.read_csv(DATA_PATH)

# 2. Clean
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df = df.drop("customerID", axis=1)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
# "No internet service" / "No phone service" just repeat information already in
# InternetService / PhoneService, so collapse them into plain "No".
df = df.replace({"No internet service": "No", "No phone service": "No"})

# 3. Encode
df = pd.get_dummies(df, drop_first=True)

# 4. Feature selection / reduction
drop_cols = ["TotalCharges",                          # redundant with tenure (corr 0.83)
             "PhoneService_Yes",                      # weakest on correlation + RF importance
             "PaymentMethod_Mailed check",
             "PaymentMethod_Credit card (automatic)"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

X = df.drop("Churn", axis=1).astype(float)
y = df["Churn"]
joblib.dump(list(X.columns), "model_columns.pkl")   # API must reproduce this exact order

# 5. Split (before scaling / SMOTE, so nothing leaks into the test set)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Scale numeric columns (fit on train only)
num_cols = ["tenure", "MonthlyCharges"]
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
joblib.dump(scaler, "scaler.pkl")

# 7. Balance classes (train only)
X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)

# 8. Train + save
model = LogisticRegression(max_iter=1000)
model.fit(X_train_bal, y_train_bal)
joblib.dump(model, "churn_model.pkl")

print(classification_report(y_test, model.predict(X_test)))
print("Saved: churn_model.pkl, scaler.pkl, model_columns.pkl")
