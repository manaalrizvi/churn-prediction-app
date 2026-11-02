"""
app.py - FastAPI backend.   Run: uvicorn app:app --reload --port 8000
Interactive docs: http://127.0.0.1:8000/docs
"""
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float = 0.0   # accepted but unused (dropped during feature selection)


def preprocess(payload: CustomerData) -> pd.DataFrame:
    raw = pd.DataFrame([payload.model_dump()])
    raw = raw.replace({"No internet service": "No", "No phone service": "No"})

    # NOTE: no drop_first here. With a single row, drop_first would delete every
    # category. Instead we encode fully and keep only the columns the model was
    # trained on; reindex fills any missing dummy column with 0.
    encoded = pd.get_dummies(raw)
    encoded = encoded.reindex(columns=model_columns, fill_value=0).astype(float)

    num_cols = ["tenure", "MonthlyCharges"]
    encoded[num_cols] = scaler.transform(encoded[num_cols])
    return encoded


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerData):
    X = preprocess(customer)
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])
    return {"churn_prediction": "Yes" if pred else "No",
            "churn_probability": round(prob, 4)}
