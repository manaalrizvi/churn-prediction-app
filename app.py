"""
app.py — FastAPI backend serving the churn model.

Run: uvicorn app:app --reload --port 8000
Docs auto-generated at: http://127.0.0.1:8000/docs
"""
from fastapi.middleware.cors import CORSMiddleware


import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Customer Churn Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load trained artifacts once at startup (not on every request — faster)
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")


class CustomerData(BaseModel):
    """Raw, human-readable customer fields — same as the original CSV columns
    (minus customerID and Churn). FastAPI validates types automatically."""
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
    TotalCharges: float



def preprocess(payload: CustomerData) -> pd.DataFrame:
    """Turn raw input into the exact encoded structure the model expects."""
    raw = pd.DataFrame([payload.dict()])

    # Same one-hot encoding as training
    encoded = pd.get_dummies(raw, drop_first=True)

    # Add any columns missing vs. training set (e.g. a category not present
    # in this single row), fill with 0, then force the exact column order.
    for col in model_columns:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[model_columns]

    # Scale the same numeric columns the same way as training
    numeric_cols = ["tenure", "MonthlyCharges"]
    encoded[numeric_cols] = scaler.transform(encoded[numeric_cols])

    return encoded


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Churn prediction API is running"}


@app.post("/predict")
def predict(customer: CustomerData):
    X = preprocess(customer)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 4),
    }
