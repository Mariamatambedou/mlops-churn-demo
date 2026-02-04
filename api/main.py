from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")

MODEL_PATH = Path("models/model.pkl")
SCALER_PATH = Path("data/processed/scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)



class Customer(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: Customer):

    df = pd.DataFrame([data.dict()])

    # scaling comme en training
    X_scaled = scaler.transform(df)

    proba = model.predict_proba(X_scaled)[0][1]

    return {
        "churn_probability": round(float(proba), 4)
    }
