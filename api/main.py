from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import wandb
from pathlib import Path

app = FastAPI(title="Churn Prediction API")

# -----------------------------
# Load model from W&B Registry
# -----------------------------
WANDB_PROJECT = "churn-mlops"
WANDB_ENTITY = "tambedou89mariama-baamtu"

run = wandb.init(
    project=WANDB_PROJECT,
    entity=WANDB_ENTITY,
    job_type="inference"
)

artifact = run.use_artifact(
    "churn_model:production",
    type="model"
)

artifact_dir = artifact.download()

model = joblib.load(Path(artifact_dir) / "model.pkl")

# -----------------------------
# Input schema
# -----------------------------
class Customer(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: Customer):
    df = pd.DataFrame([data.dict()])
    proba = model.predict_proba(df)[0][1]
    return {"churn_probability": round(float(proba), 4)}
