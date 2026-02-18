from pathlib import Path
import joblib
import pandas as pd
import wandb
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")

# -------------------
# W&B settings
# -------------------
WANDB_PROJECT = "churn-mlops"
WANDB_MODEL_NAME = "churn_model"
WANDB_ALIAS = "production"

# -------------------
# Download model from W&B
# -------------------
run = wandb.init(project=WANDB_PROJECT, job_type="inference")

artifact = run.use_artifact(
    f"{WANDB_PROJECT}/{WANDB_MODEL_NAME}:{WANDB_ALIAS}",
    type="model"
)

artifact_dir = artifact.download()

model = joblib.load(Path(artifact_dir) / "model.pkl")

# -------------------
# API schema
# -------------------
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
    proba = model.predict_proba(df)[0][1]
    return {"churn_probability": round(float(proba), 4)}
