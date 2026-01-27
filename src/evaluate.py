import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
import json

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

from joblib import load
model = load(MODEL_DIR / "model.pkl")

preds = model.predict_proba(X_test)[:, 1]

roc = roc_auc_score(y_test, preds)

metrics = {"roc_auc": roc}

Path("metrics").mkdir(exist_ok=True)
with open("metrics/eval.json", "w") as f:
    json.dump(metrics, f)

print("Evaluation:", metrics)
