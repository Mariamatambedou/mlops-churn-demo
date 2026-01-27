import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib
import wandb

wandb.init(project="churn-mlops")

DATA_DIR = Path("data/processed")

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, preds)

wandb.log({"roc_auc": roc})

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(exist_ok=True)

joblib.dump(model, MODEL_PATH / "model.pkl")

wandb.save(str(MODEL_PATH / "model.pkl"))

print("ROC AUC:", roc)
