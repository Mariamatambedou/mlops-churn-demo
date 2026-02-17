import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib
import wandb
import sklearn

print("Training sklearn:", sklearn.__version__)

wandb.init(project="churn-mlops")

DATA_DIR = Path("data/processed")

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

X_train.columns = ["tenure", "monthly_charges", "total_charges"]
X_test.columns = X_train.columns

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, preds)

wandb.log({"roc_auc": roc})

# --- SAVE MODEL ---
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

model_path = MODEL_DIR / "model.pkl"
joblib.dump(model, model_path)

# --- LOG MODEL AS ARTIFACT ---
artifact = wandb.Artifact(
    name="churn_model",
    type="model",
    description="Logistic Regression churn model"
)

artifact.add_file(str(model_path))

wandb.log_artifact(artifact)

wandb.finish()

print("ROC AUC:", roc)
