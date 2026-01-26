import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import wandb

wandb.init(project="churn-mlops-beginner")

df = pd.read_csv("data/churn.csv")

df = df.dropna()

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})

X_train, X_test, y_train, y_test = train_test_split(
    X.select_dtypes(include="number"),
    y,
    test_size=0.2,
    random_state=42,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, preds)

wandb.log({"roc_auc": roc})

print("ROC AUC:", roc)
