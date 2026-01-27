import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

RAW_PATH = Path("data/raw/churn.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_PATH)

df = df.dropna()

# ---- Target
target = "Churn"

y = df[target].map({"Yes": 1, "No": 0})

# ---- Features
X = df.drop(columns=[target])
X = X.select_dtypes(include="number")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, PROCESSED_DIR / "scaler.pkl")

pd.DataFrame(X_train).to_csv(PROCESSED_DIR / "X_train.csv", index=False)
pd.DataFrame(X_test).to_csv(PROCESSED_DIR / "X_test.csv", index=False)
y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

print("Preprocessing done.")
