import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

RAW_PATH = Path("data/churn.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load
# -------------------------
df = pd.read_csv(RAW_PATH)

# Nettoyage colonnes
df.columns = (
    df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
)

print("COLUMNS:", df.columns.tolist())

# Harmonisation noms
df = df.rename(columns={
    "monthlycharges": "monthly_charges",
    "totalcharges": "total_charges",
})

# Conversion numérique
df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

df = df.dropna()


# ---- Target
target = "churn"

y = df[target].map({"Yes": 1, "No": 0})

# ---- Features
features = ["tenure", "monthly_charges", "total_charges"]

X = df[features]

# -------------------------
# Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Scaling
# -------------------------
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=features
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=features
)

# -------------------------
# Save
# -------------------------
joblib.dump(scaler, PROCESSED_DIR / "scaler.pkl")

X_train_scaled.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
X_test_scaled.to_csv(PROCESSED_DIR / "X_test.csv", index=False)

y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

print("Preprocessing done.")
