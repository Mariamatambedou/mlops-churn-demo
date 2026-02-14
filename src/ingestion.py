import shutil
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_FILE = Path("data/churn.csv")
DEST_FILE = RAW_DIR / "churn.csv"

print("Copying raw data...")

shutil.copy(SOURCE_FILE, DEST_FILE)

print("Saved to", DEST_FILE)
