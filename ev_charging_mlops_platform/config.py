from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

# Make sure folders exist
for p in (RAW_DIR, PROCESSED_DIR, MODEL_DIR):
    p.mkdir(parents=True, exist_ok=True)
