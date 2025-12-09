import pandas as pd
from sklearn.model_selection import train_test_split

from .config import PROCESSED_DIR

TARGET_COL = "sessions_per_day"


def load_processed() -> pd.DataFrame:
    file = PROCESSED_DIR / "stations.parquet"
    if not file.exists():
        raise FileNotFoundError(f"Processed file not found: {file}. Run data_ingest first.")
    return pd.read_parquet(file)


def build_features(df: pd.DataFrame):
    df = df.copy()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in data.")

    y = df[TARGET_COL]
    # Drop ID-like columns that are not useful for prediction
    X = df.drop(columns=[TARGET_COL, "station_id"], errors="ignore")

    # One-hot encode categoricals
    X = pd.get_dummies(X, drop_first=True)

    return X, y


def get_train_test(test_size: float = 0.2, random_state: int = 42):
    df = load_processed()
    X, y = build_features(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
