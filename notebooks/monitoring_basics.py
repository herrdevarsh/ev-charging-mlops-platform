from pathlib import Path

import pandas as pd


TRAIN_FILE = Path("data/processed/stations.parquet")
LOG_FILE = Path("data/logs/predictions.parquet")


def load_data():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training data not found at {TRAIN_FILE}. Run ingest + train first.")
    if not LOG_FILE.exists():
        raise FileNotFoundError(f"Prediction log not found at {LOG_FILE}. Call /predict a few times first.")

    train = pd.read_parquet(TRAIN_FILE)
    logs = pd.read_parquet(LOG_FILE)
    return train, logs


def summarize_basic(train: pd.DataFrame, logs: pd.DataFrame) -> None:
    print("=== SHAPES ===")
    print(f"Training rows:   {len(train)}")
    print(f"Prediction logs: {len(logs)}")
    print()

    print("=== TRAINING TARGET (sessions_per_day) ===")
    print(train["sessions_per_day"].describe())
    print()

    print("=== PREDICTED SESSIONS PER DAY ===")
    print(logs["prediction"].describe())
    print()

    print("=== EXAMPLE LOG ROWS ===")
    print(logs.tail(5))
    print()


def compare_feature(train: pd.DataFrame, logs: pd.DataFrame, col: str) -> None:
    print(f"=== FEATURE: {col} ===")

    if col not in train.columns and col not in logs.columns:
        print("  -> Column not present in either training or logs.\n")
        return

    if col in train.columns:
        print("  Training value counts (top 5):")
        print(train[col].value_counts().head(5))
        print()

    if col in logs.columns:
        print("  Logs value counts (top 5):")
        print(logs[col].value_counts().head(5))
        print()

    print()


def main():
    train, logs = load_data()

    summarize_basic(train, logs)

    # Compare a few key features between training vs live usage
    for col in ["region", "num_connectors", "max_power_kw", "status_type_id"]:
        if col in train.columns or col in logs.columns:
            compare_feature(train, logs, col)


if __name__ == "__main__":
    main()
