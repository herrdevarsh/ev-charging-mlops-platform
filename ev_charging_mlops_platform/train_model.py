import json
from datetime import datetime, timezone

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from .config import MODEL_DIR
from .feature_engineering import get_train_test


def train():
    X_train, X_test, y_train, y_test = get_train_test()

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print(f"MAE on test set: {mae:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "model_type": "RandomForestRegressor",
        "n_estimators": 200,
        "mae": float(mae),
        "feature_columns": list(X_train.columns),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    meta_path = MODEL_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    train()
