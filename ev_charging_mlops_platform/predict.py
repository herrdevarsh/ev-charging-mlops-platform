import json
from typing import Any, Dict

import joblib
import pandas as pd

from .config import MODEL_DIR


class ModelService:
    def __init__(self):
        model_path = MODEL_DIR / "model.joblib"
        meta_path = MODEL_DIR / "metadata.json"

        if not model_path.exists() or not meta_path.exists():
            raise RuntimeError(
                "Model or metadata not found. "
                "Run the training pipeline first (python -m src.train_model)."
            )

        self.model = joblib.load(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.feature_columns = meta["feature_columns"]

    def _prepare_features(self, payload: Dict[str, Any]) -> pd.DataFrame:
        df = pd.DataFrame([payload])

        # One-hot encode payload in the same way as training
        df = pd.get_dummies(df, drop_first=True)

        # Add any missing feature columns with zeros
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Ensure correct column order
        df = df[self.feature_columns]

        return df

    def predict(self, payload: Dict[str, Any]) -> float:
        X = self._prepare_features(payload)
        pred = self.model.predict(X)[0]
        return float(pred)
