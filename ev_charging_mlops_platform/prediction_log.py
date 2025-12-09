from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

from .config import LOG_DIR

LOG_FILE = LOG_DIR / "predictions.parquet"


def log_prediction(payload: Dict[str, Any], prediction: float) -> None:
    """
    Append a prediction record to predictions.parquet.

    Columns:
    - timestamp_utc
    - prediction
    - all input feature fields (flattened from payload)
    """
    record: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prediction": float(prediction),
    }
    record.update(payload)

    new_df = pd.DataFrame([record])

    if LOG_FILE.exists():
        # small-scale: read, append, overwrite
        existing = pd.read_parquet(LOG_FILE)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_parquet(LOG_FILE, index=False)
    else:
        new_df.to_parquet(LOG_FILE, index=False)
