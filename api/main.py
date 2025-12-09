import json
import logging

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse

from ev_charging_mlops_platform.config import MODEL_DIR
from ev_charging_mlops_platform.predict import ModelService
from ev_charging_mlops_platform.prediction_log import log_prediction

from .schemas import StationFeatures, PredictionResponse, ModelMetadata

# basic logging setup
logger = logging.getLogger("ev_charging_api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

app = FastAPI(title="EV Charging Load Prediction API")

model_service: ModelService | None = None


@app.on_event("startup")
def load_model() -> None:
    global model_service
    try:
        model_service = ModelService()
        logger.info("Model loaded successfully.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load model on startup: %s", exc)
        model_service = None


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception(
        "Unhandled error for %s %s: %s",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_service is not None,
    }


@app.get("/metadata", response_model=ModelMetadata)
def get_metadata():
    meta_path = MODEL_DIR / "metadata.json"
    if not meta_path.exists():
        logger.error("Metadata file not found at %s", meta_path)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata not available. Train the model first.",
        )

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to read metadata file: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error reading model metadata.",
        )

    public = {
        "model_type": meta.get("model_type"),
        "mae": meta.get("mae"),
        "training_timestamp": meta.get("training_timestamp"),
    }
    return ModelMetadata(**public)


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: StationFeatures):
    if model_service is None:
        logger.error("Prediction requested but model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )

    payload = features.dict()
    try:
        pred = model_service.predict(payload)
        logger.info(
            "Prediction succeeded: payload=%s, prediction=%s",
            payload,
            pred,
        )

        # Log prediction, but don't break the API if logging fails
        try:
            log_prediction(payload, pred)
        except Exception as log_exc:  # noqa: BLE001
            logger.exception("Failed to log prediction: %s", log_exc)

        return PredictionResponse(predicted_sessions_per_day=pred)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error during prediction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during prediction.",
        )


@app.get("/")
def root():
    return {"message": "EV Charging MLOps API", "docs": "/docs"}
