from fastapi import FastAPI

from .schemas import StationFeatures, PredictionResponse
from ev_charging_mlops_platform.predict import ModelService


app = FastAPI(title="EV Charging Load Prediction API")

model_service: ModelService | None = None


@app.on_event("startup")
def load_model():
    global model_service
    model_service = ModelService()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: StationFeatures):
    if model_service is None:
        raise RuntimeError("Model not loaded")

    pred = model_service.predict(features.dict())
    return PredictionResponse(predicted_sessions_per_day=pred)

@app.get("/")
def root():
    return {"message": "EV Charging MLOps API", "docs": "/docs"}
