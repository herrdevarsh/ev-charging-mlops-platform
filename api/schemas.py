from pydantic import BaseModel, Field


class StationFeatures(BaseModel):
    region: str = Field(..., example="Berlin")
    city_type: str = Field(..., example="urban")
    charger_type: str = Field(..., example="DC")
    power_kW: float = Field(..., ge=1, example=150)
    num_connectors: int = Field(..., ge=1, example=4)


class PredictionResponse(BaseModel):
    predicted_sessions_per_day: float


class ModelMetadata(BaseModel):
    model_type: str | None = None
    mae: float | None = None
    training_timestamp: str | None = None
