from fastapi.testclient import TestClient
import pytest

from ev_charging_mlops_platform.data_ingest import run_ingest
from ev_charging_mlops_platform.train_model import train
from api.main import app


@pytest.fixture(scope="session")
def client():
    # Prepare data + model once for all API tests
    run_ingest()
    train()

    # Create TestClient AFTER model exists so startup can load it
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict(client):
    payload = {
        "region": "Berlin",
        "city_type": "urban",
        "charger_type": "DC",
        "power_kW": 150,
        "num_connectors": 4,
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_sessions_per_day" in data
    assert data["predicted_sessions_per_day"] >= 0
