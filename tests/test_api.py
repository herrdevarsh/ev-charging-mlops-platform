from fastapi.testclient import TestClient

from ev_charging_mlops_platform.data_ingest import run_ingest
from ev_charging_mlops_platform.train_model import train
from api.main import app


def setup_module():
    # Prepare data + model for API tests
    run_ingest()
    train()


client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict():
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
