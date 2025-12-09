# EV Charging MLOps Platform

End-to-end project that:

- pulls **real EV charging station data** from the Open Charge Map API (Germany),
- stores it locally as Parquet,
- trains a regression model to predict `sessions_per_day` (proxy utilization),
- serves predictions via a FastAPI API (with logging + metadata),
- logs all predictions for monitoring,
- provides a small Streamlit UI for demo,
- runs tests in CI via GitHub Actions,
- ships with a Docker image for the API.

---

## Tech Stack

- **Language**: Python 3.11
- **Data & ML**: pandas, NumPy, scikit-learn
- **API**: FastAPI, Uvicorn
- **UI**: Streamlit
- **Storage**: Parquet (pyarrow)
- **HTTP / API client**: requests, httpx (for tests)
- **CI**: GitHub Actions
- **Container**: Docker

---

## Data Source

**Open Charge Map API** – public EV charging locations.

- Country: `DE` (Germany)
- Endpoint: `https://api.openchargemap.io/v3/poi/`
- Raw response cached in: `data/raw/openchargemap_de.json`
- Processed training table: `data/processed/stations.parquet`

You need an API key from [openchargemap.org](https://openchargemap.org/):

1. Sign up → My Profile → My Apps → register app → copy API key.
2. Set it as an environment variable:

**PowerShell (persist for your user):**

```powershell
[System.Environment]::SetEnvironmentVariable(
  "OPENCHARGEMAP_API_KEY",
  "your_key_here",
  "User"
)
Then restart PowerShell.

Configuration

Basic ingest configuration is centralized in ev_charging_mlops_platform/settings.py.

Environment variables (optional overrides):

INGEST_COUNTRYCODE (default: DE)

INGEST_MAXRESULTS (default: 2000)

INGEST_OPENDATA (default: true)

Example override (PowerShell):

$env:INGEST_MAXRESULTS = "500"

Architecture:

          +----------------------+
          |  Open Charge Map API |
          +----------+-----------+
                     |
                     | HTTP (JSON)
                     v
          +------------------------------+
          |  Ingest                      |
          |  ev_charging_mlops_platform  |
          |      .data_ingest           |
          +------------------------------+
             |                    |
             | cache raw JSON     | flatten & label
             v                    v
  data/raw/openchargemap_de.json   data/processed/stations.parquet
                                           |
                                           | features + target
                                           v
                                +-----------------------------+
                                |  Training                   |
                                |  ev_charging_mlops_platform|
                                |      .train_model          |
                                +-----------------------------+
                                           |
                                           | save model + metadata
                                           v
                                models/model.joblib
                                models/metadata.json
                                           |
                                           | load on startup
                                           v
                                +-----------------------------+
                                |  FastAPI Service (api.main) |
                                |  - GET /health              |
                                |  - GET /metadata            |
                                |  - POST /predict            |
                                +-----------------------------+
                                           |
                                           | log predictions
                                           v
                             data/logs/predictions.parquet
                                           |
                                           v
                                Notebooks / monitoring


Setup:

git clone https://github.com/herrdevarsh/ev-charging-mlops-platform.git
cd ev-charging-mlops-platform

python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1

pip install -e .

Make sure OPENCHARGEMAP_API_KEY is set before running ingest.

Pipeline
1. Ingest data
python -m ev_charging_mlops_platform.data_ingest --refresh


Fetches from Open Charge Map (DE)

Caches raw JSON

Writes data/processed/stations.parquet

2. Train model
python -m ev_charging_mlops_platform.train_model


Trains a RandomForestRegressor on sessions_per_day (proxy label)

Saves:

models/model.joblib

models/metadata.json (includes MAE + training timestamp)

3. Run tests
pytest


Checks:

ingest + feature pipeline

API /health and /predict (after training)

API Service (FastAPI)

Start the API:

uvicorn api.main:app --reload


Endpoints:

GET /health
→ {"status": "ok", "model_loaded": true}

GET /metadata
→ model type, MAE, training timestamp.

POST /predict
Request body:

{
  "region": "Berlin",
  "city_type": "urban",
  "charger_type": "DC",
  "power_kW": 150,
  "num_connectors": 4
}


Response:

{
  "predicted_sessions_per_day": 23.4
}


Interactive docs:
http://127.0.0.1:8000/docs

All successful predictions are appended to data/logs/predictions.parquet.

Monitoring

Basic monitoring script: notebooks/monitoring_basics.py

Run:

python notebooks/monitoring_basics.py


Outputs:

training vs logged row counts,

distribution of training targets vs logged predictions,

simple feature distribution comparisons (e.g. region, num_connectors).

Streamlit Demo UI

Simple local UI for manual exploration.

streamlit run app_streamlit.py


Form inputs: region, city_type, charger_type, power, connectors

Shows sessions_per_day estimate

Docker

Build the API image:

docker build -f docker/Dockerfile.api -t ev-charging-api .


Run it (assuming model + metadata are already in models/):

docker run -p 8000:8000 \
  -e OPENCHARGEMAP_API_KEY=$OPENCHARGEMAP_API_KEY \
  ev-charging-api


Then:

http://127.0.0.1:8000/health

http://127.0.0.1:8000/metadata

http://127.0.0.1:8000/docs

CI (GitHub Actions)

Workflow: .github/workflows/ci.yml

On every push/PR to main:

sets up Python 3.11

installs project (pip install -e .)

runs pytest

The workflow uses the secret OPENCHARGEMAP_API_KEY defined in repo settings.

Roadmap / Possible Extensions:

Replace proxy label with real utilization/session data (if available).
Add real feature drift & performance monitoring (scheduled job).
Add model registry / versioning (e.g. MLflow).
Add periodic retraining pipeline (e.g. Prefect / Airflow).
Deploy API + Streamlit together via docker-compose.
