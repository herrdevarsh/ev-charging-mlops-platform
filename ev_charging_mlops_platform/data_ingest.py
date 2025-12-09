import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests

from .config import RAW_DIR, PROCESSED_DIR
from .settings import settings

BASE_URL = "https://api.openchargemap.io/v3/poi/"
RAW_JSON_FILE = RAW_DIR / "openchargemap_de.json"
PROCESSED_FILE = PROCESSED_DIR / "stations.parquet"


def fetch_openchargemap(
    countrycode: str,
    maxresults: int,
    opendata: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fetch real EV charging locations from Open Charge Map.

    Reads API key from env var: OPENCHARGEMAP_API_KEY
    """
    params: Dict[str, Any] = {
        "output": "json",
        "countrycode": countrycode,
        "maxresults": maxresults,
        "compact": "true",
        "verbose": "false",
    }
    if opendata:
        params["opendata"] = "true"

    api_key = os.getenv("OPENCHARGEMAP_API_KEY")
    if api_key:
        params["key"] = api_key

    resp = requests.get(BASE_URL, params=params, timeout=30)
    try:
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        snippet = resp.text[:500] if resp.text else ""
        raise RuntimeError(
            f"OpenChargeMap API call failed "
            f"(status={resp.status_code}, body_snippet={snippet})"
        ) from exc

    data = resp.json()
    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError(
            f"OpenChargeMap API returned no POIs. Params used: {params}"
        )

    print(f"Fetched {len(data)} POIs from OpenChargeMap.")
    return data


def flatten_pois(pois: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten nested POI JSON into a tabular DataFrame and build a proxy target.

    All rows/features come from the real API.
    sessions_per_day is derived as a proxy label from power/connectors/status.
    """
    records: List[Dict[str, Any]] = []

    for poi in pois:
        addr = poi.get("AddressInfo") or {}
        conns = poi.get("Connections") or []

        num_connectors = len(conns)
        max_power_kw = 0.0
        if num_connectors > 0:
            powers = [
                c.get("PowerKW") or 0.0
                for c in conns
                if isinstance(c, dict)
            ]
            if powers:
                max_power_kw = float(max(powers))

        country_iso = None
        country_obj = addr.get("Country")
        if isinstance(country_obj, dict):
            country_iso = country_obj.get("ISOCode")

        record = {
            "station_id": poi.get("ID"),
            "title": addr.get("Title"),
            "country": country_iso,
            "region": addr.get("StateOrProvince"),
            "town": addr.get("Town"),
            "latitude": addr.get("Latitude"),
            "longitude": addr.get("Longitude"),
            "num_connectors": num_connectors,
            "max_power_kw": max_power_kw,
            "usage_type_id": poi.get("UsageTypeID"),
            "status_type_id": poi.get("StatusTypeID"),
        }
        records.append(record)

    df = pd.DataFrame.from_records(records)

    # Drop obvious garbage / duplicates
    df = df.dropna(subset=["station_id"]).drop_duplicates(subset=["station_id"])

    df["num_connectors"] = df["num_connectors"].fillna(0).astype(int)
    df["max_power_kw"] = df["max_power_kw"].fillna(0.0).astype(float)

    # ---- proxy target: sessions_per_day ----
    rng = np.random.default_rng(42)

    base = (df["max_power_kw"] / 10.0) + df["num_connectors"] * 0.8

    # Treat StatusTypeID == 50 as “operational” ⇒ higher utilization
    status_factor = np.where(df["status_type_id"] == 50, 1.2, 0.7)

    noise = rng.normal(0, 1.5, size=len(df))

    df["sessions_per_day"] = (base * status_factor + noise).clip(lower=0).round(1)

    return df


def run_ingest(force_refresh: bool = False) -> None:
    """
    Real-world ingest:
    - If processed parquet exists and not forcing refresh, reuse it.
    - Else:
      * If raw JSON exists and not forcing refresh → load it
      * Otherwise → call API and cache the JSON
    - Then flatten and save to Parquet for training.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if PROCESSED_FILE.exists() and not force_refresh:
        print(f"Using existing processed data at {PROCESSED_FILE}")
        return

    if RAW_JSON_FILE.exists() and not force_refresh:
        print(f"Loading cached raw JSON from {RAW_JSON_FILE}")
        with RAW_JSON_FILE.open("r", encoding="utf-8") as f:
            pois = json.load(f)
    else:
        print("Calling Open Charge Map API for fresh data...")
        pois = fetch_openchargemap(
            countrycode=settings.ingest.countrycode,
            maxresults=settings.ingest.maxresults,
            opendata=settings.ingest.opendata,
        )
        with RAW_JSON_FILE.open("w", encoding="utf-8") as f:
            json.dump(pois, f, indent=2)
        print(f"Cached raw JSON to {RAW_JSON_FILE}")

    df = flatten_pois(pois)
    df.to_parquet(PROCESSED_FILE, index=False)
    print(
        f"Processed data saved to {PROCESSED_FILE} with "
        f"{len(df)} stations and columns: {list(df.columns)}"
    )


if __name__ == "__main__":
    import sys

    # Allow: python -m ev_charging_mlops_platform.data_ingest --refresh
    force = "--refresh" in sys.argv
    run_ingest(force_refresh=force)
