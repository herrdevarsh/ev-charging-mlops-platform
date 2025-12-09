import os
from dataclasses import dataclass, field


@dataclass
class IngestConfig:
    # Default values, can be overridden via environment variables
    countrycode: str = os.getenv("INGEST_COUNTRYCODE", "DE")
    maxresults: int = int(os.getenv("INGEST_MAXRESULTS", "2000"))
    opendata: bool = os.getenv("INGEST_OPENDATA", "true").lower() == "true"


@dataclass
class Settings:
    ingest: IngestConfig = field(default_factory=IngestConfig)


settings = Settings()
