"""Konfiguration för feature-bygget."""
from __future__ import annotations
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FI_PATH = (
    ROOT
    / "data"
    / "processed"
    / "fi"
    / "aggregate"
    / "reconstructed.jsonl"
)
PRICE_DIR = (
    ROOT
    / "data"
    / "raw"
    / "prices"
)
OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "analysis"
)
# Feature-datasetet lagras som flera JSONL-filer.
# Ingen enskild fil får överstiga GitHub/Git-begränsningen.
FEATURE_GLOB = "features_*.jsonl"
METADATA_PATH = (
    OUTPUT_DIR
    / "features_metadata.json"
)
RETURN_HORIZONS = (
    1,
    3,
    5,
    10,
    20,
    60,
)
FI_REQUIRED_COLUMNS = {
    "snapshot_date",
    "issuer",
    "isin",
    "short_interest_pct",
    "active_holders",
    "max_individual_position_pct",
    "max_position_share_pct",
}
PRICE_REQUIRED_COLUMNS = {
    "date",
    "isin",
    "issuer",
    "yahoo_symbol",
    "close",
}
THRESHOLDS = (
    1.0,
    2.0,
    3.0,
    5.0,
)
