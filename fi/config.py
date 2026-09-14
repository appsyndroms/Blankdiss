"""Konfiguration för Finansinspektionens blankningsregister."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "fi"
    / "aggregate"
)

SOURCE_DIR = RAW_DIR / "source"

SNAPSHOT_DIR = RAW_DIR / "snapshots"

MANIFEST_PATH = RAW_DIR / "manifest.json"


FI_URL = (
    "https://www.fi.se/sv/vara-register/"
    "blankningsregistret/"
)

FI_AGGREGATE_URL = (
    "https://www.fi.se"
    "/BlankningsRegister/"
    "GetBlankningsregisterAggregat"
)

FI_AGGREGATE_TIMEOUT = 60


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(compatible; Blankdiss/1.0; "
        "+https://github.com/appsyndroms/Blankdiss)"
    ),
    "Accept": (
        "application/vnd.oasis.opendocument.spreadsheet,"
        "application/vnd.oasis.opendocument.spreadsheet-template,"
        "application/octet-stream"
    ),
    "Accept-Language": "sv-SE,sv;q=0.9,en;q=0.8",
}
