from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

OUTPUT_DIR = (
    ROOT
    / "pages"
)

STATIC_DIR = (
    ROOT
    / "web"
    / "static"
)

TEMPLATE_DIR = (
    ROOT
    / "web"
    / "templates"
)

ANALYSIS_DIR = (
    ROOT
    / "data"
    / "analysis"
)

EVENT_DIR = (
    ROOT
    / "data"
    / "events"
)

FEATURE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "analysis"
)

ML_DIR = (
    ROOT
    / "data"
    / "processed"
    / "ml"
)

EVALUATION_DIR = (
    ML_DIR
    / "research"
    / "evaluation"
)

STOCKHOLM = ZoneInfo(
    "Europe/Stockholm"
)
