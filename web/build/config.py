"""
Gemensamma sökvägar och konfiguration för webbbygget.
"""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]

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

TEMPLATE_DIR = (
    ROOT
    / "web"
    / "templates"
)

STATIC_DIR = (
    ROOT
    / "web"
    / "static"
)

OUTPUT_DIR = (
    ROOT
    / "pages"
)

STOCKHOLM = ZoneInfo(
    "Europe/Stockholm"
)
