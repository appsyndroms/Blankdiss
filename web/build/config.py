from __future__ import annotations
from pathlib import Path
from zoneinfo import ZoneInfo
PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "pages"
)
STATIC_DIR = (
    PROJECT_ROOT
    / "web"
    / "static"
)
TEMPLATE_DIR = (
    PROJECT_ROOT
    / "web"
    / "templates"
)
STOCKHOLM = ZoneInfo(
    "Europe/Stockholm"
)
