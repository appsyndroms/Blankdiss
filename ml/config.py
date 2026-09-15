"""Konfiguration för Blankdiss ML."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "analysis"
    / "features.jsonl"
)

ML_OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "ml"
)

RUNS_DIR = (
    ML_OUTPUT_DIR
    / "runs"
)

RESULTS_PATH = (
    ML_OUTPUT_DIR
    / "ml_results.jsonl"
)

LATEST_RESULT_PATH = (
    ML_OUTPUT_DIR
    / "latest_run.json"
)


@dataclass(frozen=True)
class TargetConfig:
    name: str
    return_column: str
    threshold: float = 0.0


TARGETS = (
    # Kontrolltargets: går aktien upp eller ner?
    TargetConfig(
        name="positive_5d",
        return_column="forward_return_5d",
        threshold=0.0,
    ),
    TargetConfig(
        name="positive_20d",
        return_column="forward_return_20d",
        threshold=0.0,
    ),
    TargetConfig(
        name="positive_60d",
        return_column="forward_return_60d",
        threshold=0.0,
    ),

    # Ekonomiskt mer intressanta positiva rörelser.
    TargetConfig(
        name="up_5pct_5d",
        return_column="forward_return_5d",
        threshold=0.05,
    ),
    TargetConfig(
        name="up_5pct_20d",
        return_column="forward_return_20d",
        threshold=0.05,
    ),
    TargetConfig(
        name="up_10pct_60d",
        return_column="forward_return_60d",
        threshold=0.10,
    ),

    # Relevanta för blankningscaset.
    TargetConfig(
        name="down_5pct_5d",
        return_column="forward_return_5d",
        threshold=-0.05,
    ),
    TargetConfig(
        name="down_5pct_20d",
        return_column="forward_return_20d",
        threshold=-0.05,
    ),
    TargetConfig(
        name="down_10pct_60d",
        return_column="forward_return_60d",
        threshold=-0.10,
    ),
)


@dataclass(frozen=True)
class WalkForwardWindow:
    train_end: str
    validation_end: str
    test_end: str


WALK_FORWARD_WINDOWS = (
    WalkForwardWindow(
        train_end="2023-12-31",
        validation_end="2024-12-31",
        test_end="2025-12-31",
    ),
    WalkForwardWindow(
        train_end="2024-12-31",
        validation_end="2025-12-31",
        test_end="2026-12-31",
    ),
)


RANDOM_STATE = 42

TEST_MIN_ROWS = 100

VALIDATION_MIN_ROWS = 100


# Kolumner som aldrig får användas som ML-features.
# Framför allt får framtida avkastning aldrig läcka in.
FEATURE_EXCLUDE_COLUMNS = {
    "snapshot_date",
    "issuer",
    "isin",
    "security_key",
    "yahoo_symbol",
    "price_mapping_source",
    "price_date",
    "close",
    "close_on_signal_date",
    "price_match_available",
    "days_from_fi_to_price",

    "forward_return_1d",
    "forward_return_5d",
    "forward_return_20d",
    "forward_return_60d",
}


# Prisfeatures som kan användas när vi vill mäta
# "FI + pris" istället för FI-only.
PRICE_FEATURE_COLUMNS = {
    "price_return_5d",
    "price_return_20d",
    "price_return_60d",
    "price_volatility_20d",
    "price_distance_from_20d_high",
    "price_distance_from_60d_high",
}


# De här kolumnerna används för FI-only-experimentet.
# Prisfeatures läggs uttryckligen bort.
FI_ONLY_EXCLUDE_COLUMNS = (
    FEATURE_EXCLUDE_COLUMNS
    | PRICE_FEATURE_COLUMNS
)
