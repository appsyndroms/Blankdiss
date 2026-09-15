"""Konfiguration för Blankdiss signal-backtest."""
from __future__ import annotations
from ml.config import PRICE_FEATURE_COLUMNS
TOP_FRACTIONS = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
)
BACKTEST_FEATURE_SETS = (
    (
        "fi_plus_price_volatility_20d",
        {"price_volatility_20d"},
    ),
    (
        "fi_plus_all_price",
        set(PRICE_FEATURE_COLUMNS),
    ),
)
BACKTEST_TARGETS = (
    "up_5pct_5d",
    "down_5pct_5d",
)
