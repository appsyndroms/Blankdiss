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


# Ekonomiskt backtest
# Samma urvalsnivåer som det befintliga signal-backtestet.
ECONOMIC_BACKTEST_FRACTIONS = TOP_FRACTIONS

# 5 handelsdagars target: ny portfölj var femte observationsdag
# för att undvika överlappande 5-dagarsperioder i första versionen.
ECONOMIC_REBALANCE_DAYS = 5

# Antagen transaktionskostnad per rebalance, i basispunkter.
# Detta är en explicit modellparameter och inte en observerad kostnad.
TRANSACTION_COST_BPS = 10.0
