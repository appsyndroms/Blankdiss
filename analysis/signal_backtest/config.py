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
# 5 handelsdagars target:
# ny portfölj var femte observationsdag för att undvika
# överlappande 5-dagarsperioder.
ECONOMIC_REBALANCE_DAYS = 5
# Maximal vikt för ett enskilt värdepapper.
#
# Om urvalet innehåller färre värdepapper än vad som krävs
# för att fylla portföljen med denna maxvikt blir resterande
# kapital cash.
#
# Exempel:
#   1 aktie  -> 5 % investerat, 95 % cash
#   5 aktier -> 25 % investerat, 75 % cash
#   20 aktier -> 100 % investerat
ECONOMIC_MAX_POSITION_WEIGHT = 0.05
# Testa flera explicita transaktionskostnadsantaganden.
# 5 / 10 / 20 bps per rebalance.
ECONOMIC_TRANSACTION_COST_BPS = (
    5.0,
    10.0,
    20.0,
)
# Bakåtkompatibilitet för eventuell kod som fortfarande importerar
# den gamla singulara parametern.
TRANSACTION_COST_BPS = 10.0
