"""Signaler för Blankdiss research-matris."""
from __future__ import annotations
import numpy as np
import pandas as pd
SIGNAL_COLUMNS = {
    "short_interest_level": "short_interest_pct",
    "short_interest_change": "short_interest_delta_pp",
    "short_interest_acceleration": "short_interest_acceleration_pp",
    "price_momentum_5d": "price_return_5d",
    "price_momentum_20d": "price_return_20d",
    "price_momentum_60d": "price_return_60d",
    "price_volatility_20d": "price_volatility_20d",
    "distance_from_20d_high": "price_distance_from_20d_high",
    "distance_from_60d_high": "price_distance_from_60d_high",
}
def _require_column(
    frame: pd.DataFrame,
    column: str,
) -> None:
    if column not in frame.columns:
        raise ValueError(
            f"Saknar feature-kolumn '{column}'."
        )
def build_signal(
    frame: pd.DataFrame,
    signal_name: str,
) -> pd.Series:
    """
    Returnerar en numerisk signal.
    Signalerna bygger endast på information som finns på
    snapshot-datumet. Inga framtida returns används.
    """
    if signal_name not in SIGNAL_COLUMNS:
        raise ValueError(
            f"Okänd signal: {signal_name}"
        )
    column = SIGNAL_COLUMNS[signal_name]
    _require_column(
        frame,
        column,
    )
    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )
def tail_mask(
    frame: pd.DataFrame,
    signal: pd.Series,
    fraction: float,
    direction: str = "upper",
) -> pd.Series:
    """
    Väljer tvärsnittets övre eller undre tail per snapshot_date.
    Exempel:
        fraction=0.05, direction="upper"
        -> högsta 5 % varje snapshot-datum.
        fraction=0.05, direction="lower"
        -> lägsta 5 % varje snapshot-datum.
    Detta gör att ett experiment inte domineras av perioder
    med generellt högre/lägre signalnivåer.
    """
    if not 0 < fraction <= 1:
        raise ValueError(
            f"Ogiltig tail-fraktion: {fraction}"
        )
    if direction not in {"upper", "lower"}:
        raise ValueError(
            f"Ogiltig tail-riktning: {direction}"
        )
    working = pd.DataFrame(
        {
            "snapshot_date": frame["snapshot_date"],
            "signal": signal,
        },
        index=frame.index,
    )
    valid = working["signal"].notna()
    rank = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )
    rank.loc[valid] = (
        working.loc[valid]
        .groupby("snapshot_date")["signal"]
        .rank(
            pct=True,
            method="average",
        )
    )
    if direction == "upper":
        return rank >= (1.0 - fraction)
    return rank <= fraction
def signal_direction(
    signal_name: str,
) -> str:
    """
    Standardriktning för tail-test.
    För avstånd till high betyder högre värde normalt närmare
    high, medan lägre värde betyder större drawdown.
    """
    if signal_name in {
        "short_interest_level",
        "short_interest_change",
        "short_interest_acceleration",
        "price_momentum_5d",
        "price_momentum_20d",
        "price_momentum_60d",
        "price_volatility_20d",
    }:
        return "upper"
    if signal_name in {
        "distance_from_20d_high",
        "distance_from_60d_high",
    }:
        return "upper"
    raise ValueError(
        f"Saknar standardriktning för {signal_name}"
    )
def all_signal_names() -> list[str]:
    return list(SIGNAL_COLUMNS)
