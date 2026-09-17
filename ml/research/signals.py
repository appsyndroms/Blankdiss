"""
Signalbyggnad för Blankdiss research matrix.

Den här modulen innehåller endast signaler.
Den kör inte ML och skriver inga resultatfiler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalDefinition:
    name: str
    description: str
    builder: Callable[[pd.DataFrame], pd.Series]


def _numeric(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
) -> tuple[pd.Series | None, str | None]:
    """Returnerar första numeriska kolumnen som finns."""
    for column in candidates:
        if column in frame.columns:
            return (
                pd.to_numeric(
                    frame[column],
                    errors="coerce",
                ),
                column,
            )

    return None, None


def _require(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    signal_name: str,
) -> pd.Series:
    values, column = _numeric(frame, candidates)

    if values is None:
        raise ValueError(
            f"Kan inte bygga signal '{signal_name}'. "
            f"Saknar någon av: {', '.join(candidates)}"
        )

    return values


def _rank_percentile(
    values: pd.Series,
) -> pd.Series:
    """Cross-sectional percentile per snapshot-dag."""
    return values.groupby(
        values.index
    ).rank(
        pct=True,
        method="average",
    )


def _cross_sectional_percentile(
    frame: pd.DataFrame,
    values: pd.Series,
) -> pd.Series:
    return values.groupby(
        frame["snapshot_date"]
    ).rank(
        pct=True,
        method="average",
    )


def build_short_interest_level(
    frame: pd.DataFrame,
) -> pd.Series:
    values = _require(
        frame,
        (
            "short_interest_pct",
            "short_interest",
            "short_position_pct",
            "aggregate_short_pct",
        ),
        "short_interest_level",
    )

    return _cross_sectional_percentile(
        frame,
        values,
    )


def build_short_interest_change(
    frame: pd.DataFrame,
) -> pd.Series:
    values = _require(
        frame,
        (
            "short_interest_pct_change",
            "short_interest_change",
            "short_position_change",
            "aggregate_short_change",
        ),
        "short_interest_change",
    )

    return _cross_sectional_percentile(
        frame,
        values,
    )


def build_short_interest_acceleration(
    frame: pd.DataFrame,
) -> pd.Series:
    values = _require(
        frame,
        (
            "short_interest_pct_change",
            "short_interest_change",
            "short_position_change",
            "aggregate_short_change",
        ),
        "short_interest_acceleration",
    )

    acceleration = values.groupby(
        frame["security_key"]
    ).diff()

    return _cross_sectional_percentile(
        frame,
        acceleration,
    )


def build_price_momentum(
    frame: pd.DataFrame,
) -> pd.Series:
    values = _require(
        frame,
        (
            "price_return_20d",
            "price_return_5d",
        ),
        "price_momentum",
    )

    return _cross_sectional_percentile(
        frame,
        values,
    )


def build_price_volatility(
    frame: pd.DataFrame,
) -> pd.Series:
    values = _require(
        frame,
        (
            "price_volatility_20d",
        ),
        "price_volatility",
    )

    return _cross_sectional_percentile(
        frame,
        values,
    )


def build_distance_from_20d_high(
    frame: pd.DataFrame,
) -> pd.Series:
    values = _require(
        frame,
        (
            "price_distance_from_20d_high",
        ),
        "distance_from_20d_high",
    )

    return _cross_sectional_percentile(
        frame,
        values,
    )


def build_distance_from_60d_high(
    frame: pd.DataFrame,
) -> pd.Series:
    values = _require(
        frame,
        (
            "price_distance_from_60d_high",
        ),
        "distance_from_60d_high",
    )

    return _cross_sectional_percentile(
        frame,
        values,
    )


def build_event_risk(
    frame: pd.DataFrame,
) -> pd.Series:
    """
    Använder befintlig event-risk om den finns.

    Vi försöker inte återskapa event-modellen här.
    Research-lagret ska använda den signal som redan producerats
    av Blankdiss feature/event-pipeline.
    """
    values, _ = _numeric(
        frame,
        (
            "event_score",
            "event_risk",
        ),
    )

    if values is None:
        raise ValueError(
            "Saknar event-risk. Förväntade "
            "'event_score' eller 'event_risk'."
        )

    return _cross_sectional_percentile(
        frame,
        values,
    )


def build_report_timing(
    frame: pd.DataFrame,
) -> pd.Series:
    """
    Enkel rapporttiming.

    Om price_date finns används antal dagar mellan
    snapshot_date och price_date.
    """
    if "price_date" not in frame.columns:
        raise ValueError(
            "Saknar price_date för report timing."
        )

    price_date = pd.to_datetime(
        frame["price_date"],
        errors="coerce",
    )

    snapshot_date = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    return (
        price_date - snapshot_date
    ).dt.days.astype(float)


def build_signals(
    frame: pd.DataFrame,
) -> dict[str, pd.Series]:
    """
    Bygger alla signaler som kan skapas från aktuell data.

    Signaler som kräver kolumner som ännu inte finns hoppas över.
    """
    definitions = (
        SignalDefinition(
            "short_interest_level",
            "Cross-sectional short-interest level.",
            build_short_interest_level,
        ),
        SignalDefinition(
            "short_interest_change",
            "Cross-sectional short-interest change.",
            build_short_interest_change,
        ),
        SignalDefinition(
            "short_interest_acceleration",
            "Change in short-interest change.",
            build_short_interest_acceleration,
        ),
        SignalDefinition(
            "price_momentum",
            "Cross-sectional price momentum.",
            build_price_momentum,
        ),
        SignalDefinition(
            "price_volatility",
            "Cross-sectional price volatility.",
            build_price_volatility,
        ),
        SignalDefinition(
            "distance_from_20d_high",
            "Distance from 20-day high.",
            build_distance_from_20d_high,
        ),
        SignalDefinition(
            "distance_from_60d_high",
            "Distance from 60-day high.",
            build_distance_from_60d_high,
        ),
        SignalDefinition(
            "event_risk",
            "Existing event-risk score.",
            build_event_risk,
        ),
        SignalDefinition(
            "report_timing",
            "Days between snapshot and price date.",
            build_report_timing,
        ),
    )

    result: dict[str, pd.Series] = {}

    for definition in definitions:
        try:
            values = definition.builder(frame)

        except ValueError as exc:
            print(
                f"[research] SKIP {definition.name}: {exc}"
            )
            continue

        result[definition.name] = values

    return result


def add_signal_columns(
    frame: pd.DataFrame,
    signals: dict[str, pd.Series],
) -> pd.DataFrame:
    result = frame.copy()

    for name, values in signals.items():
        result[f"signal_{name}"] = values

    return result


def build_interactions(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Bygger generella interaktioner mellan redan skapade signaler.
    """
    result = frame.copy()

    pairs = (
        (
            "event_risk",
            "short_interest_change",
        ),
        (
            "event_risk",
            "short_interest_acceleration",
        ),
        (
            "event_risk",
            "price_volatility",
        ),
        (
            "event_risk",
            "price_momentum",
        ),
        (
            "short_interest_change",
            "price_volatility",
        ),
        (
            "short_interest_change",
            "price_momentum",
        ),
    )

    for left, right in pairs:
        left_column = f"signal_{left}"
        right_column = f"signal_{right}"

        if (
            left_column not in result.columns
            or right_column not in result.columns
        ):
            continue

        result[
            f"signal_{left}_x_{right}"
        ] = (
            result[left_column]
            * result[right_column]
        )

    return result
