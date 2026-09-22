from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ExperimentResult
from ml.research.signals import build_signal


def _cross_sectional_deciles(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Assign 0-9 deciles independently for each snapshot date."""
    values = pd.to_numeric(frame[column], errors="coerce")

    ranks = values.groupby(frame["snapshot_date"]).rank(
        method="first",
        pct=True,
    )

    deciles = np.ceil(ranks * 10).astype("Int64") - 1
    deciles = deciles.clip(lower=0, upper=9)

    return deciles.fillna(-1).astype(int)


def _numeric_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")

    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean()) if values.notna().any() else float("nan")


def _numeric_median(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")

    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.median()) if values.notna().any() else float("nan")


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    if "snapshot_date" not in result.columns:
        raise ValueError("Momentum/SI analysis requires snapshot_date")

    result["snapshot_date"] = pd.to_datetime(
        result["snapshot_date"],
        errors="coerce",
    )
    result = result.loc[result["snapshot_date"].notna()].copy()

    result["price_momentum_5d"] = build_signal(
        result,
        "price_momentum_5d",
    )
    result["short_interest_change"] = build_signal(
        result,
        "short_interest_change",
    )

    result["momentum_decile"] = _cross_sectional_deciles(
        result,
        "price_momentum_5d",
    )
    result["si_decile"] = _cross_sectional_deciles(
        result,
        "short_interest_change",
    )

    return result


def _cell_rows(
    frame: pd.DataFrame,
    *,
    focus_cells: tuple[tuple[int, int], ...],
    context_columns: tuple[str, ...],
    horizons: tuple[int, ...],
) -> list[dict]:
    rows: list[dict] = []

    for momentum_decile, si_decile in focus_cells:
        cell = frame.loc[
            (frame["momentum_decile"] == momentum_decile - 1)
            & (frame["si_decile"] == si_decile - 1)
        ].copy()

        row: dict = {
            "momentum_decile": momentum_decile,
            "si_decile": si_decile,
            "n": int(len(cell)),
            "momentum_mean": _numeric_mean(
                cell,
                "price_momentum_5d",
            ),
            "momentum_median": _numeric_median(
                cell,
                "price_momentum_5d",
            ),
            "si_change_mean": _numeric_mean(
                cell,
                "short_interest_change",
            ),
            "si_change_median": _numeric_median(
                cell,
                "short_interest_change",
            ),
        }

        for column in context_columns:
            row[f"{column}_mean"] = _numeric_mean(cell, column)
            row[f"{column}_median"] = _numeric_median(cell, column)

        for horizon in horizons:
            column = f"forward_return_{horizon}d"
            row[f"{column}_mean"] = _numeric_mean(cell, column)
            row[f"{column}_median"] = _numeric_median(cell, column)

        rows.append(row)

    return rows


def run_momentum_si_cell_context(
    context,
    *,
    focus_cells: tuple[tuple[int, int], ...],
    context_columns: tuple[str, ...],
    horizons: tuple[int, ...],
) -> ExperimentResult:
    """Analyze pre-defined momentum × SI cells and their price context."""
    frame = _prepare_frame(context.test)

    result = ExperimentResult(
        name="momentum_si_cell_context",
        description=(
            "Testar om de förut identifierade momentum × SI-cellerna "
            "har en särskild pris- och riskkontext."
        ),
    )

    result.add_table(
        "cell_context",
        _cell_rows(
            frame,
            focus_cells=focus_cells,
            context_columns=context_columns,
            horizons=horizons,
        ),
    )
    result.add_metric("test_rows", int(len(frame)))
    result.add_metric("focus_cell_count", int(len(focus_cells)))

    return result
