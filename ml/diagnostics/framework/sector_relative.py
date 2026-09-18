from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pandas as pd

from .base import ExperimentResult
from .si_additional import _add_si_change, _numeric


SECTOR_MAP_PATH = Path(
    "data/analysis/sector_map.json"
)


def _load_sector_map() -> dict[str, str]:
    if not SECTOR_MAP_PATH.exists():
        raise FileNotFoundError(
            "Sektormappning saknas: "
            f"{SECTOR_MAP_PATH}. "
            "Skapa sektormappningen innan "
            "sector_relative_return aktiveras."
        )

    with SECTOR_MAP_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        mapping = json.load(handle)

    if not isinstance(mapping, dict):
        raise ValueError(
            "sector_map.json måste innehålla "
            "ett JSON-objekt."
        )

    return {
        str(key): str(value)
        for key, value in mapping.items()
        if value is not None
    }


def _prepare_frame(
    frame: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    result = _add_si_change(
        frame
    )

    symbol_column = (
        "yahoo_symbol"
        if "yahoo_symbol" in result.columns
        else "security_key"
    )

    required = {
        symbol_column,
        "snapshot_date",
        "short_interest_pct_change",
    }

    missing = [
        column
        for column in required
        if column not in result.columns
    ]

    if missing:
        raise KeyError(
            "Sector-relative analysis saknar "
            "kolumner: "
            + ", ".join(missing)
        )

    result["sector"] = (
        result[symbol_column]
        .map(sector_map)
    )

    result["short_interest_pct_change"] = _numeric(
        result,
        "short_interest_pct_change",
    )

    result["snapshot_date"] = pd.to_datetime(
        result["snapshot_date"],
        errors="coerce",
    )

    return result


def _sector_relative_return(
    frame: pd.DataFrame,
    *,
    return_column: str,
) -> pd.Series:
    values = pd.to_numeric(
        frame[return_column],
        errors="coerce",
    )

    valid = frame.copy()
    valid["_return"] = values

    valid = valid.dropna(
        subset=[
            "sector",
            "_return",
        ]
    )

    if valid.empty:
        return pd.Series(
            dtype=float
        )

    sector_mean = (
        valid.groupby(
            ["snapshot_date", "sector"]
        )["_return"]
        .transform("mean")
    )

    return (
        valid["_return"]
        - sector_mean
    )


def _market_relative_return(
    frame: pd.DataFrame,
    *,
    return_column: str,
) -> pd.Series:
    values = pd.to_numeric(
        frame[return_column],
        errors="coerce",
    )

    valid = frame.copy()
    valid["_return"] = values

    valid = valid.dropna(
        subset=["_return"]
    )

    if valid.empty:
        return pd.Series(
            dtype=float
        )

    market_mean = (
        valid.groupby(
            "snapshot_date"
        )["_return"]
        .transform("mean")
    )

    return (
        valid["_return"]
        - market_mean
    )


def _quantile(
    frame: pd.DataFrame,
    column: str,
    q: float,
) -> float:
    values = _numeric(
        frame,
        column,
    ).dropna()

    if values.empty:
        return float("nan")

    return float(
        values.quantile(q)
    )


def _summary(
    values: pd.Series,
) -> dict[str, float | int]:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if values.empty:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
        }

    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
    }


def run_sector_relative_return(
    context,
    *,
    horizons: tuple[int, ...],
    si_change_cutoffs: tuple[float, ...],
) -> ExperimentResult:
    sector_map = _load_sector_map()

    pretest = _prepare_frame(
        context.pretest,
        sector_map,
    )

    test = _prepare_frame(
        context.test,
        sector_map,
    )

    if test["sector"].notna().sum() == 0:
        raise ValueError(
            "Ingen testobservation kunde kopplas "
            "till en sektor."
        )

    positive_changes = _numeric(
        pretest,
        "short_interest_pct_change",
    )

    positive_changes = positive_changes[
        positive_changes > 0
    ].dropna()

    if positive_changes.empty:
        raise ValueError(
            "Ingen positiv förändring i short interest "
            "finns i pretest."
        )

    rows: list[dict] = []

    for change_cutoff in si_change_cutoffs:
        threshold = float(
            positive_changes.quantile(
                1.0 - change_cutoff
            )
        )

        signal = (
            _numeric(
                test,
                "short_interest_pct_change",
            )
            >= threshold
        )

        signal_frame = test[
            signal
        ].copy()

        control_frame = test[
            ~signal
        ].copy()

        signal_frame = signal_frame[
            signal_frame["sector"].notna()
        ]

        control_frame = control_frame[
            control_frame["sector"].notna()
        ]

        for horizon in horizons:
            return_column = (
                f"forward_return_{horizon}d"
            )

            if return_column not in test.columns:
                continue

            signal_sector = (
                _sector_relative_return(
                    signal_frame,
                    return_column=return_column,
                )
            )

            control_sector = (
                _sector_relative_return(
                    control_frame,
                    return_column=return_column,
                )
            )

            signal_market = (
                _market_relative_return(
                    signal_frame,
                    return_column=return_column,
                )
            )

            control_market = (
                _market_relative_return(
                    control_frame,
                    return_column=return_column,
                )
            )

            signal_sector_summary = _summary(
                signal_sector
            )
            control_sector_summary = _summary(
                control_sector
            )
            signal_market_summary = _summary(
                signal_market
            )
            control_market_summary = _summary(
                control_market
            )

            sector_delta = (
                signal_sector_summary["mean"]
                - control_sector_summary["mean"]
            )

            market_delta = (
                signal_market_summary["mean"]
                - control_market_summary["mean"]
            )

            rows.append(
                {
                    "change_cutoff":
                        change_cutoff,
                    "change_threshold":
                        threshold,
                    "horizon_days":
                        horizon,
                    "signal_n":
                        signal_sector_summary["n"],
                    "control_n":
                        control_sector_summary["n"],
                    "signal_sector_relative_mean":
                        signal_sector_summary["mean"],
                    "control_sector_relative_mean":
                        control_sector_summary["mean"],
                    "sector_relative_delta":
                        sector_delta,
                    "signal_market_relative_mean":
                        signal_market_summary["mean"],
                    "control_market_relative_mean":
                        control_market_summary["mean"],
                    "market_relative_delta":
                        market_delta,
                }
            )

    result = ExperimentResult(
        name="sector_relative_return",
        description=(
            "Testar om sambandet mellan stora "
            "förändringar i short interest och "
            "framtida avkastning kvarstår efter "
            "kontroll mot sektor och marknad."
        ),
    )

    result.add_table(
        "sector_relative_returns",
        pd.DataFrame(rows),
    )

    result.add_metric(
        "sector_map_size",
        len(sector_map),
    )

    return result
