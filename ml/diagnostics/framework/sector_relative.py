from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .base import ExperimentResult
from .si_additional import (
    _add_si_change,
    _numeric,
)


SECTOR_MAP_PATH = Path(
    "data/analysis/sector_map.json"
)


def _load_sector_map() -> dict[str, str]:
    if not SECTOR_MAP_PATH.exists():
        raise FileNotFoundError(
            "Sektormappning saknas: "
            f"{SECTOR_MAP_PATH}. "
            "Skapa och frys sektormappningen innan "
            "sector_relative_return aktiveras."
        )

    payload = json.loads(
        SECTOR_MAP_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "sector_map.json måste innehålla "
            "ett JSON-objekt."
        )

    instruments = payload.get(
        "instruments"
    )

    if not isinstance(
        instruments,
        dict,
    ):
        raise ValueError(
            "sector_map.json saknar "
            "'instruments'."
        )

    mapping: dict[str, str] = {}

    for symbol, item in instruments.items():
        if not isinstance(
            item,
            dict,
        ):
            continue

        sector = item.get(
            "sector"
        )

        if sector is None:
            continue

        sector = str(
            sector
        ).strip()

        if sector:
            mapping[
                str(symbol)
            ] = sector

    if not mapping:
        raise ValueError(
            "sector_map.json innehåller "
            "ingen giltig sektormappning."
        )

    return mapping


def _prepare_frame(
    frame: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    result = _add_si_change(
        frame
    ).copy()

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
        .astype(str)
        .map(sector_map)
    )

    result[
        "short_interest_pct_change"
    ] = _numeric(
        result,
        "short_interest_pct_change",
    )

    result["snapshot_date"] = pd.to_datetime(
        result["snapshot_date"],
        errors="coerce",
    )

    return result


def _relative_returns(
    frame: pd.DataFrame,
    *,
    return_column: str,
) -> pd.DataFrame:
    required = {
        "snapshot_date",
        "sector",
        return_column,
    }

    if not required.issubset(
        frame.columns
    ):
        return pd.DataFrame(
            columns=[
                "sector_relative_return",
                "market_relative_return",
            ]
        )

    result = frame[
        [
            "snapshot_date",
            "sector",
            return_column,
        ]
    ].copy()

    result["_return"] = pd.to_numeric(
        result[return_column],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            "snapshot_date",
            "sector",
            "_return",
        ]
    )

    if result.empty:
        result[
            "sector_relative_return"
        ] = np.nan

        result[
            "market_relative_return"
        ] = np.nan

        return result

    sector_mean = (
        result.groupby(
            [
                "snapshot_date",
                "sector",
            ]
        )["_return"]
        .transform("mean")
    )

    market_mean = (
        result.groupby(
            "snapshot_date"
        )["_return"]
        .transform("mean")
    )

    result[
        "sector_relative_return"
    ] = (
        result["_return"]
        - sector_mean
    )

    result[
        "market_relative_return"
    ] = (
        result["_return"]
        - market_mean
    )

    return result


def _mean(
    frame: pd.DataFrame,
    column: str,
) -> float:
    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return float("nan")

    return float(
        values.mean()
    )


def _median(
    frame: pd.DataFrame,
    column: str,
) -> float:
    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return float("nan")

    return float(
        values.median()
    )


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

    mapped_test = test[
        test["sector"].notna()
    ].copy()

    if mapped_test.empty:
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

    for change_cutoff in (
        si_change_cutoffs
    ):
        threshold = float(
            positive_changes.quantile(
                1.0 - change_cutoff
            )
        )

        signal = (
            _numeric(
                mapped_test,
                "short_interest_pct_change",
            )
            >= threshold
        )

        signal_frame = mapped_test[
            signal
        ].copy()

        control_frame = mapped_test[
            ~signal
        ].copy()

        for horizon in horizons:
            return_column = (
                f"forward_return_{horizon}d"
            )

            if return_column not in mapped_test.columns:
                continue

            # IMPORTANT:
            # Benchmarkerna beräknas på hela testuniversumet.
            relative = _relative_returns(
                mapped_test,
                return_column=return_column,
            )

            if relative.empty:
                continue

            benchmark_index = relative.index

            signal_index = (
                signal_frame.index
                .intersection(
                    benchmark_index
                )
            )

            control_index = (
                control_frame.index
                .intersection(
                    benchmark_index
                )
            )

            signal_relative = relative.loc[
                signal_index
            ]

            control_relative = relative.loc[
                control_index
            ]

            signal_sector_mean = _mean(
                signal_relative,
                "sector_relative_return",
            )

            control_sector_mean = _mean(
                control_relative,
                "sector_relative_return",
            )

            signal_market_mean = _mean(
                signal_relative,
                "market_relative_return",
            )

            control_market_mean = _mean(
                control_relative,
                "market_relative_return",
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
                        len(signal_relative),
                    "control_n":
                        len(control_relative),
                    "signal_sector_relative_mean":
                        signal_sector_mean,
                    "control_sector_relative_mean":
                        control_sector_mean,
                    "sector_relative_delta":
                        (
                            signal_sector_mean
                            - control_sector_mean
                        ),
                    "signal_sector_relative_median":
                        _median(
                            signal_relative,
                            "sector_relative_return",
                        ),
                    "control_sector_relative_median":
                        _median(
                            control_relative,
                            "sector_relative_return",
                        ),
                    "signal_market_relative_mean":
                        signal_market_mean,
                    "control_market_relative_mean":
                        control_market_mean,
                    "market_relative_delta":
                        (
                            signal_market_mean
                            - control_market_mean
                        ),
                    "mapped_test_fraction":
                        (
                            len(mapped_test)
                            / len(test)
                            if len(test)
                            else 0.0
                        ),
                }
            )

    if not rows:
        raise ValueError(
            "Sector-relative analysis producerade "
            "inga resultat."
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

    result.add_metric(
        "mapped_test_rows",
        int(len(mapped_test)),
    )

    result.add_metric(
        "test_rows",
        int(len(test)),
    )

    result.add_metric(
        "mapped_test_fraction",
        (
            float(
                len(mapped_test)
                / len(test)
            )
            if len(test)
            else 0.0
        ),
    )

    return result
