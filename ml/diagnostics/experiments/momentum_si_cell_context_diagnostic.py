from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.config import TARGETS
from ml.dataset import build_target, load_features
from ml.research.signals import build_signal


DATE_CUTOFF = pd.Timestamp("2025-12-19")

TARGET_NAMES = (
    "down_5pct_5d",
    "down_7pct_5d",
    "down_10pct_5d",
)

MOMENTUM_SIGNAL_NAME = "price_momentum_5d"
SI_CHANGE_SIGNAL_NAME = "short_interest_change"
SI_LEVEL_COLUMN = "short_interest_pct"

MOMENTUM_COLUMNS = (
    "price_momentum_5d",
    "price_momentum_20d",
    "price_momentum_60d",
)

CONTEXT_COLUMNS = (
    "price_volatility_20d",
    "price_distance_from_20d_high",
    "price_distance_from_60d_high",
)

FORWARD_RETURN_HORIZONS = (
    1,
    3,
    5,
    10,
    20,
)

MIN_CELL_N = 50

OUTPUT_DIR = Path(
    "data/processed/ml/research/momentum_si_cell_context"
)

SECTOR_MAP_PATH = Path(
    "data/analysis/sector_map.json"
)

# These are the strongest/repeating coordinates from the
# pre-specified momentum × SI stability analysis.
#
# The diagnostic does not search for new cells. It only
# describes these already identified coordinates.
FOCUS_CELLS = (
    (1, 3),
    (10, 6),
    (6, 9),
    (9, 10),
    (7, 10),
)


def get_target(name: str):
    for target in TARGETS:
        if target.name == name:
            return target

    raise ValueError(
        f"Unknown target: {name}"
    )


def load_sector_map() -> dict[str, str]:
    if not SECTOR_MAP_PATH.exists():
        return {}

    payload = json.loads(
        SECTOR_MAP_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "sector_map.json must contain a JSON object."
        )

    instruments = payload.get(
        "instruments"
    )

    if not isinstance(
        instruments,
        dict,
    ):
        return {}

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

    return mapping


def cross_sectional_deciles(
    frame: pd.DataFrame,
    values: pd.Series,
) -> np.ndarray:
    """
    Assign cross-sectional deciles independently per
    snapshot date.

    0 = lowest decile
    9 = highest decile
    """

    result = np.full(
        len(frame),
        -1,
        dtype=np.int8,
    )

    working = pd.DataFrame(
        {
            "snapshot_date": frame[
                "snapshot_date"
            ],
            "value": pd.to_numeric(
                values,
                errors="coerce",
            ),
        },
        index=frame.index,
    )

    for _, index in working.groupby(
        "snapshot_date",
        sort=False,
    ).groups.items():
        local = working.loc[
            index,
            "value",
        ]

        valid = local.notna()

        if not valid.any():
            continue

        ranks = local.loc[
            valid
        ].rank(
            method="first",
            pct=True,
        ).to_numpy()

        bins = (
            np.ceil(
                ranks * 10
            ).astype(int)
            - 1
        )

        bins = np.clip(
            bins,
            0,
            9,
        )

        positions = (
            frame.index.get_indexer(
                local.loc[
                    valid
                ].index
            )
        )

        result[
            positions
        ] = bins

    return result


def safe_mean(
    values: pd.Series | np.ndarray,
) -> float | None:
    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(
        values.mean()
    )


def safe_median(
    values: pd.Series | np.ndarray,
) -> float | None:
    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(
        values.median()
    )


def event_statistics(
    values: pd.Series | np.ndarray,
) -> dict[str, Any]:
    numeric = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return {
            "n": 0,
            "events": 0,
            "event_rate": None,
        }

    events = (
        numeric > 0
    )

    return {
        "n": int(
            len(numeric)
        ),
        "events": int(
            events.sum()
        ),
        "event_rate": float(
            events.mean()
        ),
    }


def add_sector_information(
    frame: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    result = frame.copy()

    symbol_column = (
        "yahoo_symbol"
        if "yahoo_symbol" in result.columns
        else "security_key"
    )

    if symbol_column not in result.columns:
        result["sector"] = pd.NA
        return result

    result["sector"] = (
        result[
            symbol_column
        ]
        .astype(str)
        .map(sector_map)
    )

    return result


def add_relative_returns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate sector- and market-relative forward returns.

    Benchmarks exclude the stock itself.

    This is deliberately calculated on each snapshot date
    and each forward-return horizon.
    """

    result = frame.copy()

    symbol_column = (
        "yahoo_symbol"
        if "yahoo_symbol" in result.columns
        else "security_key"
    )

    if (
        "sector" not in result.columns
        or symbol_column not in result.columns
    ):
        return result

    for horizon in FORWARD_RETURN_HORIZONS:
        return_column = (
            f"forward_return_{horizon}d"
        )

        if return_column not in result.columns:
            continue

        working = result[
            [
                "snapshot_date",
                symbol_column,
                "sector",
                return_column,
            ]
        ].copy()

        working["_return"] = pd.to_numeric(
            working[
                return_column
            ],
            errors="coerce",
        )

        working = working.dropna(
            subset=[
                "snapshot_date",
                symbol_column,
                "sector",
                "_return",
            ]
        )

        if working.empty:
            continue

        stock_returns = (
            working.groupby(
                [
                    "snapshot_date",
                    symbol_column,
                ],
                as_index=False,
            )
            .agg(
                sector=(
                    "sector",
                    "first",
                ),
                _return=(
                    "_return",
                    "first",
                ),
            )
        )

        sector_sum = (
            stock_returns.groupby(
                [
                    "snapshot_date",
                    "sector",
                ]
            )["_return"]
            .transform("sum")
        )

        sector_count = (
            stock_returns.groupby(
                [
                    "snapshot_date",
                    "sector",
                ]
            )["_return"]
            .transform("count")
        )

        market_sum = (
            stock_returns.groupby(
                "snapshot_date"
            )["_return"]
            .transform("sum")
        )

        market_count = (
            stock_returns.groupby(
                "snapshot_date"
            )["_return"]
            .transform("count")
        )

        sector_ex_self = np.where(
            sector_count > 1,
            (
                sector_sum
                - stock_returns[
                    "_return"
                ]
            )
            / (
                sector_count - 1
            ),
            np.nan,
        )

        market_ex_self = np.where(
            market_count > 1,
            (
                market_sum
                - stock_returns[
                    "_return"
                ]
            )
            / (
                market_count - 1
            ),
            np.nan,
        )

        stock_returns[
            f"_sector_relative_{horizon}d"
        ] = (
            stock_returns[
                "_return"
            ]
            - sector_ex_self
        )

        stock_returns[
            f"_market_relative_{horizon}d"
        ] = (
            stock_returns[
                "_return"
            ]
            - market_ex_self
        )

        merge_columns = [
            "snapshot_date",
            symbol_column,
            f"_sector_relative_{horizon}d",
            f"_market_relative_{horizon}d",
        ]

        result = result.merge(
            stock_returns[
                merge_columns
            ],
            on=[
                "snapshot_date",
                symbol_column,
            ],
            how="left",
        )

    return result


def prepare_data() -> pd.DataFrame:
    print(
        "Loading features...",
        flush=True,
    )

    frame = load_features().copy()

    if "snapshot_date" not in frame.columns:
        raise ValueError(
            "Feature data must contain "
            "'snapshot_date'."
        )

    frame["snapshot_date"] = pd.to_datetime(
        frame[
            "snapshot_date"
        ],
        errors="coerce",
    )

    frame = frame.loc[
        frame[
            "snapshot_date"
        ].notna()
    ].copy()

    frame = frame.loc[
        frame[
            "snapshot_date"
        ] <= DATE_CUTOFF
    ].copy()

    momentum = build_signal(
        frame,
        MOMENTUM_SIGNAL_NAME,
    )

    si_change = build_signal(
        frame,
        SI_CHANGE_SIGNAL_NAME,
    )

    frame[
        "momentum"
    ] = pd.to_numeric(
        momentum,
        errors="coerce",
    )

    frame[
        "si_change"
    ] = pd.to_numeric(
        si_change,
        errors="coerce",
    )

    if SI_LEVEL_COLUMN in frame.columns:
        frame[
            "si_level"
        ] = pd.to_numeric(
            frame[
                SI_LEVEL_COLUMN
            ],
            errors="coerce",
        )
    else:
        frame[
            "si_level"
        ] = np.nan

    for column in (
        MOMENTUM_COLUMNS
        + CONTEXT_COLUMNS
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    for target_name in TARGET_NAMES:
        frame[target_name] = (
            build_target(
                frame,
                get_target(
                    target_name
                ),
            )
        )

    sector_map = load_sector_map()

    frame = add_sector_information(
        frame,
        sector_map,
    )

    frame = add_relative_returns(
        frame
    )

    frame = frame.sort_values(
        [
            "snapshot_date",
            "security_key",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    print(
        f"Prepared rows: {len(frame):,}",
        flush=True,
    )

    print(
        f"Sector mappings: "
        f"{len(sector_map):,}",
        flush=True,
    )

    return frame


def describe_cell(
    frame: pd.DataFrame,
    momentum_bins: np.ndarray,
    si_bins: np.ndarray,
    momentum_decile: int,
    si_decile: int,
    target_name: str,
) -> dict[str, Any]:
    """
    Describe one fixed momentum × SI cell.

    The target is evaluated inside the cell, while all
    context statistics describe the observations belonging
    to that same cell.
    """

    cell = frame.loc[
        (
            momentum_bins
            == momentum_decile - 1
        )
        & (
            si_bins
            == si_decile - 1
        )
    ].copy()

    target = pd.to_numeric(
        cell[
            target_name
        ],
        errors="coerce",
    )

    target_stats = event_statistics(
        target
    )

    row: dict[str, Any] = {
        "target": target_name,
        "momentum_decile": momentum_decile,
        "si_decile": si_decile,
        "n": int(
            len(cell)
        ),
        "events": target_stats[
            "events"
        ],
        "event_rate": target_stats[
            "event_rate"
        ],
        "mean_momentum_5d": safe_mean(
            cell.get(
                "price_momentum_5d",
                pd.Series(
                    dtype=float
                ),
            )
        ),
        "median_momentum_5d": safe_median(
            cell.get(
                "price_momentum_5d",
                pd.Series(
                    dtype=float
                ),
            )
        ),
        "mean_si_change": safe_mean(
            cell[
                "si_change"
            ]
        ),
        "median_si_change": safe_median(
            cell[
                "si_change"
            ]
        ),
        "mean_si_level": safe_mean(
            cell[
                "si_level"
            ]
        ),
        "median_si_level": safe_median(
            cell[
                "si_level"
            ]
        ),
    }

    for column in (
        "price_momentum_20d",
        "price_momentum_60d",
        "price_volatility_20d",
        "price_distance_from_20d_high",
        "price_distance_from_60d_high",
    ):
        row[
            f"mean_{column}"
        ] = safe_mean(
            cell.get(
                column,
                pd.Series(
                    dtype=float
                ),
            )
        )

        row[
            f"median_{column}"
        ] = safe_median(
            cell.get(
                column,
                pd.Series(
                    dtype=float
                ),
            )
        )

    for horizon in FORWARD_RETURN_HORIZONS:
        return_column = (
            f"forward_return_{horizon}d"
        )

        if return_column in cell.columns:
            row[
                f"mean_forward_return_{horizon}d"
            ] = safe_mean(
                cell[
                    return_column
                ]
            )

            row[
                f"median_forward_return_{horizon}d"
            ] = safe_median(
                cell[
                    return_column
                ]
            )

        sector_column = (
            f"_sector_relative_{horizon}d"
        )

        if sector_column in cell.columns:
            row[
                f"mean_sector_relative_return_{horizon}d"
            ] = safe_mean(
                cell[
                    sector_column
                ]
            )

            row[
                f"median_sector_relative_return_{horizon}d"
            ] = safe_median(
                cell[
                    sector_column
                ]
            )

        market_column = (
            f"_market_relative_{horizon}d"
        )

        if market_column in cell.columns:
            row[
                f"mean_market_relative_return_{horizon}d"
            ] = safe_mean(
                cell[
                    market_column
                ]
            )

            row[
                f"median_market_relative_return_{horizon}d"
            ] = safe_median(
                cell[
                    market_column
                ]
            )

    if "sector" in cell.columns:
        sector_counts = (
            cell[
                "sector"
            ]
            .dropna()
            .astype(str)
            .value_counts()
        )

        row[
            "sector_mapped_n"
        ] = int(
            sector_counts.sum()
        )

        row[
            "sector_count"
        ] = int(
            len(sector_counts)
        )

        row[
            "top_sector"
        ] = (
            str(
                sector_counts.index[0]
            )
            if len(sector_counts)
            else None
        )

        row[
            "top_sector_fraction"
        ] = (
            float(
                sector_counts.iloc[0]
                / sector_counts.sum()
            )
            if len(sector_counts)
            else None
        )

    return row


def build_sector_breakdown(
    frame: pd.DataFrame,
    momentum_bins: np.ndarray,
    si_bins: np.ndarray,
) -> pd.DataFrame:
    """
    Sector-level event-rate breakdown for the fixed focus cells.

    This does not select sectors by performance. It reports
    all sufficiently populated sectors represented in the
    focus cells.
    """

    rows: list[dict[str, Any]] = []

    if "sector" not in frame.columns:
        return pd.DataFrame()

    for momentum_decile, si_decile in FOCUS_CELLS:
        cell_mask = (
            (momentum_bins == momentum_decile - 1)
            & (si_bins == si_decile - 1)
        )

        cell = frame.loc[
            cell_mask
        ].copy()

        if cell.empty:
            continue

        for sector, sector_frame in (
            cell.dropna(
                subset=[
                    "sector"
                ]
            )
            .groupby(
                "sector",
                sort=True,
            )
        ):
            if len(sector_frame) < MIN_CELL_N:
                continue

            row: dict[str, Any] = {
                "momentum_decile": momentum_decile,
                "si_decile": si_decile,
                "sector": str(
                    sector
                ),
                "n": int(
                    len(sector_frame)
                ),
            }

            for target_name in TARGET_NAMES:
                stats = event_statistics(
                    sector_frame[
                        target_name
                    ]
                )

                row[
                    f"{target_name}_event_rate"
                ] = stats[
                    "event_rate"
                ]

                row[
                    f"{target_name}_events"
                ] = stats[
                    "events"
                ]

            for horizon in (
                1,
                3,
                5,
            ):
                column = (
                    f"forward_return_{horizon}d"
                )

                if column in sector_frame.columns:
                    row[
                        f"mean_forward_return_{horizon}d"
                    ] = safe_mean(
                        sector_frame[
                            column
                        ]
                    )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def build_half_breakdown(
    frame: pd.DataFrame,
    momentum_bins: np.ndarray,
    si_bins: np.ndarray,
) -> pd.DataFrame:
    """
    Re-express the fixed cells by H1/H2.

    This is descriptive only. The stability test itself
    remains the pre-existing analysis.
    """

    rows: list[dict[str, Any]] = []

    frame = frame.copy()

    frame["half"] = np.where(
        frame[
            "snapshot_date"
        ] <= pd.Timestamp(
            "2025-06-30"
        ),
        "H1",
        "H2",
    )

    for momentum_decile, si_decile in FOCUS_CELLS:
        for half in (
            "H1",
            "H2",
        ):
            mask = (
                (
                    momentum_bins
                    == momentum_decile - 1
                )
                & (
                    si_bins
                    == si_decile - 1
                )
                & (
                    frame["half"]
                    == half
                )
            )

            cell = frame.loc[
                mask
            ]

            row: dict[str, Any] = {
                "half": half,
                "momentum_decile": momentum_decile,
                "si_decile": si_decile,
                "n": int(
                    len(cell)
                ),
                "mean_si_change": safe_mean(
                    cell[
                        "si_change"
                    ]
                ),
                "median_si_change": safe_median(
                    cell[
                        "si_change"
                    ]
                ),
            }

            for target_name in TARGET_NAMES:
                stats = event_statistics(
                    cell[
                        target_name
                    ]
                )

                row[
                    f"{target_name}_event_rate"
                ] = stats[
                    "event_rate"
                ]

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def write_report(
    cell_results: pd.DataFrame,
    sector_results: pd.DataFrame,
    half_results: pd.DataFrame,
    available_columns: list[str],
    sector_map_size: int,
) -> None:
    lines = [
        "# Momentum × SI Cell Context",
        "",
        "## Purpose",
        "",
        (
            "Descriptive follow-up of the fixed "
            "Momentum × Short Interest cells identified "
            "by the temporal stability analysis."
        ),
        "",
        (
            "This diagnostic does not search for new "
            "cells. It describes the existing focus "
            "coordinates and their market/sector context."
        ),
        "",
        "## Focus cells",
        "",
        "| Momentum decile | SI decile |",
        "|---:|---:|",
    ]

    for momentum_decile, si_decile in FOCUS_CELLS:
        lines.append(
            f"| {momentum_decile} | {si_decile} |"
        )

    lines.extend(
        [
            "",
            "## Data",
            "",
            f"- Date cutoff: `{DATE_CUTOFF.date()}`",
            f"- Minimum cell N: `{MIN_CELL_N}`",
            f"- Sector mappings available: `{sector_map_size}`",
            "",
            "### Available context columns",
            "",
        ]
    )

    if available_columns:
        for column in available_columns:
            lines.append(
                f"- `{column}`"
            )
    else:
        lines.append(
            "- No optional context columns detected."
        )

    lines.extend(
        [
            "",
            "## Main observations",
            "",
        ]
    )

    if cell_results.empty:
        lines.append(
            "No focus-cell observations were produced."
        )
    else:
        for _, row in cell_results.iterrows():
            lines.append(
                "- "
                f"`{row['target']}` "
                f"M{int(row['momentum_decile'])}/"
                f"SI{int(row['si_decile'])}: "
                f"N={int(row['n'])}, "
                f"event rate="
                f"{row['event_rate']:.4f}"
                if pd.notna(
                    row["event_rate"]
                )
                else
                "- "
                f"`{row['target']}` "
                f"M{int(row['momentum_decile'])}/"
                f"SI{int(row['si_decile'])}: "
                f"N={int(row['n'])}, "
                "event rate=n/a"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The purpose is to identify whether "
                "replicating cells correspond to distinct "
                "SI-change, momentum, volatility or "
                "relative-return regimes."
            ),
            "",
            (
                "Sector-relative returns use the frozen "
                "sector mapping and exclude the stock "
                "itself from the sector and market "
                "benchmark."
            ),
            "",
            (
                "This diagnostic is descriptive and "
                "does not establish causality or "
                "profitability."
            ),
            "",
            (
                "Report-date proximity is deliberately "
                "not mixed into this analysis. It should "
                "be tested as a separate hypothesis."
            ),
        ]
    )

    (
        OUTPUT_DIR
        / "report.md"
    ).write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=== Momentum × SI Cell Context Analysis ===",
        flush=True,
    )

    frame = prepare_data()

    momentum_bins = cross_sectional_deciles(
        frame,
        frame["momentum"],
    )

    si_bins = cross_sectional_deciles(
        frame,
        frame["si_change"],
    )

    cell_rows: list[dict[str, Any]] = []

    for target_name in TARGET_NAMES:
        for momentum_decile, si_decile in FOCUS_CELLS:
            row = describe_cell(
                frame,
                momentum_bins,
                si_bins,
                momentum_decile,
                si_decile,
                target_name,
            )

            cell_rows.append(
                row
            )

    cell_results = pd.DataFrame(
        cell_rows
    )

    sector_results = build_sector_breakdown(
        frame,
        momentum_bins,
        si_bins,
    )

    half_results = build_half_breakdown(
        frame,
        momentum_bins,
        si_bins,
    )

    cell_results.to_csv(
        OUTPUT_DIR
        / "cell_context.csv",
        index=False,
    )

    sector_results.to_csv(
        OUTPUT_DIR
        / "sector_breakdown.csv",
        index=False,
    )

    half_results.to_csv(
        OUTPUT_DIR
        / "half_breakdown.csv",
        index=False,
    )

    available_columns = [
        column
        for column in (
            MOMENTUM_COLUMNS
            + CONTEXT_COLUMNS
            + (
                "forward_return_1d",
                "forward_return_3d",
                "forward_return_5d",
                "forward_return_10d",
                "forward_return_20d",
            )
        )
        if column in frame.columns
    ]

    metadata = {
        "analysis": (
            "momentum_si_cell_context"
        ),
        "date_cutoff": str(
            DATE_CUTOFF.date()
        ),
        "targets": list(
            TARGET_NAMES
        ),
        "focus_cells": [
            {
                "momentum_decile": momentum,
                "si_decile": si,
            }
            for momentum, si in FOCUS_CELLS
        ],
        "min_cell_n": MIN_CELL_N,
        "rows": int(
            len(frame)
        ),
        "sector_map_size": int(
            len(
                load_sector_map()
            )
        ),
        "available_context_columns": (
            available_columns
        ),
        "forward_return_horizons": list(
            FORWARD_RETURN_HORIZONS
        ),
        "method": {
            "cell_selection": (
                "Fixed coordinates from the "
                "pre-existing temporal stability "
                "analysis; no new cell search."
            ),
            "deciles": (
                "Cross-sectional per snapshot_date."
            ),
            "sector_relative_return": (
                "Stock return minus same-date "
                "sector mean excluding the stock."
            ),
            "market_relative_return": (
                "Stock return minus same-date "
                "market mean excluding the stock."
            ),
        },
    }

    (
        OUTPUT_DIR
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_report(
        cell_results,
        sector_results,
        half_results,
        available_columns,
        len(
            load_sector_map()
        ),
    )

    results = {
        "analysis": (
            "momentum_si_cell_context"
        ),
        "status": "completed",
        "n_rows": int(
            len(frame)
        ),
        "focus_cells": [
            {
                "momentum_decile": momentum,
                "si_decile": si,
            }
            for momentum, si in FOCUS_CELLS
        ],
        "cell_result_rows": int(
            len(cell_results)
        ),
        "sector_result_rows": int(
            len(sector_results)
        ),
        "half_result_rows": int(
            len(half_results)
        ),
    }

    (
        OUTPUT_DIR
        / "results.json"
    ).write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "Results written to:",
        OUTPUT_DIR,
        flush=True,
    )


if __name__ == "__main__":
    main()
