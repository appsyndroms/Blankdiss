from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.config import TARGETS
from ml.research.signals import build_signal

from .base import ExperimentResult


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

SECTOR_MAP_PATH = Path(
    "data/analysis/sector_map.json"
)

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


def _ensure_targets(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    for target_name in TARGET_NAMES:
        if target_name in result.columns:
            continue

        result[target_name] = build_target(
            result,
            get_target(
                target_name
            ),
        )

    return result


def _prepare_frame(
    frame: pd.DataFrame,
    *,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    result = frame.copy()

    if "snapshot_date" not in result.columns:
        raise KeyError(
            "Momentum/SI analysis requires "
            "'snapshot_date'."
        )

    result["snapshot_date"] = pd.to_datetime(
        result["snapshot_date"],
        errors="coerce",
    )

    result = result.loc[
        result["snapshot_date"].notna()
    ].copy()

    result = result.loc[
        result["snapshot_date"]
        <= DATE_CUTOFF
    ].copy()

    momentum = build_signal(
        result,
        MOMENTUM_SIGNAL_NAME,
    )

    si_change = build_signal(
        result,
        SI_CHANGE_SIGNAL_NAME,
    )

    result["momentum"] = pd.to_numeric(
        momentum,
        errors="coerce",
    )

    result["si_change"] = pd.to_numeric(
        si_change,
        errors="coerce",
    )

    if SI_LEVEL_COLUMN in result.columns:
        result["si_level"] = pd.to_numeric(
            result[
                SI_LEVEL_COLUMN
            ],
            errors="coerce",
        )
    else:
        result["si_level"] = np.nan

    for column in (
        MOMENTUM_COLUMNS
        + CONTEXT_COLUMNS
    ):
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    result = _ensure_targets(
        result
    )

    symbol_column = (
        "yahoo_symbol"
        if "yahoo_symbol" in result.columns
        else "security_key"
    )

    if symbol_column in result.columns:
        result["sector"] = (
            result[
                symbol_column
            ]
            .astype(str)
            .map(sector_map)
        )
    else:
        result["sector"] = pd.NA

    return result


def add_relative_returns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate sector- and market-relative forward returns.

    Benchmarks exclude the stock itself.

    The original row index is preserved so that downstream
    cell selection remains aligned with the source frame.
    """

    result = frame.copy()

    symbol_column = (
        "yahoo_symbol"
        if "yahoo_symbol" in result.columns
        else "security_key"
    )

    required = {
        "snapshot_date",
        "sector",
        symbol_column,
    }

    if not required.issubset(
        result.columns
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

        working["_original_index"] = (
            working.index
        )

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

        stock_returns[
            "_sector_mean_ex_self"
        ] = np.where(
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

        stock_returns[
            "_market_mean_ex_self"
        ] = np.where(
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
            - stock_returns[
                "_sector_mean_ex_self"
            ]
        )

        stock_returns[
            f"_market_relative_{horizon}d"
        ] = (
            stock_returns[
                "_return"
            ]
            - stock_returns[
                "_market_mean_ex_self"
            ]
        )

        relative = stock_returns[
            [
                "snapshot_date",
                symbol_column,
                f"_sector_relative_{horizon}d",
                f"_market_relative_{horizon}d",
            ]
        ]

        result = result.merge(
            relative,
            on=[
                "snapshot_date",
                symbol_column,
            ],
            how="left",
        )

    return result


def describe_cell(
    frame: pd.DataFrame,
    momentum_bins: np.ndarray,
    si_bins: np.ndarray,
    momentum_decile: int,
    si_decile: int,
    target_name: str,
) -> dict[str, Any]:
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


def build_cell_context(
    frame: pd.DataFrame,
    *,
    focus_cells: tuple[tuple[int, int], ...],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    momentum_bins = cross_sectional_deciles(
        frame,
        frame["momentum"],
    )

    si_bins = cross_sectional_deciles(
        frame,
        frame["si_change"],
    )

    rows: list[dict[str, Any]] = []

    for target_name in TARGET_NAMES:
        for momentum_decile, si_decile in focus_cells:
            rows.append(
                describe_cell(
                    frame,
                    momentum_bins,
                    si_bins,
                    momentum_decile,
                    si_decile,
                    target_name,
                )
            )

    return pd.DataFrame(
        rows
    )


def build_sector_breakdown(
    frame: pd.DataFrame,
    *,
    focus_cells: tuple[tuple[int, int], ...],
) -> pd.DataFrame:
    if "sector" not in frame.columns:
        return pd.DataFrame()

    momentum_bins = cross_sectional_deciles(
        frame,
        frame["momentum"],
    )

    si_bins = cross_sectional_deciles(
        frame,
        frame["si_change"],
    )

    rows: list[dict[str, Any]] = []

    for momentum_decile, si_decile in focus_cells:
        cell_mask = (
            (
                momentum_bins
                == momentum_decile - 1
            )
            & (
                si_bins
                == si_decile - 1
            )
        )

        cell = frame.loc[
            cell_mask
        ].copy()

        if cell.empty:
            continue

        grouped = (
            cell.dropna(
                subset=[
                    "sector"
                ]
            )
            .groupby(
                "sector",
                sort=True,
            )
        )

        for sector, sector_frame in grouped:
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

            for horizon in FORWARD_RETURN_HORIZONS:
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
    *,
    focus_cells: tuple[tuple[int, int], ...],
) -> pd.DataFrame:
    working = frame.copy()

    working["half"] = np.where(
        working[
            "snapshot_date"
        ]
        <= pd.Timestamp(
            "2025-06-30"
        ),
        "H1",
        "H2",
    )

    momentum_bins = cross_sectional_deciles(
        working,
        working["momentum"],
    )

    si_bins = cross_sectional_deciles(
        working,
        working["si_change"],
    )

    rows: list[dict[str, Any]] = []

    for momentum_decile, si_decile in focus_cells:
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
                    working[
                        "half"
                    ]
                    == half
                )
            )

            cell = working.loc[
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


def run_momentum_si_cell_context(
    context,
    *,
    focus_cells: tuple[tuple[int, int], ...] = FOCUS_CELLS,
    horizons: tuple[int, ...] = FORWARD_RETURN_HORIZONS,
    context_columns: tuple[str, ...] = CONTEXT_COLUMNS,
) -> ExperimentResult:
    """
    Run the original Momentum × SI cell-context diagnostic
    inside the diagnostic framework.

    The experiment remains descriptive and uses only the
    pre-specified focus cells. No new cells are searched for.
    """

    sector_map = load_sector_map()

    frame = _prepare_frame(
        context.test,
        sector_map=sector_map,
    )

    frame = add_relative_returns(
        frame
    )

    cell_context = build_cell_context(
        frame,
        focus_cells=focus_cells,
        horizons=horizons,
    )

    sector_breakdown = build_sector_breakdown(
        frame,
        focus_cells=focus_cells,
    )

    half_breakdown = build_half_breakdown(
        frame,
        focus_cells=focus_cells,
    )

    result = ExperimentResult(
        name="momentum_si_cell_context",
        description=(
            "Deskriptiv analys av de förut "
            "identifierade momentum × SI-cellerna "
            "med pris-, risk-, sektor- och "
            "forward-return-kontext."
        ),
    )

    result.add_table(
        "cell_context",
        cell_context,
    )

    result.add_table(
        "sector_breakdown",
        sector_breakdown,
    )

    result.add_table(
        "half_breakdown",
        half_breakdown,
    )

    result.add_metric(
        "test_rows",
        int(
            len(frame)
        ),
    )

    result.add_metric(
        "focus_cell_count",
        int(
            len(focus_cells)
        ),
    )

    result.add_metric(
        "sector_map_size",
        int(
            len(sector_map)
        ),
    )

    result.add_metric(
        "cell_result_rows",
        int(
            len(cell_context)
        ),
    )

    result.add_metric(
        "sector_result_rows",
        int(
            len(sector_breakdown)
        ),
    )

    result.add_metric(
        "half_result_rows",
        int(
            len(half_breakdown)
        ),
    )

    result.add_metadata(
        "date_cutoff",
        str(
            DATE_CUTOFF.date()
        ),
    )

    result.add_metadata(
        "targets",
        list(
            TARGET_NAMES
        ),
    )

    result.add_metadata(
        "focus_cells",
        [
            {
                "momentum_decile": momentum,
                "si_decile": si,
            }
            for momentum, si in focus_cells
        ],
    )

    result.add_metadata(
        "min_cell_n",
        MIN_CELL_N,
    )

    result.add_metadata(
        "forward_return_horizons",
        list(
            horizons
        ),
    )

    result.add_metadata(
        "context_columns",
        list(
            context_columns
        ),
    )

    result.add_metadata(
        "decile_method",
        (
            "Cross-sectional per snapshot_date "
            "using rank(method='first', pct=True)."
        ),
    )

    result.add_metadata(
        "cell_selection",
        (
            "Fixed coordinates from the existing "
            "temporal stability analysis; no new "
            "cell search."
        ),
    )

    result.add_metadata(
        "relative_return_method",
        (
            "Stock return minus same-date sector "
            "or market mean, excluding the stock itself."
        ),
    )

    return result
