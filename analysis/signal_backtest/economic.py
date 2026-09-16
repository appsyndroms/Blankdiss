"""Ekonomiskt backtest av Blankdiss OOS-signaler."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis.signal_backtest.config import (
    ECONOMIC_BACKTEST_FRACTIONS,
    ECONOMIC_REBALANCE_DAYS,
    TRANSACTION_COST_BPS,
)


def _clean_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "snapshot_date",
        "security_key",
        "target_return",
        "probability",
    }

    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(
            "Ekonomiskt backtest saknar kolumner: "
            + ", ".join(sorted(missing))
        )

    clean = predictions.copy()

    clean["snapshot_date"] = pd.to_datetime(
        clean["snapshot_date"],
        errors="coerce",
    )

    clean["target_return"] = pd.to_numeric(
        clean["target_return"],
        errors="coerce",
    )

    clean["probability"] = pd.to_numeric(
        clean["probability"],
        errors="coerce",
    )

    clean = clean.dropna(
        subset=[
            "snapshot_date",
            "security_key",
            "target_return",
            "probability",
        ]
    )

    return clean.sort_values(
        [
            "snapshot_date",
            "probability",
            "security_key",
        ],
        ascending=[
            True,
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def _period_return(
    selected: pd.DataFrame,
    direction: str,
    transaction_cost: float,
) -> dict[str, Any]:
    if selected.empty:
        return {
            "rows": 0,
            "gross_return": None,
            "net_return": None,
        }

    raw = selected[
        "target_return"
    ].to_numpy(
        dtype=float
    )

    if direction == "short":
        raw = -raw

    gross = float(
        np.mean(raw)
    )

    net = (
        gross
        - transaction_cost
    )

    return {
        "rows": int(
            len(selected)
        ),
        "gross_return": gross,
        "net_return": float(net),
    }


def _compound(
    returns: list[float],
) -> float:
    if not returns:
        return 0.0

    return float(
        np.prod(
            1.0
            + np.asarray(
                returns
            )
        )
        - 1.0
    )


def _max_drawdown(
    returns: list[float],
) -> float:
    if not returns:
        return 0.0

    equity = np.cumprod(
        1.0
        + np.asarray(
            returns
        )
    )

    peaks = np.maximum.accumulate(
        equity
    )

    drawdowns = (
        equity / peaks
        - 1.0
    )

    return float(
        drawdowns.min()
    )


def _build_strategy(
    predictions: pd.DataFrame,
    fraction: float,
    direction: str,
) -> dict[str, Any]:
    dates = sorted(
        predictions[
            "snapshot_date"
        ]
        .dt.normalize()
        .unique()
    )

    # Använd 5-dagars target utan
    # överlappande positioner:
    # välj en ny portfölj var femte
    # observationsdag.
    rebalance_dates = dates[
        ::ECONOMIC_REBALANCE_DAYS
    ]

    transaction_cost = (
        TRANSACTION_COST_BPS
        / 10_000.0
    )

    period_returns: list[
        float
    ] = []

    gross_returns: list[
        float
    ] = []

    benchmark_returns: list[
        float
    ] = []

    periods: list[
        dict[str, Any]
    ] = []

    for date in rebalance_dates:
        day = predictions.loc[
            predictions[
                "snapshot_date"
            ].dt.normalize()
            == date
        ].copy()

        if day.empty:
            continue

        day = day.sort_values(
            [
                "probability",
                "security_key",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )

        count = max(
            1,
            int(
                np.ceil(
                    len(day)
                    * fraction
                )
            ),
        )

        selected = day.iloc[
            :count
        ]

        result = _period_return(
            selected,
            direction,
            transaction_cost,
        )

        if result[
            "net_return"
        ] is None:
            continue

        universe = day[
            "target_return"
        ].to_numpy(
            dtype=float
        )

        if direction == "short":
            universe = -universe

        benchmark = float(
            np.mean(
                universe
            )
        )

        period_returns.append(
            result[
                "net_return"
            ]
        )

        gross_returns.append(
            result[
                "gross_return"
            ]
        )

        benchmark_returns.append(
            benchmark
        )

        periods.append(
            {
                "date": str(
                    pd.Timestamp(
                        date
                    ).date()
                ),
                "rows": result[
                    "rows"
                ],
                "gross_return": result[
                    "gross_return"
                ],
                "net_return": result[
                    "net_return"
                ],
                "benchmark_return": benchmark,
            }
        )

    benchmark_compounded = _compound(
        benchmark_returns
    )

    net_compounded = _compound(
        period_returns
    )

    return {
        "fraction": float(
            fraction
        ),
        "percentage": float(
            fraction * 100
        ),
        "direction": direction,
        "rebalance_days": int(
            ECONOMIC_REBALANCE_DAYS
        ),
        "transaction_cost_bps": float(
            TRANSACTION_COST_BPS
        ),
        "periods": int(
            len(period_returns)
        ),
        "trades": int(
            sum(
                period["rows"]
                for period in periods
            )
        ),
        "gross_compounded_return": (
            _compound(
                gross_returns
            )
        ),
        "net_compounded_return": (
            net_compounded
        ),
        "benchmark_compounded_return": (
            benchmark_compounded
        ),
        "excess_return_vs_benchmark": (
            float(
                net_compounded
                - benchmark_compounded
            )
        ),
        "mean_period_return": (
            float(
                np.mean(
                    period_returns
                )
            )
            if period_returns
            else None
        ),
        "median_period_return": (
            float(
                np.median(
                    period_returns
                )
            )
            if period_returns
            else None
        ),
        "max_drawdown": _max_drawdown(
            period_returns
        ),
        "periods_detail": periods,
    }


def run_economic_backtest(
    predictions: pd.DataFrame,
    target_name: str,
) -> dict[str, Any]:
    """
    Kör ett OOS-ekonomiskt backtest
    utan överlappande 5-dagarsperioder.
    """
    clean = _clean_predictions(
        predictions
    )

    if clean.empty:
        raise ValueError(
            "Ekonomiskt backtest fick "
            "inga giltiga OOS-prediktioner."
        )

    direction = (
        "short"
        if target_name.startswith(
            "down_"
        )
        else "long"
    )

    strategies = [
        _build_strategy(
            clean,
            fraction,
            direction,
        )
        for fraction
        in ECONOMIC_BACKTEST_FRACTIONS
    ]

    return {
        "target": target_name,
        "direction": direction,
        "selection": (
            "highest predicted probability"
        ),
        "no_overlapping_periods": True,
        "strategies": strategies,
    }
