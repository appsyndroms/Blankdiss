"""Strategikörning för Blankdiss ekonomiska backtest."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis.signal_backtest.config import (
    ECONOMIC_MAX_POSITION_WEIGHT,
    ECONOMIC_REBALANCE_DAYS,
)
from analysis.signal_backtest.portfolio import (
    compound,
    invested_weight,
    max_drawdown,
    portfolio_return,
    portfolio_weights,
    turnover,
)


def build_strategy(
    predictions: pd.DataFrame,
    fraction: float,
    direction: str,
    transaction_cost_bps: float,
    rebalance_days: int | None = None,
) -> dict[str, Any]:
    """
    Bygg en ekonomisk strategi.

    rebalance_days anger hur många observationsdagar som ska
    hoppas mellan nya portföljurval.

    Targeten är fortfarande den befintliga target_return,
    normalt en 5-dagars forward return.

    Därför ska rebalance_days inte tolkas som holding period.
    """
    if rebalance_days is None:
        rebalance_days = ECONOMIC_REBALANCE_DAYS

    if rebalance_days < 1:
        raise ValueError(
            "rebalance_days måste vara >= 1."
        )

    dates = sorted(
        predictions[
            "snapshot_date"
        ]
        .dt.normalize()
        .unique()
    )

    rebalance_dates = dates[
        ::rebalance_days
    ]

    transaction_cost_rate = (
        transaction_cost_bps
        / 10_000.0
    )

    period_returns: list[float] = []
    gross_returns: list[float] = []
    benchmark_returns: list[float] = []
    turnover_values: list[float] = []
    invested_weights: list[float] = []

    periods: list[
        dict[str, Any]
    ] = []

    previous_weights: dict[str, float] = {}

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
        ].copy()

        current_weights = portfolio_weights(
            selected,
            ECONOMIC_MAX_POSITION_WEIGHT,
        )

        current_turnover = turnover(
            previous_weights,
            current_weights,
        )

        gross = portfolio_return(
            selected,
            direction,
            ECONOMIC_MAX_POSITION_WEIGHT,
        )

        transaction_cost = (
            current_turnover
            * transaction_cost_rate
        )

        net = (
            gross
            - transaction_cost
        )

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

        current_invested_weight = invested_weight(
            current_weights
        )

        period_returns.append(
            net
        )

        gross_returns.append(
            gross
        )

        benchmark_returns.append(
            benchmark
        )

        turnover_values.append(
            current_turnover
        )

        invested_weights.append(
            current_invested_weight
        )

        periods.append(
            {
                "date": str(
                    pd.Timestamp(
                        date
                    ).date()
                ),
                "rows": int(
                    len(selected)
                ),
                "invested_weight": (
                    current_invested_weight
                ),
                "cash_weight": (
                    1.0
                    - current_invested_weight
                ),
                "gross_return": gross,
                "transaction_cost": (
                    transaction_cost
                ),
                "turnover": current_turnover,
                "net_return": net,
                "benchmark_return": (
                    benchmark
                ),
            }
        )

        previous_weights = current_weights

    benchmark_compounded = compound(
        benchmark_returns
    )

    gross_compounded = compound(
        gross_returns
    )

    net_compounded = compound(
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
            rebalance_days
        ),
        "transaction_cost_bps": float(
            transaction_cost_bps
        ),
        "max_position_weight": float(
            ECONOMIC_MAX_POSITION_WEIGHT
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
        "mean_turnover": (
            float(
                np.mean(
                    turnover_values
                )
            )
            if turnover_values
            else 0.0
        ),
        "median_turnover": (
            float(
                np.median(
                    turnover_values
                )
            )
            if turnover_values
            else 0.0
        ),
        "mean_invested_weight": (
            float(
                np.mean(
                    invested_weights
                )
            )
            if invested_weights
            else 0.0
        ),
        "median_invested_weight": (
            float(
                np.median(
                    invested_weights
                )
            )
            if invested_weights
            else 0.0
        ),
        "gross_compounded_return": (
            gross_compounded
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
        "max_drawdown": max_drawdown(
            period_returns
        ),
        "periods_detail": periods,
    }


def build_yearly_results(
    predictions: pd.DataFrame,
    fraction: float,
    direction: str,
    transaction_cost_bps: float,
    rebalance_days: int | None = None,
) -> list[dict[str, Any]]:
    """
    Kör samma ekonomiska strategi separat per kalenderår.

    Detta används diagnostiskt för att skilja 2025 från 2026
    och undvika att ett starkt år döljer ett svagt år.
    """
    yearly_results: list[
        dict[str, Any]
    ] = []

    years = sorted(
        predictions[
            "snapshot_date"
        ]
        .dt.year
        .unique()
        .tolist()
    )

    for year in years:
        year_predictions = predictions.loc[
            predictions[
                "snapshot_date"
            ].dt.year
            == year
        ].copy()

        if year_predictions.empty:
            continue

        result = build_strategy(
            year_predictions,
            fraction,
            direction,
            transaction_cost_bps,
            rebalance_days,
        )

        result["year"] = int(
            year
        )

        yearly_results.append(
            result
        )

    return yearly_results
