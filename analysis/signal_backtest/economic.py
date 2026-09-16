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


MAX_POSITION_WEIGHT = 0.05


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


def _portfolio_weights(
    selected: pd.DataFrame,
) -> dict[str, float]:
    """Skapa lika vikter med ett max på 5 % per värdepapper."""
    if selected.empty:
        return {}

    securities = (
        selected["security_key"]
        .astype(str)
        .tolist()
    )

    count = len(securities)

    if count * MAX_POSITION_WEIGHT < 1.0:
        raise ValueError(
            "För få värdepapper för att bygga en fullt investerad "
            f"portfölj med maxvikt {MAX_POSITION_WEIGHT:.1%}: {count}."
        )

    weight = 1.0 / count

    if weight <= MAX_POSITION_WEIGHT:
        return {
            security: weight
            for security in securities
        }

    weights = {
        security: MAX_POSITION_WEIGHT
        for security in securities
    }

    remaining = 1.0 - sum(weights.values())

    uncapped = [
        security
        for security in securities
        if weights[security] < MAX_POSITION_WEIGHT
    ]

    while remaining > 1e-12 and uncapped:
        add = remaining / len(uncapped)
        next_uncapped = []

        for security in uncapped:
            capacity = (
                MAX_POSITION_WEIGHT
                - weights[security]
            )

            increase = min(
                capacity,
                add,
            )

            weights[security] += increase
            remaining -= increase

            if (
                weights[security]
                < MAX_POSITION_WEIGHT - 1e-12
            ):
                next_uncapped.append(
                    security
                )

        uncapped = next_uncapped

    return weights


def _portfolio_return(
    selected: pd.DataFrame,
    direction: str,
) -> float:
    """Beräkna lika/viktad portföljavkastning före kostnader."""
    if selected.empty:
        raise ValueError(
            "Kan inte beräkna portföljavkastning utan innehav."
        )

    returns = selected[
        "target_return"
    ].to_numpy(
        dtype=float
    )

    if direction == "short":
        returns = -returns

    weights = _portfolio_weights(
        selected
    )

    weight_array = np.asarray(
        [
            weights[str(security)]
            for security in selected[
                "security_key"
            ]
        ],
        dtype=float,
    )

    return float(
        np.sum(
            weight_array
            * returns
        )
    )


def _turnover(
    previous_weights: dict[str, float],
    current_weights: dict[str, float],
) -> float:
    """Total portföljomsättning som halv-L1-avstånd mellan vikter."""
    securities = (
        set(previous_weights)
        | set(current_weights)
    )

    return float(
        0.5
        * sum(
            abs(
                current_weights.get(
                    security,
                    0.0,
                )
                - previous_weights.get(
                    security,
                    0.0,
                )
            )
            for security in securities
        )
    )


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

    # Targeten är fem handelsdagar. Vi startar en ny portfölj
    # var femte observationsdag så att targetperioderna
    # inte överlappar.
    rebalance_dates = dates[
        ::ECONOMIC_REBALANCE_DAYS
    ]

    transaction_cost_rate = (
        TRANSACTION_COST_BPS
        / 10_000.0
    )

    period_returns: list[float] = []
    gross_returns: list[float] = []
    benchmark_returns: list[float] = []
    turnover_values: list[float] = []

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

        current_weights = _portfolio_weights(
            selected
        )

        turnover = _turnover(
            previous_weights,
            current_weights,
        )

        gross = _portfolio_return(
            selected,
            direction,
        )

        transaction_cost = (
            turnover
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
            turnover
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
                "gross_return": gross,
                "transaction_cost": transaction_cost,
                "turnover": turnover,
                "net_return": net,
                "benchmark_return": benchmark,
            }
        )

        previous_weights = current_weights

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
        "max_position_weight": float(
            MAX_POSITION_WEIGHT
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
    Kör ett OOS-ekonomiskt backtest utan överlappande
    5-dagarsperioder.

    Portföljen är lika viktad inom urvalet, med max 5 %
    per värdepapper. Transaktionskostnad beräknas på faktisk
    portföljomsättning.
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
        "portfolio_weighting": (
            "equal_weighted"
        ),
        "max_position_weight": float(
            MAX_POSITION_WEIGHT
        ),
        "turnover_cost_model": (
            "actual_portfolio_turnover"
        ),
        "strategies": strategies,
    }
