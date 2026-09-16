"""Känslighetsanalyser för Blankdiss ekonomiska backtest."""
from __future__ import annotations
from collections import Counter
from typing import Any
import numpy as np
import pandas as pd
from analysis.signal_backtest.config import (
    ECONOMIC_REBALANCE_DAYS_SENSITIVITY,
    ECONOMIC_TRANSACTION_COST_BPS,
)
from analysis.signal_backtest.strategy import build_strategy
def build_cost_sensitivity(
    predictions: pd.DataFrame,
    fraction: float,
    direction: str,
    rebalance_days: int | None = None,
) -> list[dict[str, Any]]:
    """Kör samma strategi vid flera transaktionskostnader."""
    return [
        build_strategy(
            predictions,
            fraction,
            direction,
            transaction_cost_bps,
            rebalance_days,
        )
        for transaction_cost_bps in ECONOMIC_TRANSACTION_COST_BPS
    ]
def build_rebalance_sensitivity(
    predictions: pd.DataFrame,
    fraction: float,
    direction: str,
    transaction_cost_bps: float,
) -> list[dict[str, Any]]:
    """Kör samma strategi vid flera rebalance-intervall."""
    return [
        build_strategy(
            predictions,
            fraction,
            direction,
            transaction_cost_bps,
            rebalance_days,
        )
        for rebalance_days in ECONOMIC_REBALANCE_DAYS_SENSITIVITY
    ]
def selected_security_counts(
    predictions: pd.DataFrame,
    fraction: float,
    rebalance_days: int,
) -> Counter[str]:
    """Räkna hur ofta varje värdepapper väljs."""
    required = {
        "snapshot_date",
        "security_key",
        "score",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            "selected_security_counts saknar kolumner: "
            + ", ".join(sorted(missing))
        )
    dates = sorted(
        predictions["snapshot_date"]
        .dt.normalize()
        .unique()
    )
    rebalance_dates = dates[::rebalance_days]
    counts: Counter[str] = Counter()
    for date in rebalance_dates:
        day = predictions.loc[
            predictions["snapshot_date"]
            .dt.normalize()
            == date
        ].copy()
        if day.empty:
            continue
        day = day.sort_values(
            ["score", "security_key"],
            ascending=[False, True],
            kind="mergesort",
        )
        count = max(
            1,
            int(
                np.ceil(
                    len(day) * fraction
                )
            ),
        )
        selected = day.iloc[:count]
        for security in selected["security_key"].astype(str):
            counts[security] += 1
    return counts
def build_concentration_sensitivity(
    predictions: pd.DataFrame,
    fraction: float,
    direction: str,
    transaction_cost_bps: float,
    rebalance_days: int,
    top_n: int = 10,
) -> dict[str, Any]:
    """Testa hur beroende strategin är av de mest valda värdepappren."""
    baseline = build_strategy(
        predictions,
        fraction,
        direction,
        transaction_cost_bps,
        rebalance_days,
    )
    baseline_net = float(
        baseline["net_compounded_return"]
    )
    baseline_benchmark = float(
        baseline["benchmark_compounded_return"]
    )
    baseline_excess = (
        baseline_net
        - baseline_benchmark
    )
    baseline_max_drawdown = float(
        baseline["max_drawdown"]
    )
    selection_counts = selected_security_counts(
        predictions,
        fraction,
        rebalance_days,
    )
    top_securities = [
        security
        for security, _count in
        selection_counts.most_common(top_n)
    ]
    tests: list[dict[str, Any]] = []
    def run_exclusion_test(
        label: str,
        excluded_securities: set[str],
        selection_count: int,
    ) -> None:
        filtered = predictions.loc[
            ~predictions["security_key"]
            .astype(str)
            .isin(excluded_securities)
        ].copy()
        result = build_strategy(
            filtered,
            fraction,
            direction,
            transaction_cost_bps,
            rebalance_days,
        )
        result_net = float(
            result["net_compounded_return"]
        )
        result_max_drawdown = float(
            result["max_drawdown"]
        )
        result_excess_vs_baseline_benchmark = (
            result_net
            - baseline_benchmark
        )
        tests.append(
            {
                "label": label,
                "excluded_securities": sorted(
                    excluded_securities
                ),
                "selection_count": int(
                    selection_count
                ),
                "net_compounded_return": result_net,
                "delta_net_return": (
                    result_net
                    - baseline_net
                ),
                "excess_return_vs_baseline_benchmark": (
                    result_excess_vs_baseline_benchmark
                ),
                "delta_excess_return": (
                    result_excess_vs_baseline_benchmark
                    - baseline_excess
                ),
                "max_drawdown": result_max_drawdown,
                "delta_max_drawdown": (
                    result_max_drawdown
                    - baseline_max_drawdown
                ),
            }
        )
    for security in top_securities:
        run_exclusion_test(
            label=security,
            excluded_securities={security},
            selection_count=selection_counts[
                security
            ],
        )
    top_n_set = set(top_securities)
    run_exclusion_test(
        label=f"top_{top_n}_excluded",
        excluded_securities=top_n_set,
        selection_count=sum(
            selection_counts[security]
            for security in top_n_set
        ),
    )
    return {
        "fraction": float(fraction),
        "rebalance_days": int(rebalance_days),
        "transaction_cost_bps": float(
            transaction_cost_bps
        ),
        "top_n": int(top_n),
        "baseline": {
            "net_compounded_return": baseline_net,
            "benchmark_compounded_return": (
                baseline_benchmark
            ),
            "excess_return_vs_benchmark": (
                baseline_excess
            ),
            "max_drawdown": baseline_max_drawdown,
            "selection_counts": {
                security: int(count)
                for security, count
                in selection_counts.items()
            },
        },
        "tests": tests,
    }
