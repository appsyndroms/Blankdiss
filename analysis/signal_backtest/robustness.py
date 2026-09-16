"""Robusthetstester för Blankdiss ekonomiska backtest."""
from __future__ import annotations
from typing import Any, Callable
import numpy as np
import pandas as pd
DEFAULT_REMOVAL_FRACTIONS = (
    0.10,
    0.20,
    0.30,
)
DEFAULT_RANDOM_SEEDS = (
    11,
    22,
    33,
    44,
    55,
)
def build_random_security_sensitivity(
    predictions: pd.DataFrame,
    strategy_builder: Callable[..., dict[str, Any]],
    fraction: float,
    direction: str,
    transaction_cost_bps: float,
    rebalance_days: int,
    removal_fractions: tuple[float, ...] = (
        DEFAULT_REMOVAL_FRACTIONS
    ),
    seeds: tuple[int, ...] = (
        DEFAULT_RANDOM_SEEDS
    ),
) -> dict[str, Any]:
    """
    Testa om strategins ekonomiska resultat överlever
    slumpmässigt borttagande av värdepapper.
    Endast värdepappersuniversumet förändras.
    """
    required = {
        "snapshot_date",
        "security_key",
        "target_return",
        "score",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            "Random robustness saknar kolumner: "
            + ", ".join(sorted(missing))
        )
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError(
            "fraction måste vara > 0 och <= 1."
        )
    if not removal_fractions:
        raise ValueError(
            "removal_fractions får inte vara tom."
        )
    if not seeds:
        raise ValueError(
            "seeds får inte vara tom."
        )
    clean = predictions.copy()
    clean["security_key"] = (
        clean["security_key"]
        .astype(str)
    )
    securities = sorted(
        clean["security_key"]
        .unique()
        .tolist()
    )
    if len(securities) < 2:
        raise ValueError(
            "Random robustness kräver minst "
            "två värdepapper."
        )
    baseline = strategy_builder(
        clean,
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
    levels: list[dict[str, Any]] = []
    for removal_fraction in removal_fractions:
        if not 0.0 < removal_fraction < 1.0:
            raise ValueError(
                "Varje removal_fraction måste vara "
                "> 0 och < 1."
            )
        removal_count = max(
            1,
            int(
                np.floor(
                    len(securities)
                    * removal_fraction
                )
            ),
        )
        removal_count = min(
            removal_count,
            len(securities) - 1,
        )
        runs: list[dict[str, Any]] = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            removed = set(
                rng.choice(
                    securities,
                    size=removal_count,
                    replace=False,
                ).tolist()
            )
            filtered = clean.loc[
                ~clean["security_key"]
                .isin(removed)
            ].copy()
            result = strategy_builder(
                filtered,
                fraction,
                direction,
                transaction_cost_bps,
                rebalance_days,
            )
            net_return = float(
                result["net_compounded_return"]
            )
            max_drawdown = float(
                result["max_drawdown"]
            )
            excess_return = (
                net_return
                - baseline_benchmark
            )
            runs.append(
                {
                    "seed": int(seed),
                    "removed_count": int(
                        removal_count
                    ),
                    "remaining_securities": int(
                        len(securities)
                        - removal_count
                    ),
                    "net_compounded_return": (
                        net_return
                    ),
                    "excess_return_vs_baseline_benchmark": (
                        excess_return
                    ),
                    "delta_net_return": (
                        net_return
                        - baseline_net
                    ),
                    "delta_excess_return": (
                        excess_return
                        - baseline_excess
                    ),
                    "max_drawdown": max_drawdown,
                    "delta_max_drawdown": (
                        max_drawdown
                        - baseline_max_drawdown
                    ),
                }
            )
        net_returns = np.asarray(
            [
                run["net_compounded_return"]
                for run in runs
            ],
            dtype=float,
        )
        excess_returns = np.asarray(
            [
                run[
                    "excess_return_vs_baseline_benchmark"
                ]
                for run in runs
            ],
            dtype=float,
        )
        drawdowns = np.asarray(
            [
                run["max_drawdown"]
                for run in runs
            ],
            dtype=float,
        )
        positive_excess_runs = int(
            np.sum(excess_returns > 0.0)
        )
        levels.append(
            {
                "removal_fraction": float(
                    removal_fraction
                ),
                "removal_percentage": float(
                    removal_fraction * 100
                ),
                "removed_count": int(
                    removal_count
                ),
                "runs": runs,
                "summary": {
                    "runs": len(runs),
                    "positive_excess_runs": (
                        positive_excess_runs
                    ),
                    "positive_excess_share": (
                        positive_excess_runs
                        / len(runs)
                    ),
                    "median_net_return": float(
                        np.median(net_returns)
                    ),
                    "min_net_return": float(
                        np.min(net_returns)
                    ),
                    "max_net_return": float(
                        np.max(net_returns)
                    ),
                    "median_excess_return": float(
                        np.median(excess_returns)
                    ),
                    "min_excess_return": float(
                        np.min(excess_returns)
                    ),
                    "max_excess_return": float(
                        np.max(excess_returns)
                    ),
                    "median_max_drawdown": float(
                        np.median(drawdowns)
                    ),
                },
            }
        )
    return {
        "method": "random_security_removal",
        "fraction": float(fraction),
        "direction": direction,
        "transaction_cost_bps": float(
            transaction_cost_bps
        ),
        "rebalance_days": int(
            rebalance_days
        ),
        "removal_fractions": [
            float(value)
            for value in removal_fractions
        ],
        "seeds": [
            int(value)
            for value in seeds
        ],
        "universe_securities": int(
            len(securities)
        ),
        "baseline": {
            "net_compounded_return": baseline_net,
            "benchmark_compounded_return": (
                baseline_benchmark
            ),
            "excess_return_vs_benchmark": (
                baseline_excess
            ),
            "max_drawdown": baseline_max_drawdown,
        },
        "levels": levels,
    }
