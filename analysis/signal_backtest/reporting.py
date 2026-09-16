"""Rapportering för Blankdiss signal-backtest."""
from __future__ import annotations

from typing import Any


def print_economic(
    economic: dict[str, Any],
) -> None:
    """Skriv ut ekonomiskt backtest med fokuserad rapportering."""
    print()
    print("Economic backtest:")

    print(
        "  Direction: "
        f"{economic['direction']}"
    )

    print(
        "  Primary: "
        f"top 1%, "
        f"{economic['primary_transaction_cost_bps']:.1f} bps, "
        f"rebalance="
        f"{economic['primary_rebalance_days']}d"
    )

    strategies = economic["strategies"]

    primary = next(
        (
            strategy
            for strategy in strategies
            if abs(
                strategy["fraction"] - 0.01
            ) < 1e-12
        ),
        None,
    )

    if primary is not None:
        print()
        print("  Top 1% economic result:")

        print(
            "    Net: "
            f"{primary['net_compounded_return']:.2%}"
        )

        print(
            "    Benchmark: "
            f"{primary['benchmark_compounded_return']:.2%}"
        )

        print(
            "    Excess: "
            f"{primary['excess_return_vs_benchmark']:.2%}"
        )

        print(
            "    Max drawdown: "
            f"{primary['max_drawdown']:.2%}"
        )

    concentration = economic.get(
        "concentration_sensitivity"
    )

    if concentration:
        print()
        print(
            "  Concentration sensitivity:"
        )

        baseline = concentration["baseline"]

        print(
            "    Baseline net: "
            f"{baseline['net_compounded_return']:.2%}"
        )

        print(
            "    Leave-one-out:"
        )

        for test in concentration["tests"]:
            print(
                f"      {test['label']}: "
                f"net="
                f"{test['net_compounded_return']:.2%}, "
                f"delta="
                f"{test['delta_net_return']:.2%}"
            )

    random_sensitivity = economic.get(
        "random_security_sensitivity"
    )

    if random_sensitivity:
        print()
        print(
            "  Random security removal:"
        )

        baseline = random_sensitivity["baseline"]

        print(
            "    Baseline net: "
            f"{baseline['net_compounded_return']:.2%}"
        )

        print(
            "    Baseline excess: "
            f"{baseline['excess_return_vs_benchmark']:.2%}"
        )

        print(
            "    Universe securities: "
            f"{random_sensitivity['universe_securities']}"
        )

        for level in random_sensitivity["levels"]:
            summary = level["summary"]

            print(
                f"    Remove "
                f"{level['removal_percentage']:g}%: "
                f"positive excess="
                f"{summary['positive_excess_runs']}/"
                f"{summary['runs']} "
                f"({summary['positive_excess_share']:.0%}), "
                f"median net="
                f"{summary['median_net_return']:.2%}, "
                f"range="
                f"{summary['min_net_return']:.2%}"
                " .. "
                f"{summary['max_net_return']:.2%}, "
                f"median excess="
                f"{summary['median_excess_return']:.2%}, "
                f"median DD="
                f"{summary['median_max_drawdown']:.2%}"
            )


def print_experiment(
    result: dict[str, Any],
) -> None:
    """Skriv ut ett experiment med endast relevant OOS-information."""
    print()
    print("--------------------------------")

    print(
        "Feature set: "
        f"{result['feature_set']}"
    )

    print(
        "Target: "
        f"{result['target']}"
    )

    for window in result["windows"]:
        print()
        print(
            "Window: "
            f"{window['window']['train_end']}"
            " -> "
            f"{window['window']['validation_end']}"
            " -> "
            f"{window['window']['test_end']}"
        )

        print(
            "  Test AUC: "
            f"{window['test_auc']:.4f}"
        )

    print_economic(
        result["economic"]
    )


def print_header() -> None:
    """Skriv huvudrubriken."""
    print("================================")
    print("BLANKDISS SIGNAL BACKTEST")
    print("================================")

    print(
        "Focus: DOWN robustness and "
        "security concentration"
    )
