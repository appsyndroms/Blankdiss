"""Rapportering av Blankdiss ML- och ekonomiska resultat."""
from __future__ import annotations

from typing import Any


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"

    return f"{value * 100:.2f}%"


def _metric(
    value: float | None,
    decimals: int = 4,
) -> str:
    if value is None:
        return "n/a"

    return f"{value:.{decimals}f}"


def _find_primary_strategy(
    economic: dict[str, Any],
) -> dict[str, Any] | None:
    for strategy in economic.get(
        "strategies",
        [],
    ):
        if abs(
            float(
                strategy.get(
                    "fraction",
                    0.0,
                )
            )
            - 0.01
        ) < 1e-12:
            return strategy

    return None


def _format_range(
    minimum: float | None,
    maximum: float | None,
) -> str:
    if minimum is None or maximum is None:
        return "n/a"

    return (
        f"{minimum * 100:.2f}%"
        f"–"
        f"{maximum * 100:.2f}%"
    )


def print_ml_result(
    result: dict[str, Any],
) -> None:
    """Skriv ut ett ML-resultat."""
    print()
    print("=" * 72)
    print(
        f"Feature set: "
        f"{result.get('feature_set')}"
    )
    print(
        f"Target: "
        f"{result.get('target')}"
    )
    print(
        f"Model: "
        f"{result.get('model')}"
    )
    print(
        f"Task: "
        f"{result.get('target_task', result.get('task'))}"
    )

    validation_score = result.get(
        "validation_score"
    )

    if validation_score is not None:
        print(
            "Validation score: "
            f"{_metric(validation_score)}"
        )

    test = result.get(
        "test",
        {},
    )

    task = result.get(
        "target_task",
        result.get("task"),
    )

    if task == "classification":
        print(
            "Test AUC: "
            f"{_metric(test.get('roc_auc'))}"
        )
        print(
            "Test Brier: "
            f"{_metric(test.get('brier'))}"
        )

    elif task == "regression":
        print(
            "Test MAE: "
            f"{_metric(test.get('mae'))}"
        )
        print(
            "Test RMSE: "
            f"{_metric(test.get('rmse'))}"
        )
        print(
            "Test Spearman: "
            f"{_metric(test.get('spearman'))}"
        )


def _print_random_security_robustness(
    random_security: dict[str, Any],
) -> None:
    """Skriv ut en kompakt sammanfattning av random robustness."""
    if not random_security:
        return

    baseline = random_security.get(
        "baseline",
        {},
    )

    print(
        "Random security robustness:"
    )

    print(
        "  Baseline excess: "
        f"{_pct(baseline.get('excess_return_vs_benchmark'))}, "
        "max DD: "
        f"{_pct(baseline.get('max_drawdown'))}"
    )

    for level in random_security.get(
        "levels",
        [],
    ):
        summary = level.get(
            "summary",
            {},
        )

        removal_percentage = (
            level.get(
                "removal_percentage"
            )
        )

        median_excess = summary.get(
            "median_excess_return"
        )

        minimum_excess = summary.get(
            "min_excess_return"
        )

        maximum_excess = summary.get(
            "max_excess_return"
        )

        positive_runs = summary.get(
            "positive_excess_runs"
        )

        runs = summary.get(
            "runs"
        )

        print(
            f"  {removal_percentage:.0f}% removal: "
            "median excess "
            f"{_pct(median_excess)}, "
            "range "
            f"{_format_range(minimum_excess, maximum_excess)}, "
            f"positive {positive_runs}/{runs}"
        )


def print_economic_result(
    result: dict[str, Any],
) -> None:
    """Skriv ut ekonomiskt resultat för ett experiment."""
    print()
    print("=" * 72)

    print(
        f"Feature set : "
        f"{result.get('feature_set')}"
    )

    print(
        f"Target      : "
        f"{result.get('target')}"
    )

    print(
        f"Task        : "
        f"{result.get('task')}"
    )

    print(
        f"Model       : "
        f"{result.get('model')}"
    )

    print(
        f"Direction   : "
        f"{result.get('direction')}"
    )

    print(
        f"OOS         : "
        f"{result.get('oos_start')} "
        f"-> "
        f"{result.get('oos_end')}"
    )

    primary = _find_primary_strategy(
        result
    )

    if primary is None:
        print(
            "Ingen top-1%-strategi hittades."
        )
        return

    print()
    print(
        "PRIMARY — top 1%, "
        "10 bps, rebalance 5"
    )

    print(
        "Net return  : "
        f"{_pct(primary.get('net_compounded_return'))}"
    )

    print(
        "Benchmark   : "
        f"{_pct(primary.get('benchmark_compounded_return'))}"
    )

    print(
        "Excess      : "
        f"{_pct(primary.get('excess_return_vs_benchmark'))}"
    )

    print(
        "Max DD      : "
        f"{_pct(primary.get('max_drawdown'))}"
    )

    print(
        "Turnover    : "
        f"{_metric(primary.get('mean_turnover'))}"
    )

    print(
        "Invested    : "
        f"{_pct(primary.get('mean_invested_weight'))}"
    )

    print()

    print(
        "Concentration:"
    )

    concentration = result.get(
        "concentration_sensitivity",
        {},
    )

    if concentration:
        print(
            f"  {concentration}"
        )

    print()

    random_security = result.get(
        "random_security_sensitivity",
        {},
    )

    _print_random_security_robustness(
        random_security
    )


def print_economic_results(
    results: list[dict[str, Any]],
) -> None:
    """Skriv ut alla ekonomiska experiment."""
    for result in results:
        print_economic_result(
            result
        )
