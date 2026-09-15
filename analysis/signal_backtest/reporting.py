"""Rapportering för Blankdiss signal-backtest."""
from __future__ import annotations
from typing import Any
def print_bucket(
    bucket: dict[str, Any],
) -> None:
    """Skriv ut ett top-N-urval."""
    percentage = bucket["percentage"]
    print(
        f"  Top {percentage:g}%: "
        f"rows={bucket['rows']}, "
        f"event_rate="
        f"{bucket['event_rate']:.4f}, "
        f"lift="
        f"{bucket['lift_ratio']:.2f}x, "
        f"mean="
        f"{bucket['mean_return']:.4%}, "
        f"median="
        f"{bucket['median_return']:.4%}, "
        f"p10="
        f"{bucket['p10_return']:.4%}, "
        f"p90="
        f"{bucket['p90_return']:.4%}"
    )
def print_experiment(
    result: dict[str, Any],
) -> None:
    """Skriv ut resultatet för ett experiment."""
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
    print(
        "Features: "
        f"{result['dataset_summary']['features']}"
    )
    print(
        "Rows: "
        f"{result['dataset_summary']['rows']:,}"
    )
    print(
        "Date range: "
        f"{result['dataset_summary']['date_start']}"
        " -> "
        f"{result['dataset_summary']['date_end']}"
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
            "Selected model: "
            f"{window['model']}"
        )
        print(
            "Validation AUC: "
            f"{window['validation_auc']:.4f}"
        )
        print(
            "Test AUC: "
            f"{window['test_auc']:.4f}"
        )
    for yearly in result["yearly"]:
        print()
        print(
            f"Year: {yearly['year']}"
        )
        print(
            "Rows: "
            f"{yearly['rows']:,}"
        )
        print(
            "Baseline event rate: "
            f"{yearly['baseline_event_rate']:.4f}"
        )
        print(
            "Baseline mean return: "
            f"{yearly['baseline_mean_return']:.4%}"
        )
        print("Prediction buckets:")
        for bucket in yearly["buckets"]:
            print_bucket(bucket)
def print_header() -> None:
    """Skriv huvudrubriken."""
    print("================================")
    print("BLANKDISS SIGNAL BACKTEST")
    print("================================")
    print(
        "Top fractions: "
        "0.1%, 0.5%, 1%, 2%, 5%, 10%, 20%"
    )
    print(
        "Feature sets: "
        "volatility-only, all-price"
    )
    print(
        "Targets: "
        "up_5pct_5d, down_5pct_5d"
    )
