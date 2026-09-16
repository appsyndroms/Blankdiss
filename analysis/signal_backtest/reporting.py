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


def print_diagnostics(
    diagnostics: dict[str, Any],
) -> None:
    """Skriv ut diagnostik för signalen."""
    print()
    print("Diagnostics:")

    calibration = diagnostics[
        "calibration"
    ]

    print(
        "  Brier score: "
        f"{calibration['brier_score']:.6f}"
    )

    print(
        "  Mean probability: "
        f"{calibration['mean_probability']:.4f}"
    )

    print(
        "  Mean actual: "
        f"{calibration['mean_actual']:.4f}"
    )

    print("  Calibration bins:")

    for bucket in calibration["bins"]:
        print(
            f"    Bin {bucket['bin']}: "
            f"rows={bucket['rows']}, "
            f"prob="
            f"{bucket['mean_probability']:.4f}, "
            f"actual="
            f"{bucket['event_rate']:.4f}"
        )

    print()
    print("  Monthly top 1%:")

    for month in diagnostics[
        "monthly_top_1pct"
    ]:
        lift = month["top_lift"]

        lift_text = (
            f"{lift:.2f}x"
            if lift is not None
            else "n/a"
        )

        print(
            f"    {month['month']}: "
            f"rows={month['rows']}, "
            f"top_rows={month['top_rows']}, "
            f"events={month['top_events']}, "
            f"event_rate="
            f"{month['top_event_rate']:.4f}, "
            f"lift={lift_text}, "
            f"mean="
            f"{month['top_mean_return']:.4%}"
        )

    security = diagnostics[
        "security_top_1pct"
    ]

    print()
    print("  Security concentration top 1%:")

    print(
        "    Top rows: "
        f"{security['top_rows']}"
    )

    print(
        "    Unique securities: "
        f"{security['unique_securities']}"
    )

    largest = (
        security["largest_security_share"]
    )

    top_5 = (
        security["top_5_security_share"]
    )

    top_10 = (
        security["top_10_security_share"]
    )

    print(
        "    Largest security share: "
        f"{largest:.2%}"
    )

    print(
        "    Top 5 security share: "
        f"{top_5:.2%}"
    )

    print(
        "    Top 10 security share: "
        f"{top_10:.2%}"
    )

    print(
        "    Largest securities:"
    )

    for security_row in security[
        "top_securities"
    ]:
        print(
            f"      "
            f"{security_row['security_key']}: "
            f"rows={security_row['rows']}, "
            f"event_rate="
            f"{security_row['event_rate']:.4f}, "
            f"mean_probability="
            f"{security_row['mean_probability']:.4f}, "
            f"mean_return="
            f"{security_row['mean_return']:.4%}"
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

    print_diagnostics(
        result["diagnostics"]
    )


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
