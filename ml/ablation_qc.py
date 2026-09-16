"""QC for Blankdiss price-feature ablation experiments."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ml.config import ML_OUTPUT_DIR


INPUT_PATH = ML_OUTPUT_DIR / "latest_run.json"
OUTPUT_PATH = ML_OUTPUT_DIR / "ablation_qc.json"

AUC_SIGNAL_THRESHOLD = 0.55
AUC_STRONG_THRESHOLD = 0.60

TOP_FRACTIONS = (
    0.01,
    0.05,
    0.10,
    0.20,
)


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        value = float(value)

        if np.isfinite(value):
            return value

    return None


def fmt(value: Any, digits: int = 3) -> str:
    value = number(value)

    if value is None:
        return "-"

    return f"{value:.{digits}f}"


def ratio(value: Any) -> str:
    value = number(value)

    if value is None:
        return "-"

    return f"{value:.2f}x"


def pct(value: Any) -> str:
    value = number(value)

    if value is None:
        return "-"

    return f"{value:.1%}"


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None

    return float(np.mean(values))


def safe_median(values: list[float]) -> float | None:
    if not values:
        return None

    return float(np.median(values))


def unique(values: list[Any]) -> list[Any]:
    result = []
    seen = set()

    for value in values:
        marker = json.dumps(
            value,
            sort_keys=True,
            default=str,
        )

        if marker not in seen:
            seen.add(marker)
            result.append(value)

    return result


def result_key(
    result: dict[str, Any],
) -> tuple[str, str]:
    return (
        str(
            result.get(
                "feature_set",
                "unknown",
            )
        ),
        str(
            result.get(
                "target",
                "unknown",
            )
        ),
    )


def extract_auc(
    result: dict[str, Any],
) -> float | None:
    test = result.get("test")

    if isinstance(test, dict):
        value = number(
            test.get("roc_auc")
        )

        if value is not None:
            return value

    return number(
        result.get("test_roc_auc")
    )


def extract_validation_auc(
    result: dict[str, Any],
) -> float | None:
    return number(
        result.get(
            "validation_roc_auc"
        )
    )


def extract_test_rows(
    result: dict[str, Any],
) -> int | None:
    test = result.get("test")

    if isinstance(test, dict):
        value = number(
            test.get("rows")
        )

        if value is not None:
            return int(value)

    value = number(
        result.get("test_rows")
    )

    if value is None:
        return None

    return int(value)


def extract_window(
    result: dict[str, Any],
) -> dict[str, Any]:
    window = result.get("window")

    if isinstance(window, dict):
        return window

    return {}


def extract_ranking_buckets(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Return top-fraction buckets from supported schemas.

    Important:
    return_buckets in the current ablation experiment are
    probability intervals, not ranking fractions. They are
    therefore not treated as top 1/5/10/20% buckets unless
    top_fraction is explicitly present.
    """

    candidates: list[Any] = []

    for key in (
        "ranking_buckets",
        "top_buckets",
        "prediction_buckets",
    ):
        value = result.get(key)

        if isinstance(value, list):
            candidates.extend(value)

        elif isinstance(value, dict):
            for nested in value.values():

                if isinstance(nested, list):
                    candidates.extend(nested)

                elif isinstance(nested, dict):
                    buckets = nested.get(
                        "buckets"
                    )

                    if isinstance(
                        buckets,
                        list,
                    ):
                        candidates.extend(
                            buckets
                        )

    value = result.get(
        "return_buckets"
    )

    if isinstance(value, list):
        for bucket in value:
            if (
                isinstance(bucket, dict)
                and "top_fraction" in bucket
            ):
                candidates.append(bucket)

    return [
        bucket
        for bucket in candidates
        if isinstance(bucket, dict)
        and number(
            bucket.get(
                "top_fraction"
            )
        )
        is not None
    ]


def extract_probability_buckets(
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    value = result.get(
        "return_buckets"
    )

    if not isinstance(
        value,
        list,
    ):
        return []

    return [
        bucket
        for bucket in value
        if (
            isinstance(bucket, dict)
            and number(
                bucket.get("lower")
            )
            is not None
        )
    ]


def bucket_metric(
    bucket: dict[str, Any],
    key: str,
) -> float | None:
    return number(
        bucket.get(key)
    )


def summarize_group(
    feature_set: str,
    target: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:

    test_aucs = [
        value
        for result in results
        if (
            value := extract_auc(result)
        )
        is not None
    ]

    validation_aucs = [
        value
        for result in results
        if (
            value := extract_validation_auc(
                result
            )
        )
        is not None
    ]

    windows = []

    for result in results:
        window = extract_window(
            result
        )

        windows.append(
            {
                "train_end": window.get(
                    "train_end"
                ),
                "validation_end": window.get(
                    "validation_end"
                ),
                "test_end": window.get(
                    "test_end"
                ),
                "model": result.get(
                    "model",
                    result.get(
                        "selected_model"
                    ),
                ),
                "validation_auc": (
                    extract_validation_auc(
                        result
                    )
                ),
                "test_auc": extract_auc(
                    result
                ),
                "test_rows": extract_test_rows(
                    result
                ),
            }
        )

    ranking_rows = []
    probability_rows = []

    for result in results:

        window = extract_window(
            result
        )

        test_end = window.get(
            "test_end"
        )

        for bucket in extract_ranking_buckets(
            result
        ):
            ranking_rows.append(
                {
                    "test_end": test_end,
                    "top_fraction": number(
                        bucket.get(
                            "top_fraction"
                        )
                    ),
                    "rows": number(
                        bucket.get(
                            "rows"
                        )
                    ),
                    "event_rate": bucket_metric(
                        bucket,
                        "event_rate",
                    ),
                    "baseline_event_rate": (
                        bucket_metric(
                            bucket,
                            "baseline_event_rate",
                        )
                    ),
                    "lift_ratio": bucket_metric(
                        bucket,
                        "event_rate_lift_ratio",
                    ),
                    "mean_return": bucket_metric(
                        bucket,
                        "mean_return",
                    ),
                    "median_return": bucket_metric(
                        bucket,
                        "median_return",
                    ),
                    "baseline_mean_return": (
                        bucket_metric(
                            bucket,
                            "baseline_mean_return",
                        )
                    ),
                }
            )

        for bucket in extract_probability_buckets(
            result
        ):
            probability_rows.append(
                {
                    "test_end": test_end,
                    "lower": bucket_metric(
                        bucket,
                        "lower",
                    ),
                    "upper": bucket_metric(
                        bucket,
                        "upper",
                    ),
                    "rows": bucket_metric(
                        bucket,
                        "rows",
                    ),
                    "mean_return": bucket_metric(
                        bucket,
                        "mean_return",
                    ),
                    "median_return": bucket_metric(
                        bucket,
                        "median_return",
                    ),
                }
            )

    windows_with_auc = [
        window
        for window in windows
        if window["test_auc"] is not None
    ]

    signal_windows = [
        window
        for window in windows_with_auc
        if window["test_auc"]
        >= AUC_SIGNAL_THRESHOLD
    ]

    strong_windows = [
        window
        for window in windows_with_auc
        if window["test_auc"]
        >= AUC_STRONG_THRESHOLD
    ]

    ranking_by_fraction = {}

    for fraction in TOP_FRACTIONS:

        rows = [
            row
            for row in ranking_rows
            if (
                row["top_fraction"]
                is not None
                and abs(
                    row["top_fraction"]
                    - fraction
                )
                < 1e-9
            )
        ]

        lifts = [
            row["lift_ratio"]
            for row in rows
            if row["lift_ratio"]
            is not None
        ]

        event_rates = [
            row["event_rate"]
            for row in rows
            if row["event_rate"]
            is not None
        ]

        returns = [
            row["mean_return"]
            for row in rows
            if row["mean_return"]
            is not None
        ]

        ranking_by_fraction[
            str(fraction)
        ] = {
            "windows": len(rows),
            "mean_lift_ratio": safe_mean(
                lifts
            ),
            "median_lift_ratio": safe_median(
                lifts
            ),
            "mean_event_rate": safe_mean(
                event_rates
            ),
            "mean_return": safe_mean(
                returns
            ),
        }

    return {
        "feature_set": feature_set,
        "target": target,
        "result_rows": len(results),
        "windows": len(windows),
        "windows_with_test_auc": len(
            windows_with_auc
        ),
        "test_auc": {
            "mean": safe_mean(
                test_aucs
            ),
            "median": safe_median(
                test_aucs
            ),
            "min": (
                min(test_aucs)
                if test_aucs
                else None
            ),
            "max": (
                max(test_aucs)
                if test_aucs
                else None
            ),
            "above_0_55": len(
                signal_windows
            ),
            "above_0_60": len(
                strong_windows
            ),
        },
        "validation_auc": {
            "mean": safe_mean(
                validation_aucs
            ),
            "median": safe_median(
                validation_aucs
            ),
        },
        "ranking_signal": {
            "available": bool(
                ranking_rows
            ),
            "by_top_fraction": (
                ranking_by_fraction
            ),
        },
        "probability_bucket_signal": {
            "available": bool(
                probability_rows
            ),
            "bucket_rows": len(
                probability_rows
            ),
        },
        "windows_detail": windows,
    }


def compare_feature_sets(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Produce a neutral feature-set comparison.

    This intentionally does not rank or score feature sets.
    """

    grouped = defaultdict(list)

    for summary in summaries:
        grouped[
            summary["feature_set"]
        ].append(summary)

    rows = []

    for feature_set, items in grouped.items():

        aucs = [
            summary["test_auc"]["mean"]
            for summary in items
            if summary["test_auc"]["mean"]
            is not None
        ]

        signal_window_count = sum(
            summary["test_auc"][
                "above_0_55"
            ]
            for summary in items
        )

        auc_window_count = sum(
            summary[
                "windows_with_test_auc"
            ]
            for summary in items
        )

        rows.append(
            {
                "feature_set": feature_set,
                "targets": unique(
                    [
                        summary["target"]
                        for summary in items
                    ]
                ),
                "mean_test_auc_across_targets": (
                    safe_mean(aucs)
                ),
                "test_auc_windows": (
                    auc_window_count
                ),
                "test_auc_windows_above_0_55": (
                    signal_window_count
                ),
            }
        )

    return rows


def determine_status(
    summaries: list[dict[str, Any]],
) -> tuple[str, list[str]]:

    reasons = []

    if not summaries:
        return (
            "FAIL",
            [
                "Inga ablation-resultat hittades."
            ],
        )

    total_windows = sum(
        summary[
            "windows_with_test_auc"
        ]
        for summary in summaries
    )

    if total_windows == 0:
        return (
            "FAIL",
            [
                "Inga test-AUC-värden hittades."
            ],
        )

    ranking_available = any(
        summary[
            "ranking_signal"
        ]["available"]
        for summary in summaries
    )

    if not ranking_available:
        reasons.append(
            "Top 1/5/10/20%-ranking saknas "
            "i latest_run.json. "
            "Nuvarande experiment innehåller "
            "sannolikhetsintervall, inte "
            "rangordnade toppfraktioner."
        )

    signal_windows = sum(
        summary["test_auc"][
            "above_0_55"
        ]
        for summary in summaries
    )

    if signal_windows == 0:
        reasons.append(
            "Ingen test-window har "
            "AUC >= 0.55."
        )

    elif signal_windows < total_windows:
        reasons.append(
            f"AUC >= 0.55 i "
            f"{signal_windows} av "
            f"{total_windows} test-windows."
        )

    if not reasons:
        return (
            "PASS",
            reasons,
        )

    return (
        "WARN",
        reasons,
    )


def main() -> None:

    if not INPUT_PATH.exists():
        raise SystemExit(
            f"Saknar {INPUT_PATH}"
        )

    data = json.loads(
        INPUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    results = data.get(
        "results"
    )

    if not isinstance(
        results,
        list,
    ):
        raise SystemExit(
            "latest_run.json saknar "
            "en results-lista."
        )

    grouped = defaultdict(list)

    for result in results:

        if not isinstance(
            result,
            dict,
        ):
            continue

        grouped[
            result_key(result)
        ].append(result)

    summaries = [
        summarize_group(
            feature_set,
            target,
            group,
        )
        for (
            feature_set,
            target,
        ), group in sorted(
            grouped.items()
        )
    ]

    status, reasons = determine_status(
        summaries
    )

    output = {
        "dataset": data.get(
            "experiment",
            "unknown",
        ),
        "source_run_id": data.get(
            "run_id"
        ),
        "created_at": data.get(
            "created_at"
        ),
        "status": status,
        "status_reasons": reasons,
        "experiment_summary": {
            "feature_sets": unique(
                data.get(
                    "feature_sets",
                    [],
                )
            ),
            "targets": unique(
                data.get(
                    "ablation_targets",
                    [],
                )
            ),
            "result_rows": len(
                results
            ),
            "groups": len(
                summaries
            ),
        },
        "groups": summaries,
        "feature_set_comparison": (
            compare_feature_sets(
                summaries
            )
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "================================"
    )
    print(
        "BLANKDISS ABLATION QC"
    )
    print(
        "================================"
    )

    print(
        f"Run: {data.get('run_id', '?')}"
    )

    print(
        f"Results: {len(results)}"
    )

    print(
        f"Groups: {len(summaries)}"
    )

    print(
        f"Status: {status}"
    )

    print()

    for summary in summaries:

        auc = summary[
            "test_auc"
        ]

        print(
            f"{summary['feature_set']} | "
            f"{summary['target']} | "
            f"windows={summary['windows']} | "
            f"AUC median={fmt(auc['median'])} | "
            f"min={fmt(auc['min'])} | "
            f"max={fmt(auc['max'])} | "
            f">=0.55={auc['above_0_55']}"
        )

        ranking = summary[
            "ranking_signal"
        ]

        if ranking["available"]:

            for fraction in TOP_FRACTIONS:

                bucket = ranking[
                    "by_top_fraction"
                ][
                    str(fraction)
                ]

                if bucket[
                    "windows"
                ]:

                    print(
                        f"  top {fraction:.0%} | "
                        f"windows={bucket['windows']} | "
                        f"lift={ratio(bucket['mean_lift_ratio'])} | "
                        f"return={pct(bucket['mean_return'])}"
                    )

    print()

    for reason in reasons:
        print(
            f"WARN: {reason}"
        )

    print()

    print(
        f"QC written: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
