"""Aggregation of walk-forward Blankdiss research results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def pool_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Pool walk-forward results by experiment_id.

    Individual split results remain untouched.
    """
    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for result in results:
        grouped[
            result["experiment_id"]
        ].append(result)

    return [
        _pool_experiment(
            experiment_id,
            rows,
        )
        for experiment_id, rows
        in grouped.items()
    ]


def _pool_experiment(
    experiment_id: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    first = rows[0]

    auc_values = _values(
        rows,
        "auc",
    )

    event_rate_values = _values(
        rows,
        "event_rate",
    )

    baseline_event_rate_values = _values(
        rows,
        "baseline_event_rate",
    )

    lift_values = _values(
        rows,
        "lift",
    )

    mean_return_values = _values(
        rows,
        "mean_return",
    )

    median_return_values = _values(
        rows,
        "median_return",
    )

    return_ci_low_values = _values(
        rows,
        "bootstrap_ci_low",
    )

    return_ci_high_values = _values(
        rows,
        "bootstrap_ci_high",
    )

    n_values = _values(
        rows,
        "n",
    )

    return_n_values = _values(
        rows,
        "return_n",
    )

    return {
        "experiment_id": experiment_id,
        "signal_name": first["signal_name"],
        "target_name": first["target_name"],
        "tail_fraction": first["tail_fraction"],
        "tail_direction": first["tail_direction"],
        "windows": len(rows),
        "auc": _mean(auc_values),
        "event_rate": _mean(event_rate_values),
        "baseline_event_rate": _mean(
            baseline_event_rate_values
        ),
        "lift": _mean(lift_values),
        "mean_return": _mean(
            mean_return_values
        ),
        "median_return": _mean(
            median_return_values
        ),
        "bootstrap_ci_low": _mean(
            return_ci_low_values
        ),
        "bootstrap_ci_high": _mean(
            return_ci_high_values
        ),
        "n": int(sum(n_values))
        if n_values
        else 0,
        "return_n": int(sum(return_n_values))
        if return_n_values
        else 0,
        "status": _pooled_status(
            rows
        ),
    }


def _pooled_status(
    rows: list[dict[str, Any]],
) -> str:
    statuses = {
        row.get(
            "status",
            "NO_SIGNAL",
        )
        for row in rows
    }

    if "STRONG RESEARCH CANDIDATE" in statuses:
        return "STRONG RESEARCH CANDIDATE"

    if "INTERESTING" in statuses:
        return "INTERESTING"

    if "UNSTABLE" in statuses:
        return "UNSTABLE"

    if "INSUFFICIENT_DATA" in statuses:
        return "INSUFFICIENT_DATA"

    return "NO_SIGNAL"


def _values(
    rows: list[dict[str, Any]],
    key: str,
) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if row.get(key) is not None
    ]


def _mean(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return sum(values) / len(values)
