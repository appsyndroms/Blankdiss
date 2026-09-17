"""Aggregation of walk-forward Blankdiss research results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def pool_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Pool walk-forward results by experiment_id.

    The individual window results remain untouched.
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

    hit_values = _values(
        rows,
        "hit_rate",
    )

    return_values = _values(
        rows,
        "return_difference",
    )

    return {
        "experiment_id": experiment_id,
        "signal_name": first[
            "signal_name"
        ],
        "target_name": first[
            "target_name"
        ],
        "tail_fraction": first[
            "tail_fraction"
        ],
        "tail_direction": first[
            "tail_direction"
        ],
        "windows": len(rows),
        "auc": _mean(auc_values),
        "hit_rate": _mean(hit_values),
        "return_difference": _mean(
            return_values
        ),
        "status": _pooled_status(rows),
    }


def _pooled_status(
    rows: list[dict[str, Any]],
) -> str:
    statuses = {
        row.get("status")
        for row in rows
    }

    if (
        "STRONG RESEARCH CANDIDATE"
        in statuses
    ):
        return "STRONG RESEARCH CANDIDATE"

    if "INTERESTING" in statuses:
        return "INTERESTING"

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
