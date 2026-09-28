from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any


NUMERIC_FIELDS = (
    "n",
    "events",
    "event_rate",
    "auc",
    "lift",
    "mean_return",
    "median_return",
)


def _numeric_values(
    rows: list[dict[str, Any]],
    field: str,
) -> list[float]:
    values: list[float] = []

    for row in rows:
        value = row.get(field)

        if isinstance(value, bool):
            continue

        if isinstance(value, (int, float)):
            values.append(float(value))

    return values


def _summarize_field(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, float | int | None]:
    values = _numeric_values(
        rows,
        field,
    )

    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
        }

    return {
        "count": len(values),
        "mean": mean(values),
        "min": min(values),
        "max": max(values),
    }


def _sign_consistency(
    values: list[float],
) -> dict[str, int | float | None]:
    non_zero = [
        value
        for value in values
        if value != 0
    ]

    if not non_zero:
        return {
            "count": 0,
            "positive": 0,
            "negative": 0,
            "zero": len(values),
            "positive_fraction": None,
            "negative_fraction": None,
        }

    positive = sum(
        value > 0
        for value in non_zero
    )

    negative = sum(
        value < 0
        for value in non_zero
    )

    return {
        "count": len(non_zero),
        "positive": positive,
        "negative": negative,
        "zero": len(values) - len(non_zero),
        "positive_fraction": (
            positive / len(non_zero)
        ),
        "negative_fraction": (
            negative / len(non_zero)
        ),
    }


def aggregate_walk_forward(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for result in results:
        window = str(
            result.get(
                "window",
                {}
            ).get(
                "name",
                "",
            )
        )

        for row in result.get(
            "results",
            [],
        ):
            grouped[window].append(row)

    windows: list[dict[str, Any]] = []

    for window_name, rows in sorted(
        grouped.items()
    ):
        summary = {
            "name": window_name,
            "result_count": len(rows),
            "metrics": {
                field: _summarize_field(
                    rows,
                    field,
                )
                for field in NUMERIC_FIELDS
            },
            "stability": {
                "event_rate": _sign_consistency(
                    _numeric_values(
                        rows,
                        "event_rate",
                    )
                ),
                "lift": _sign_consistency(
                    _numeric_values(
                        rows,
                        "lift",
                    )
                ),
                "mean_return": _sign_consistency(
                    _numeric_values(
                        rows,
                        "mean_return",
                    )
                ),
            },
        }

        windows.append(summary)

    all_rows = [
        row
        for rows in grouped.values()
        for row in rows
    ]

    return {
        "window_count": len(windows),
        "result_count": len(all_rows),
        "windows": windows,
        "overall": {
            "metrics": {
                field: _summarize_field(
                    all_rows,
                    field,
                )
                for field in NUMERIC_FIELDS
            }
        },
    }
