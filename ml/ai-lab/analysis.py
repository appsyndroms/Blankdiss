from __future__ import annotations

import math
from typing import Any

from config import (
    ADAPTIVE_PREFIX,
    MIN_VALID_N,
    ROOT,
    RUNS_DIR,
    SIGN_EPSILON,
)
from state import read_json


def load_result(
    path,
) -> dict[str, Any]:
    payload = read_json(
        path
    )

    if payload is None:
        raise ValueError(
            f"Could not read research result: {path}"
        )

    if not isinstance(
        payload.get("results"),
        list,
    ):
        raise ValueError(
            f"Research result has no results list: {path}"
        )

    return payload


def classify_outcome(
    rows: list[dict[str, Any]],
) -> str:
    differences: list[
        float
    ] = []

    for row in rows:
        if not isinstance(
            row,
            dict,
        ):
            continue

        if str(
            row.get(
                "split",
                "",
            )
        ).lower() != "validation":
            continue

        try:
            n = int(
                row.get(
                    "combined_n",
                    row.get(
                        "n",
                        0,
                    ),
                )
            )

            difference = float(
                row[
                    "absolute_event_rate_difference"
                ]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        if (
            n >= MIN_VALID_N
            and math.isfinite(
                difference
            )
        ):
            differences.append(
                difference
            )

    if len(
        differences
    ) < 2:
        return (
            "OUTCOME_INCONCLUSIVE"
        )

    positive = any(
        value > SIGN_EPSILON
        for value in differences
    )

    negative = any(
        value < -SIGN_EPSILON
        for value in differences
    )

    if positive and negative:
        return "OUTCOME_MIXED"

    if positive:
        return "OUTCOME_POSITIVE"

    if negative:
        return "OUTCOME_NEGATIVE"

    return "OUTCOME_INCONCLUSIVE"


def summarize_result(
    path,
) -> dict[str, Any]:
    payload = load_result(
        path
    )

    rows = payload[
        "results"
    ]

    outcome = classify_outcome(
        rows
    )

    differences: list[
        float
    ] = []

    for row in rows:
        if (
            not isinstance(
                row,
                dict,
            )
            or str(
                row.get(
                    "split",
                    "",
                )
            ).lower()
            != "validation"
        ):
            continue

        value = row.get(
            "absolute_event_rate_difference"
        )

        if (
            isinstance(
                value,
                (
                    int,
                    float,
                ),
            )
            and math.isfinite(
                float(value)
            )
        ):
            differences.append(
                float(value)
            )

    return {
        "path": str(
            path.relative_to(
                ROOT
            )
        ),
        "experiment_id": payload.get(
            "id"
        ),
        "outcome": outcome,
        "validation_difference_count": len(
            differences
        ),
        "validation_min_difference": (
            min(differences)
            if differences
            else None
        ),
        "validation_max_difference": (
            max(differences)
            if differences
            else None
        ),
    }


def all_adaptive_summaries() -> list[
    dict[str, Any]
]:
    summaries: list[
        dict[str, Any]
    ] = []

    if not RUNS_DIR.exists():
        return summaries

    for manifest_path in sorted(
        RUNS_DIR.glob(
            "*/manifest.json"
        )
    ):
        manifest = read_json(
            manifest_path
        )

        if not manifest:
            continue

        for item in manifest.get(
            "specs",
            [],
        ):
            if not isinstance(
                item,
                dict,
            ):
                continue

            spec_id = str(
                item.get(
                    "id",
                    "",
                )
            )

            if not spec_id.startswith(
                ADAPTIVE_PREFIX
            ):
                continue

            result_path = (
                ROOT
                / str(
                    item.get(
                        "result",
                        "",
                    )
                )
            )

            if result_path.is_file():
                summaries.append(
                    summarize_result(
                        result_path
                    )
                )

    return summaries
