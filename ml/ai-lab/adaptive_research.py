"""Controlled adaptive research refinement for Blankdiss AI Lab.

This module proposes bounded follow-up specifications after the predefined
research inventory is exhausted. It never changes an existing specification
and never touches locked confirmation specs.

The adaptive loop is intentionally finite and conservative:
- at most three generated refinement experiments;
- only completed, non-locked research results are inspected;
- the strongest stable parameter cell is used only to define a narrower
  exploratory validation scan;
- generated scans use validation data, not test data;
- locked confirmation remains a separate final checkpoint.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

MAX_ADAPTIVE_ITERATIONS = 3
MIN_VALID_N = 100


def _completed_ids(
    research_runs: list[dict[str, Any]],
) -> set[str]:
    result: set[str] = set()

    for run in research_runs:
        for item in run.get("specs", []):
            if isinstance(item, str):
                result.add(item)
                continue

            if not isinstance(item, dict):
                continue

            for key in (
                "id",
                "spec_id",
                "experiment_id",
            ):
                if item.get(key) is not None:
                    result.add(
                        str(item[key])
                    )
                    break

    return result


def _latest_result(
    research_state: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
] | None:
    research = research_state.get(
        "research",
        {},
    )

    specs = {
        str(spec.get("id")): spec
        for spec in research.get(
            "specs",
            [],
        )
        if (
            isinstance(spec, dict)
            and spec.get("id")
        )
    }

    runs = sorted(
        research_state
        .get(
            "research_runs",
            {},
        )
        .get(
            "runs",
            [],
        ),
        key=lambda item: str(
            item.get(
                "created_at_utc",
                "",
            )
        ),
    )

    for run in reversed(runs):
        for item in reversed(
            run.get(
                "specs",
                [],
            )
        ):
            if not isinstance(
                item,
                dict,
            ):
                continue

            spec_id = (
                item.get("id")
                or item.get("spec_id")
            )

            spec = specs.get(
                str(spec_id)
            )

            if not spec:
                continue

            stage = str(
                spec.get(
                    "stage",
                    "",
                )
            ).lower()

            if spec.get(
                "locked"
            ) is True:
                continue

            if stage == "migration":
                continue

            if stage not in {
                "signal_mapping",
                "hypothesis_test",
                "adaptive_refinement",
            }:
                continue

            result_path = item.get(
                "result"
            )

            if not result_path:
                continue

            path = (
                ROOT
                / str(result_path)
            )

            if not path.exists():
                continue

            try:
                payload = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )
            except (
                OSError,
                json.JSONDecodeError,
            ):
                continue

            if isinstance(
                payload,
                dict,
            ):
                return (
                    spec,
                    payload,
                )

    return None


def _stable_candidate(
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    rows = payload.get(
        "results",
        [],
    )

    if not isinstance(
        rows,
        list,
    ):
        return None

    groups: dict[
        tuple[float, float, str],
        list[dict[str, Any]],
    ] = {}

    for row in rows:
        if not isinstance(
            row,
            dict,
        ):
            continue

        try:
            fraction_x = float(
                row["fraction_x"]
            )

            fraction_y = float(
                row["fraction_y"]
            )

            target = str(
                row["target"]
            )

            lift = float(
                row["lift"]
            )

            n = int(
                row.get(
                    "n_valid",
                    row.get(
                        "n",
                        0,
                    ),
                )
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        if not math.isfinite(
            lift
        ):
            continue

        if n < MIN_VALID_N:
            continue

        key = (
            fraction_x,
            fraction_y,
            target,
        )

        groups.setdefault(
            key,
            [],
        ).append(row)

    candidates: list[
        dict[str, Any]
    ] = []

    for (
        (
            fraction_x,
            fraction_y,
            target,
        ),
        group,
    ) in groups.items():
        windows = {
            str(
                row.get(
                    "window"
                )
            )
            for row in group
        }

        if len(windows) < 2:
            continue

        lifts = [
            float(
                row["lift"]
            )
            for row in group
        ]

        ns = [
            int(
                row.get(
                    "n_valid",
                    row.get(
                        "n",
                        0,
                    ),
                )
            )
            for row in group
        ]

        # Require the candidate to remain above baseline
        # in every observed window.
        if any(
            lift <= 1.0
            for lift in lifts
        ):
            continue

        mean_lift = (
            sum(lifts)
            / len(lifts)
        )

        min_lift = min(
            lifts
        )

        candidates.append(
            {
                "fraction_x": fraction_x,
                "fraction_y": fraction_y,
                "target": target,
                "mean_lift": mean_lift,
                "min_lift": min_lift,
                "min_n": min(ns),
                "windows": sorted(
                    windows
                ),
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item["min_lift"],
            item["mean_lift"],
            item["min_n"],
        ),
        reverse=True,
    )

    return candidates[0]


def _refined_values(
    value: float,
) -> list[float]:
    """Return a small neighbourhood around a candidate fraction.

    These values are exploratory only. They are never used to modify
    a locked confirmation specification.
    """
    values = {
        round(
            value,
            4,
        ),
        round(
            max(
                0.01,
                value * 0.75,
            ),
            4,
        ),
        round(
            min(
                0.25,
                value * 1.25,
            ),
            4,
        ),
    }

    return sorted(
        values,
        reverse=True,
    )


def build_adaptive_plan(
    research_state: dict[str, Any],
) -> dict[str, Any]:
    """Build one bounded adaptive research proposal."""

    research = research_state.get(
        "research",
        {}
    )

    runs = (
        research_state
        .get(
            "research_runs",
            {},
        )
        .get(
            "runs",
            [],
        )
    )

    completed = _completed_ids(
        runs
    )

    adaptive_ids = [
        spec_id
        for spec_id in completed
        if spec_id.startswith(
            "adaptive_"
        )
    ]

    if len(adaptive_ids) >= (
        MAX_ADAPTIVE_ITERATIONS
    ):
        return {
            "status": (
                "no_adaptive_proposal"
            ),
            "reason": (
                "adaptive_iteration_limit_reached"
            ),
            "iteration_count": len(
                adaptive_ids
            ),
        }

    latest = _latest_result(
        research_state
    )

    if latest is None:
        return {
            "status": (
                "no_adaptive_proposal"
            ),
            "reason": (
                "no_eligible_completed_result"
            ),
            "iteration_count": len(
                adaptive_ids
            ),
        }

    source_spec, payload = latest

    candidate = _stable_candidate(
        payload
    )

    if candidate is None:
        return {
            "status": (
                "no_adaptive_proposal"
            ),
            "reason": (
                "no_stable_candidate_with_"
                "sufficient_sample"
            ),
            "source_spec": source_spec.get(
                "id"
            ),
            "iteration_count": len(
                adaptive_ids
            ),
        }

    iteration = (
        len(adaptive_ids)
        + 1
    )

    source_id = str(
        source_spec["id"]
    )

    new_id = (
        f"adaptive_"
        f"{source_id}"
        f"_r{iteration}"
    )

    if new_id in completed:
        return {
            "status": (
                "no_adaptive_proposal"
            ),
            "reason": (
                "generated_spec_already_completed"
            ),
            "source_spec": source_id,
            "iteration_count": iteration,
        }

    source_signals = (
        source_spec.get(
            "signals"
        )
        or []
    )

    if len(source_signals) < 2:
        return {
            "status": (
                "no_adaptive_proposal"
            ),
            "reason": (
                "source_spec_does_not_have_"
                "two_signals"
            ),
            "source_spec": source_id,
        }

    signals = []

    for index, signal in enumerate(
        source_signals[:2]
    ):
        if not isinstance(
            signal,
            dict,
        ):
            return {
                "status": (
                    "no_adaptive_proposal"
                ),
                "reason": (
                    "invalid_source_signal_definition"
                ),
                "source_spec": source_id,
            }

        fraction = (
            candidate[
                "fraction_x"
            ]
            if index == 0
            else candidate[
                "fraction_y"
            ]
        )

        signals.append(
            {
                "name": signal.get(
                    "name"
                ),
                "direction": signal.get(
                    "direction",
                    "upper",
                ),
                "bins": _refined_values(
                    float(fraction)
                ),
            }
        )

    source_analysis = (
        source_spec.get(
            "analysis"
        )
    )

    if isinstance(
        source_analysis,
        dict,
    ):
        analysis_type = source_analysis.get(
            "type",
            "interaction",
        )
    else:
        analysis_type = "interaction"

    generated_spec = {
        "id": new_id,
        "question": (
            f"Adaptive refinement of "
            f"{source_id}: "
            "test whether the strongest "
            "stable interaction remains "
            "informative in a narrower "
            "parameter neighbourhood."
        ),
        "mode": source_spec.get(
            "mode",
            "scan",
        ),
        "signals": signals,
        "targets": [
            candidate["target"]
        ],
        "analysis": {
            "type": analysis_type,
            "bootstrap": False,
        },
        "windows": (
            source_spec.get(
                "windows"
            )
            or [
                "window_1",
                "window_2",
            ]
        ),
        "splits": [
            "validation"
        ],
        "metadata": {
            "stage": (
                "adaptive_refinement"
            ),
            "purpose": (
                "controlled_parameter_refinement"
            ),
            "source_spec": source_id,
            "adaptive_iteration": iteration,
            "selection_policy": (
                "stable_multi_window_candidate"
            ),
            "selection_source": (
                "previous_exploratory_result"
            ),
            "min_valid_n": MIN_VALID_N,
            "candidate": candidate,
            "note": (
                "Generated automatically "
                "as an exploratory validation "
                "scan. It must not be treated "
                "as confirmation and must not "
                "modify the locked "
                "prospective-confirmation "
                "specification."
            ),
        },
    }

    return {
        "status": (
            "adaptive_proposal_created"
        ),
        "iteration_count": iteration,
        "source_spec": source_id,
        "generated_spec": generated_spec,
        "analysis": {
            "candidate": candidate,
            "source_result": payload.get(
                "id",
                source_id,
            ),
        },
    }
