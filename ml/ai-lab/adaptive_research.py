"""Controlled adaptive research refinement for Blankdiss AI Lab.

Adaptive refinement is deliberately restricted to validation data. It may
generate a new exploratory specification, but it never changes an existing
specification and never touches locked prospective confirmation.

The first adaptive family is regime-comparison research, because the
momentum/short-interest walk-forward experiment already provides validation
splits that can be used for controlled parameter refinement.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

MAX_ADAPTIVE_ITERATIONS = 3
MIN_VALID_N = 100

SUPPORTED_SOURCE = "momentum_si_regime_walk_forward"


def _completed_ids(
    runs: list[dict[str, Any]],
) -> set[str]:
    """Return all research specification IDs already executed."""
    completed: set[str] = set()

    for run in runs:
        for item in run.get("specs", []):
            if isinstance(item, str):
                completed.add(item)
                continue

            if not isinstance(item, dict):
                continue

            for key in (
                "id",
                "spec_id",
                "experiment_id",
            ):
                value = item.get(key)

                if value is not None:
                    completed.add(str(value))
                    break

    return completed


def _specs_by_id(
    research_state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return research specifications indexed by ID."""
    specs = (
        research_state
        .get("research", {})
        .get("specs", [])
    )

    if not isinstance(specs, list):
        return {}

    return {
        str(spec.get("id")): spec
        for spec in specs
        if (
            isinstance(spec, dict)
            and spec.get("id")
        )
    }


def _runs(
    research_state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return normalized research runs."""
    runs = (
        research_state
        .get("research_runs", {})
        .get("runs", [])
    )

    if not isinstance(runs, list):
        return []

    return [
        run
        for run in runs
        if isinstance(run, dict)
    ]


def _result_path(
    manifest_item: dict[str, Any],
) -> Path | None:
    """Resolve a research result path inside the repository."""
    result = manifest_item.get("result")

    if not result:
        return None

    path = ROOT / str(result)

    if not path.is_file():
        return None

    return path


def _latest_eligible_result(
    research_state: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
] | None:
    """Find the newest result from the supported adaptive family.

    The first adaptive family is intentionally narrow. This prevents the
    adaptive engine from interpreting unrelated research result schemas as
    if they were compatible with regime-comparison experiments.
    """
    specs = _specs_by_id(
        research_state
    )

    candidates: list[
        tuple[
            str,
            dict[str, Any],
            dict[str, Any],
        ]
    ] = []

    for run in _runs(
        research_state
    ):
        created = str(
            run.get(
                "created_at_utc",
                "",
            )
        )

        for item in run.get(
            "specs",
            [],
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

            if spec_id is None:
                continue

            spec = specs.get(
                str(spec_id)
            )

            if not spec:
                continue

            if spec.get(
                "locked"
            ) is True:
                continue

            stage = str(
                spec.get(
                    "stage",
                    "",
                )
            ).strip().lower()

            if stage == "migration":
                continue

            normalized_id = str(
                spec_id
            )

            if (
                normalized_id != SUPPORTED_SOURCE
                and not normalized_id.startswith(
                    "adaptive_"
                )
            ):
                continue

            result_path = _result_path(
                item
            )

            if result_path is None:
                continue

            try:
                payload = json.loads(
                    result_path.read_text(
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
                candidates.append(
                    (
                        created,
                        spec,
                        payload,
                    )
                )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0]
    )

    _, spec, payload = candidates[-1]

    return (
        spec,
        payload,
    )


def _refined_values(
    value: float,
) -> list[float]:
    """Build a small bounded neighbourhood around a candidate value.

    These values are exploratory only. They never modify the source
    specification or the locked confirmation specification.
    """
    raw_values = (
        value * 0.75,
        value,
        value * 1.25,
    )

    values = {
        round(
            max(
                0.01,
                min(
                    0.50,
                    candidate,
                ),
            ),
            4,
        )
        for candidate in raw_values
    }

    return sorted(
        values,
        reverse=True,
    )


def _stable_validation_candidate(
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Find a stable candidate using validation results only.

    A candidate must:
    - occur in at least two validation windows;
    - have sufficient sample size in every window;
    - have a positive incremental event-rate difference in every window.

    No test result is considered by this function.
    """
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
        tuple[
            float,
            float,
            str,
        ],
        list[dict[str, Any]],
    ] = {}

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
            baseline_fraction = float(
                row[
                    "baseline_fraction"
                ]
            )

            incremental_fraction = float(
                row[
                    "incremental_fraction"
                ]
            )

            target = str(
                row["target"]
            )

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

        if not math.isfinite(
            difference
        ):
            continue

        if n < MIN_VALID_N:
            continue

        key = (
            baseline_fraction,
            incremental_fraction,
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
            baseline_fraction,
            incremental_fraction,
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

        differences = [
            float(
                row[
                    "absolute_event_rate_difference"
                ]
            )
            for row in group
        ]

        sample_sizes = [
            int(
                row.get(
                    "combined_n",
                    row.get(
                        "n",
                        0,
                    ),
                )
            )
            for row in group
        ]

        # Require the incremental effect to remain positive in
        # every validation window.
        if any(
            difference <= 0.0
            for difference in differences
        ):
            continue

        candidates.append(
            {
                "baseline_fraction": (
                    baseline_fraction
                ),
                "incremental_fraction": (
                    incremental_fraction
                ),
                "target": target,
                "mean_difference": (
                    sum(differences)
                    / len(differences)
                ),
                "min_difference": min(
                    differences
                ),
                "min_n": min(
                    sample_sizes
                ),
                "windows": sorted(
                    windows
                ),
            }
        )

    if not candidates:
        return None

    # Stability first, then average effect, then sample size.
    candidates.sort(
        key=lambda item: (
            item["min_difference"],
            item["mean_difference"],
            item["min_n"],
        ),
        reverse=True,
    )

    return candidates[0]


def build_adaptive_plan(
    research_state: dict[str, Any],
) -> dict[str, Any]:
    """Build one bounded adaptive research proposal."""
    runs = _runs(
        research_state
    )

    completed = _completed_ids(
        runs
    )

    adaptive_ids = {
        spec_id
        for spec_id in completed
        if spec_id.startswith(
            "adaptive_"
        )
    }

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

    latest = _latest_eligible_result(
        research_state
    )

    if latest is None:
        return {
            "status": (
                "no_adaptive_proposal"
            ),
            "reason": (
                "no_validation_result_for_supported_family"
            ),
            "iteration_count": len(
                adaptive_ids
            ),
        }

    source_spec, payload = latest

    candidate = _stable_validation_candidate(
        payload
    )

    if candidate is None:
        return {
            "status": (
                "no_adaptive_proposal"
            ),
            "reason": (
                "no_stable_validation_candidate"
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

    source_signals = source_spec.get(
        "signals"
    )

    if (
        not isinstance(
            source_signals,
            list,
        )
        or len(source_signals) != 2
    ):
        return {
            "status": (
                "no_adaptive_proposal"
            ),
            "reason": (
                "source_spec_does_not_have_"
                "exactly_two_signals"
            ),
            "source_spec": source_id,
        }

    baseline_signal = source_signals[0]
    incremental_signal = source_signals[1]

    if (
        not isinstance(
            baseline_signal,
            dict,
        )
        or not isinstance(
            incremental_signal,
            dict,
        )
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

    baseline_direction = str(
        baseline_signal.get(
            "direction",
            "lower",
        )
    )

    incremental_direction = str(
        incremental_signal.get(
            "direction",
            "upper",
        )
    )

    generated_spec = {
        "id": new_id,
        "question": (
            f"Kontrollerad adaptiv förfining av "
            f"{source_id}: är den stabila "
            "validation-effekten fortsatt synlig "
            "i ett smalare parameterområde?"
        ),
        "mode": source_spec.get(
            "mode",
            "deep",
        ),
        "signals": [
            {
                "name": baseline_signal.get(
                    "name"
                ),
                "direction": baseline_direction,
                "bins": _refined_values(
                    candidate[
                        "baseline_fraction"
                    ]
                ),
            },
            {
                "name": incremental_signal.get(
                    "name"
                ),
                "direction": incremental_direction,
                "bins": _refined_values(
                    candidate[
                        "incremental_fraction"
                    ]
                ),
            },
        ],
        "targets": [
            candidate[
                "target"
            ]
        ],
        "analysis": {
            "type": (
                "regime_comparison"
            ),
            "bootstrap": False,
        },
        "windows": source_spec.get(
            "windows",
            [
                "window_1",
                "window_2",
            ],
        ),
        "splits": [
            "validation"
        ],
        "metadata": {
            "stage": (
                "adaptive_refinement"
            ),
            "purpose": (
                "controlled_validation_"
                "parameter_refinement"
            ),
            "source_spec": source_id,
            "adaptive_iteration": iteration,
            "selection_policy": (
                "stable_multi_window_"
                "validation_effect"
            ),
            "candidate": candidate,
            "constraints": [
                "Validation data only.",
                "No test-data threshold selection.",
                "Do not modify any existing specification.",
                "Do not modify the locked prospective confirmation.",
                "This experiment is exploratory and is not confirmation.",
            ],
        },
    }

    return {
        "status": (
            "adaptive_proposal_created"
        ),
        "iteration_count": iteration,
        "source_spec": source_id,
        "generated_spec": generated_spec,
        "output_path": (
            f"ml/research/specs/"
            f"{new_id}.yaml"
        ),
    }


def main(
    experiment: dict[str, Any],
) -> dict[str, Any]:
    """Entry point used by the AI Lab runner."""
    research_state = experiment.get(
        "research_state"
    )

    if not isinstance(
        research_state,
        dict,
    ):
        raise ValueError(
            "adaptive_research requires "
            "a research_state object."
        )

    return build_adaptive_plan(
        research_state
    )
