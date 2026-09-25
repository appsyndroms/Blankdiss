"""
Deterministic planner for the Blankdiss AI Lab.

The planner does not execute research experiments. It inspects the
current research state and proposes the next eligible existing
research specification.

Design principles:
- deterministic
- conservative
- no modification of research specifications
- no parameter optimization
- no test-set selection
- no automatic selection of locked prospective confirmations
- already executed specifications are skipped
- hypothesis tests have priority over discovery scans
"""

from __future__ import annotations

from typing import Any


PLANNER_VERSION = 2


def _spec_id(spec: dict[str, Any]) -> str:
    """Return the specification identifier."""
    value = spec.get("id")

    if value is None:
        value = spec.get("spec_id")

    if value is None:
        raise ValueError(
            "Research specification is missing an id/spec_id."
        )

    return str(value)


def _stage(spec: dict[str, Any]) -> str:
    """Return the normalized research stage."""
    return str(
        spec.get("stage", "")
    ).strip().lower()


def _is_locked(spec: dict[str, Any]) -> bool:
    """Return whether the specification is locked."""
    return bool(
        spec.get("locked", False)
    )


def _run_spec_ids(
    research_runs: list[dict[str, Any]],
) -> set[str]:
    """Return spec IDs already represented in research run manifests."""
    completed: set[str] = set()

    for run in research_runs:
        specs = run.get("specs", [])

        if not isinstance(specs, list):
            continue

        for item in specs:
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


def _select_existing_spec(
    specs: list[dict[str, Any]],
    research_runs: list[dict[str, Any]],
) -> tuple[
    dict[str, Any] | None,
    str,
]:
    """Select the next eligible existing spec.

    Locked prospective-confirmation specifications are never selected
    automatically. They are controlled checkpoints that require their
    own execution decision.

    Selection priority:
        1. unexecuted hypothesis tests
        2. unexecuted discovery/signal-mapping specs

    Already represented research specs are skipped so the planner does
    not repeatedly propose the same experiment.
    """
    completed_ids = _run_spec_ids(
        research_runs
    )

    hypothesis = [
        spec
        for spec in specs
        if _stage(spec) == "hypothesis_test"
        and not _is_locked(spec)
        and _spec_id(spec) not in completed_ids
    ]

    if hypothesis:
        return (
            hypothesis[0],
            "hypothesis_test",
        )

    discovery = [
        spec
        for spec in specs
        if _stage(spec) == "signal_mapping"
        and not _is_locked(spec)
        and _spec_id(spec) not in completed_ids
    ]

    if discovery:
        return (
            discovery[0],
            "discovery",
        )

    return None, "none"


def _copy_locked_parameters(
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Copy parameters from a locked specification.

    This is intentionally a shallow structural copy. The planner does
    not alter parameter values.
    """
    parameters = spec.get(
        "parameters",
        {},
    )

    if not isinstance(parameters, dict):
        return {}

    return dict(parameters)


def _build_constraints(
    spec: dict[str, Any],
) -> list[str]:
    """Build conservative execution constraints."""
    constraints = [
        "Do not modify the research specification.",
        "Do not introduce arbitrary parameters.",
        "Do not optimize thresholds on the test set.",
        "Do not select the test data based on observed outcomes.",
        "Do not declare the research hypothesis confirmed from the planner output.",
        "Do not automatically select a locked prospective-confirmation specification.",
    ]

    if _is_locked(spec):
        constraints.extend(
            [
                "Treat all locked parameters as immutable.",
                "Do not add new bins.",
                "Do not optimize thresholds.",
                "Do not switch the primary endpoint.",
                "Do not use test-data results to alter the specification.",
            ]
        )

    return constraints


def _open_question_ids(
    research_state: dict[str, Any],
) -> set[str]:
    """Return specification IDs currently represented as open questions."""
    questions = (
        research_state
        .get("research", {})
        .get("open_questions", [])
    )

    if not isinstance(questions, list):
        return set()

    result: set[str] = set()

    for item in questions:
        if not isinstance(item, dict):
            continue

        value = item.get("spec_id")

        if value is not None:
            result.add(str(value))

    return result


def build_experiment_plan(
    research_state: dict[str, Any],
) -> dict[str, Any]:
    """Build a deterministic next-experiment proposal."""
    research = research_state.get(
        "research",
        {},
    )

    if not isinstance(research, dict):
        research = {}

    specs = research.get(
        "specs",
        [],
    )

    if not isinstance(specs, list):
        specs = []

    normalized_specs = [
        spec
        for spec in specs
        if isinstance(spec, dict)
    ]

    research_runs_section = research_state.get(
        "research_runs",
        {},
    )

    if not isinstance(
        research_runs_section,
        dict,
    ):
        research_runs_section = {}

    research_runs = research_runs_section.get(
        "runs",
        [],
    )

    if not isinstance(research_runs, list):
        research_runs = []

    research_runs = [
        run
        for run in research_runs
        if isinstance(run, dict)
    ]

    completed_spec_ids = _run_spec_ids(
        research_runs
    )

    selected_spec, stage = _select_existing_spec(
        normalized_specs,
        research_runs,
    )

    locked_specs = [
        spec
        for spec in normalized_specs
        if _is_locked(spec)
    ]

    locked_confirmation_specs = [
        spec
        for spec in locked_specs
        if _stage(spec) == "prospective_confirmation"
    ]

    warnings: list[str] = []

    if locked_confirmation_specs:
        warnings.append(
            "Locked prospective-confirmation specification(s) "
            "exist but are not automatically selected by the planner."
        )

    migration_specs = [
        spec
        for spec in normalized_specs
        if _stage(spec) == "migration"
    ]

    if migration_specs:
        warnings.append(
            "Migration specifications are visible in the research "
            "state but are not treated as current research evidence "
            "or automatic planning candidates."
        )

    if not research_runs:
        warnings.append(
            "No research-engine runs are visible in the current "
            "research state."
        )

    if selected_spec is None:
        remaining_hypothesis = [
            spec
            for spec in normalized_specs
            if _stage(spec) == "hypothesis_test"
            and not _is_locked(spec)
            and _spec_id(spec) not in completed_spec_ids
        ]

        remaining_discovery = [
            spec
            for spec in normalized_specs
            if _stage(spec) == "signal_mapping"
            and not _is_locked(spec)
            and _spec_id(spec) not in completed_spec_ids
        ]

        if not remaining_hypothesis and not remaining_discovery:
            warnings.append(
                "No unexecuted hypothesis-test or discovery "
                "specification is available for automatic planning."
            )

        return {
            "planner_version": PLANNER_VERSION,
            "status": "no_proposal",
            "selection": {
                "source_spec": None,
                "stage": "none",
                "locked": False,
            },
            "parameters": {},
            "constraints": [
                "Do not create a new research specification "
                "automatically.",
                "Do not modify existing research specifications.",
                "Do not automatically select a locked "
                "prospective-confirmation specification.",
            ],
            "research_context": {
                "spec_count": len(
                    normalized_specs
                ),
                "research_run_count": len(
                    research_runs
                ),
                "completed_spec_count": len(
                    completed_spec_ids
                ),
                "open_question_count": len(
                    _open_question_ids(
                        research_state
                    )
                ),
                "locked_spec_count": len(
                    locked_specs
                ),
            },
            "warnings": warnings,
        }

    selected_id = _spec_id(
        selected_spec
    )

    selection = {
        "source_spec": selected_id,
        "stage": stage,
        "locked": _is_locked(
            selected_spec
        ),
    }

    parameters = _copy_locked_parameters(
        selected_spec
    )

    constraints = _build_constraints(
        selected_spec
    )

    question = selected_spec.get(
        "question"
    )

    if question is not None:
        selection["question"] = str(
            question
        )

    return {
        "planner_version": PLANNER_VERSION,
        "status": "proposal_created",
        "selection": selection,
        "parameters": parameters,
        "constraints": constraints,
        "research_context": {
            "spec_count": len(
                normalized_specs
            ),
            "research_run_count": len(
                research_runs
            ),
            "completed_spec_count": len(
                completed_spec_ids
            ),
            "open_question_count": len(
                _open_question_ids(
                    research_state
                )
            ),
            "locked_spec_count": len(
                locked_specs
            ),
        },
        "warnings": warnings,
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
            "experiment_planner requires a "
            "research_state object."
        )

    return build_experiment_plan(
        research_state
    )
