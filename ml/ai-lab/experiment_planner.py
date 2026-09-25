"""Controlled AI Lab experiment planner.

The planner consumes an explicit research-state snapshot and produces
an auditable experiment proposal.

It does not:
    - execute experiments
    - modify research specifications
    - modify locked parameters
    - optimize thresholds on test data
    - declare hypotheses confirmed

The first implementation is deliberately deterministic. It establishes
the planner contract before introducing model-based reasoning.
"""

from __future__ import annotations

from typing import Any


PLANNER_VERSION = 1


def _as_list(
    value: Any,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    return [
        item
        for item in value
        if isinstance(item, dict)
    ]


def _research(
    state: dict[str, Any],
) -> dict[str, Any]:
    value = state.get(
        "research",
        {},
    )

    if not isinstance(
        value,
        dict,
    ):
        return {}

    return value


def _locked_specs(
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    research = _research(state)

    return _as_list(
        research.get(
            "locked_specs",
            [],
        )
    )


def _specs(
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    research = _research(state)

    return _as_list(
        research.get(
            "specs",
            [],
        )
    )


def _research_runs(
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    value = state.get(
        "research_runs",
        {},
    )

    if not isinstance(
        value,
        dict,
    ):
        return []

    return _as_list(
        value.get(
            "runs",
            [],
        )
    )


def _stage(
    spec: dict[str, Any],
) -> str | None:
    value = spec.get(
        "stage"
    )

    if value is None:
        return None

    return str(value)


def _is_locked(
    spec: dict[str, Any],
) -> bool:
    return (
        spec.get(
            "locked",
            False,
        )
        is True
    )


def _spec_id(
    spec: dict[str, Any],
) -> str | None:
    value = spec.get(
        "id"
    )

    if value is None:
        return None

    return str(value)


def _select_existing_spec(
    specs: list[dict[str, Any]],
) -> tuple[
    dict[str, Any] | None,
    str,
]:
    """Select an existing spec without changing its parameters.

    Priority is deliberately conservative:

    1. locked prospective confirmation
    2. hypothesis test
    3. discovery

    A locked confirmation is selected only as an existing experiment
    to inspect/execute. Its parameters are never reconstructed or changed.
    """

    confirmation = [
        spec
        for spec in specs
        if _stage(spec)
        == "prospective_confirmation"
        and _is_locked(spec)
    ]

    if confirmation:
        return (
            confirmation[0],
            "prospective_confirmation",
        )

    hypothesis = [
        spec
        for spec in specs
        if _stage(spec)
        == "hypothesis_test"
        and not _is_locked(spec)
    ]

    if hypothesis:
        return (
            hypothesis[0],
            "hypothesis_test",
        )

    discovery = [
        spec
        for spec in specs
        if _stage(spec)
        == "signal_mapping"
        and not _is_locked(spec)
    ]

    if discovery:
        return (
            discovery[0],
            "discovery",
        )

    return (
        None,
        "none",
    )


def _build_constraints(
    selected_spec: dict[str, Any] | None,
    stage: str,
) -> list[str]:
    constraints = [
        "Do not modify research specifications.",
        "Do not execute arbitrary code.",
        "Do not optimize parameters against test data.",
        "Do not declare a hypothesis confirmed.",
    ]

    if selected_spec is not None:
        constraints.append(
            "Use the selected specification as declared; "
            "do not silently change its parameters."
        )

    if stage == "prospective_confirmation":
        constraints.extend(
            [
                "Treat confirmation parameters as locked.",
                "Do not search additional bins.",
                "Do not optimize thresholds on confirmation data.",
                "Do not switch the primary endpoint after test start.",
                "Do not use confirmation data for parameter selection.",
            ]
        )

    return constraints


def _build_proposal(
    selected_spec: dict[str, Any],
    stage: str,
    research_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    spec_id = _spec_id(
        selected_spec
    )

    question = selected_spec.get(
        "question"
    )

    metadata = selected_spec.get(
        "metadata",
        {},
    )

    if not isinstance(
        metadata,
        dict,
    ):
        metadata = {}

    hypothesis = metadata.get(
        "hypothesis"
    )

    if hypothesis is None:
        hypothesis = (
            "Evaluate the research question "
            "defined by the existing specification."
        )

    targets = selected_spec.get(
        "targets",
        [],
    )

    if not isinstance(
        targets,
        list,
    ):
        targets = []

    signals = selected_spec.get(
        "signals",
        [],
    )

    if not isinstance(
        signals,
        list,
    ):
        signals = []

    windows = selected_spec.get(
        "windows",
        [],
    )

    if not isinstance(
        windows,
        list,
    ):
        windows = []

    splits = selected_spec.get(
        "splits",
        [],
    )

    if not isinstance(
        splits,
        list,
    ):
        splits = []

    analysis = selected_spec.get(
        "analysis",
        {},
    )

    if not isinstance(
        analysis,
        dict,
    ):
        analysis = {}

    rationale = (
        "An existing declarative research specification "
        "already represents this research question. "
        "The planner therefore proposes using that specification "
        "rather than creating a new parameter search."
    )

    if stage == "prospective_confirmation":
        rationale = (
            "A locked prospective confirmation specification "
            "already exists. The planner preserves its predefined "
            "parameters and endpoint and proposes no optimization."
        )

    return {
        "proposal_version": 1,
        "research_question": question,
        "hypothesis": hypothesis,
        "stage": stage,
        "source_specs": (
            [spec_id]
            if spec_id
            else []
        ),
        "data_requirements": {
            "windows": windows,
            "splits": splits,
            "required_signals": signals,
            "targets": targets,
        },
        "parameters": {
            "source_spec": spec_id,
            "locked": (
                _is_locked(
                    selected_spec
                )
            ),
        },
        "validation": {
            "analysis": analysis,
            "declared_by_source_spec": True,
            "research_run_count_visible": len(
                research_runs
            ),
        },
        "constraints": _build_constraints(
            selected_spec,
            stage,
        ),
        "rationale": rationale,
        "expected_observation": (
            "Determine what the declared research specification "
            "actually observes without changing its design."
        ),
        "status": "proposed",
    }


def build_experiment_plan(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Build a deterministic proposal from the research state."""

    if not isinstance(
        state,
        dict,
    ):
        raise ValueError(
            "Research state must be a JSON object."
        )

    specs = _specs(
        state
    )

    locked_specs = _locked_specs(
        state
    )

    research_runs = _research_runs(
        state
    )

    selected_spec, stage = (
        _select_existing_spec(
            specs
        )
    )

    warnings: list[str] = []

    if not research_runs:
        warnings.append(
            "No research-engine run manifests are visible "
            "in the current research-state snapshot."
        )

    migration_count = sum(
        1
        for spec in specs
        if _stage(spec) == "migration"
    )

    if migration_count:
        warnings.append(
            f"{migration_count} migration-stage specification(s) "
            "are visible but are not treated as new evidence."
        )

    if selected_spec is None:
        return {
            "planner_version": PLANNER_VERSION,
            "research_state_version": state.get(
                "state_version"
            ),
            "status": "no_existing_experiment_selected",
            "selection": {
                "source_spec": None,
                "stage": "none",
            },
            "proposal": None,
            "constraints": _build_constraints(
                None,
                "none",
            ),
            "warnings": warnings,
        }

    proposal = _build_proposal(
        selected_spec,
        stage,
        research_runs,
    )

    return {
        "planner_version": PLANNER_VERSION,
        "research_state_version": state.get(
            "state_version"
        ),
        "status": "proposal_created",
        "selection": {
            "source_spec": _spec_id(
                selected_spec
            ),
            "stage": stage,
            "locked": _is_locked(
                selected_spec
            ),
        },
        "proposal": proposal,
        "warnings": warnings,
        "research_context": {
            "spec_count": len(
                specs
            ),
            "locked_spec_count": len(
                locked_specs
            ),
            "research_run_count": len(
                research_runs
            ),
        },
    }


def main(
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """AI Lab runner entry point."""

    if not isinstance(
        spec,
        dict,
    ):
        raise ValueError(
            "Planner specification must be a JSON object."
        )

    state = spec.get(
        "research_state"
    )

    if not isinstance(
        state,
        dict,
    ):
        raise ValueError(
            "Planner requires a "
            "'research_state' object."
        )

    return build_experiment_plan(
        state
    )
