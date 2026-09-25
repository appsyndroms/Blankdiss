"""Build a controlled snapshot of the current Blankdiss research state.

The AI Lab should reason from an explicit research state rather than
discovering repository structure ad hoc.

This module is intentionally read-only.

It collects:
    - research specifications
    - AI Lab experiment results
    - research-engine spec-run results
    - locked/prospective specifications

It does not:
    - execute experiments
    - modify specifications
    - modify research results
    - select parameters
    - optimize hypotheses
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

RESEARCH_SPEC_DIR = (
    ROOT
    / "ml"
    / "research"
    / "specs"
)

AI_LAB_RESULTS_DIR = (
    ROOT
    / "data"
    / "ai_lab"
    / "results"
)

RESEARCH_RUNS_DIR = (
    ROOT
    / "data"
    / "processed"
    / "ml"
    / "research"
    / "spec_runs"
)


def _relative(path: Path) -> str:
    """Return a repository-relative POSIX path."""
    return path.relative_to(ROOT).as_posix()


def _read_text(
    path: Path,
) -> str:
    return path.read_text(
        encoding="utf-8"
    )


def _read_json(
    path: Path,
) -> dict[str, Any] | None:
    try:
        payload = json.loads(
            _read_text(path)
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    if not isinstance(payload, dict):
        return None

    return payload


def _load_yaml_metadata(
    path: Path,
) -> dict[str, Any]:
    """Extract useful metadata from a research YAML spec."""
    try:
        import yaml

        payload = yaml.safe_load(
            _read_text(path)
        )
    except (
        OSError,
        Exception,
    ):
        return {}

    if not isinstance(payload, dict):
        return {}

    metadata = payload.get(
        "metadata",
        {},
    )

    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "id": payload.get("id"),
        "question": payload.get("question"),
        "mode": payload.get("mode"),
        "analysis": payload.get("analysis"),
        "targets": payload.get("targets"),
        "signals": payload.get("signals"),
        "windows": payload.get("windows"),
        "splits": payload.get("splits"),
        "metadata": metadata,
    }


def collect_research_specs() -> list[dict[str, Any]]:
    """Collect all declarative research specifications."""
    if not RESEARCH_SPEC_DIR.exists():
        return []

    specs: list[dict[str, Any]] = []

    for path in sorted(
        RESEARCH_SPEC_DIR.glob("*.yaml")
    ):
        metadata = _load_yaml_metadata(
            path
        )

        specs.append(
            {
                "path": _relative(path),
                "id": metadata.get("id"),
                "question": metadata.get(
                    "question"
                ),
                "mode": metadata.get("mode"),
                "analysis": metadata.get(
                    "analysis"
                ),
                "targets": metadata.get(
                    "targets"
                ),
                "signals": metadata.get(
                    "signals"
                ),
                "windows": metadata.get(
                    "windows"
                ),
                "splits": metadata.get(
                    "splits"
                ),
                "metadata": metadata.get(
                    "metadata",
                    {},
                ),
                "locked": bool(
                    metadata.get(
                        "metadata",
                        {},
                    ).get(
                        "locked",
                        False,
                    )
                ),
                "stage": metadata.get(
                    "metadata",
                    {},
                ).get(
                    "stage"
                ),
            }
        )

    return specs


def collect_ai_lab_results() -> list[dict[str, Any]]:
    """Collect completed AI Lab experiment results."""
    if not AI_LAB_RESULTS_DIR.exists():
        return []

    experiments: list[dict[str, Any]] = []

    for result_path in sorted(
        AI_LAB_RESULTS_DIR.glob(
            "*/results.json"
        )
    ):
        payload = _read_json(
            result_path
        )

        if payload is None:
            continue

        execution = payload.get(
            "execution",
            {},
        )

        if not isinstance(
            execution,
            dict,
        ):
            execution = {}

        experiments.append(
            {
                "path": _relative(
                    result_path
                ),
                "experiment_id": payload.get(
                    "experiment_id"
                ),
                "description": payload.get(
                    "description"
                ),
                "success": execution.get(
                    "success"
                ),
                "experiment": execution.get(
                    "experiment"
                ),
                "started_at": execution.get(
                    "started_at"
                ),
                "finished_at": execution.get(
                    "finished_at"
                ),
            }
        )

    return experiments


def collect_research_runs() -> list[dict[str, Any]]:
    """Collect research-engine run manifests."""
    if not RESEARCH_RUNS_DIR.exists():
        return []

    runs: list[dict[str, Any]] = []

    for manifest_path in sorted(
        RESEARCH_RUNS_DIR.glob(
            "*/manifest.json"
        )
    ):
        payload = _read_json(
            manifest_path
        )

        if payload is None:
            continue

        specs = payload.get(
            "specs",
            [],
        )

        if not isinstance(
            specs,
            list,
        ):
            specs = []

        runs.append(
            {
                "path": _relative(
                    manifest_path
                ),
                "created_at_utc": payload.get(
                    "created_at_utc"
                ),
                "feature_rows": payload.get(
                    "feature_rows"
                ),
                "spec_count": len(specs),
                "specs": specs,
            }
        )

    return runs


def identify_locked_research(
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return research specifications marked as locked."""
    return [
        spec
        for spec in specs
        if spec.get("locked") is True
    ]


def identify_open_questions(
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return questions represented by current research specs."""
    questions: list[dict[str, Any]] = []

    for spec in specs:
        question = spec.get(
            "question"
        )

        if not question:
            continue

        questions.append(
            {
                "spec_id": spec.get(
                    "id"
                ),
                "question": question,
                "stage": spec.get(
                    "stage"
                ),
                "locked": spec.get(
                    "locked",
                    False,
                ),
            }
        )

    return questions


def build_research_state() -> dict[str, Any]:
    """Build the complete read-only research state."""
    specs = collect_research_specs()
    ai_lab_results = collect_ai_lab_results()
    research_runs = collect_research_runs()

    locked_specs = identify_locked_research(
        specs
    )

    open_questions = identify_open_questions(
        specs
    )

    return {
        "state_version": 1,
        "purpose": (
            "Controlled research-state snapshot "
            "for Blankdiss AI Lab."
        ),
        "repository": "appsyndroms/Blankdiss",
        "research": {
            "spec_count": len(specs),
            "specs": specs,
            "locked_spec_count": len(
                locked_specs
            ),
            "locked_specs": [
                {
                    "id": spec.get("id"),
                    "path": spec.get("path"),
                    "stage": spec.get("stage"),
                    "question": spec.get(
                        "question"
                    ),
                }
                for spec in locked_specs
            ],
            "open_questions": open_questions,
        },
        "ai_lab": {
            "result_count": len(
                ai_lab_results
            ),
            "results": ai_lab_results,
        },
        "research_runs": {
            "run_count": len(
                research_runs
            ),
            "runs": research_runs,
        },
    }


def render_research_state_report(
    state: dict[str, Any],
) -> str:
    """Render a human-readable research-state report."""
    research = state["research"]
    ai_lab = state["ai_lab"]
    research_runs = state[
        "research_runs"
    ]

    lines = [
        "# Blankdiss AI Lab — Research State",
        "",
        "## Overview",
        "",
        f"- Research specs: "
        f"{research['spec_count']}",
        f"- Locked specs: "
        f"{research['locked_spec_count']}",
        f"- AI Lab results: "
        f"{ai_lab['result_count']}",
        f"- Research runs: "
        f"{research_runs['run_count']}",
        "",
        "## Locked research",
        "",
    ]

    locked_specs = research[
        "locked_specs"
    ]

    if locked_specs:
        for spec in locked_specs:
            lines.extend(
                [
                    f"### `{spec['id']}`",
                    "",
                    f"- Path: `{spec['path']}`",
                    f"- Stage: `{spec['stage']}`",
                    f"- Question: "
                    f"{spec['question']}",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "No locked research specifications "
                "were found.",
                "",
            ]
        )

    lines.extend(
        [
            "## Current research questions",
            "",
        ]
    )

    questions = research[
        "open_questions"
    ]

    if questions:
        for question in questions:
            lines.extend(
                [
                    f"### `{question['spec_id']}`",
                    "",
                    question["question"],
                    "",
                    f"- Stage: "
                    f"{question['stage']}",
                    f"- Locked: "
                    f"{question['locked']}",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "No research questions found.",
                "",
            ]
        )

    lines.extend(
        [
            "## AI Lab experiments",
            "",
        ]
    )

    results = ai_lab[
        "results"
    ]

    if results:
        for result in results:
            lines.extend(
                [
                    f"### `{result['experiment_id']}`",
                    "",
                    f"- Experiment: "
                    f"`{result['experiment']}`",
                    f"- Success: "
                    f"`{result['success']}`",
                    f"- Result: "
                    f"`{result['path']}`",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "No completed AI Lab experiments "
                "were found.",
                "",
            ]
        )

    lines.extend(
        [
            "## Research-engine runs",
            "",
        ]
    )

    if research_runs:
        for run in research_runs:
            lines.extend(
                [
                    f"### `{run['path']}`",
                    "",
                    f"- Created: "
                    f"`{run['created_at_utc']}`",
                    f"- Feature rows: "
                    f"`{run['feature_rows']}`",
                    f"- Specs: "
                    f"`{run['spec_count']}`",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "No research-engine runs found.",
                "",
            ]
        )

    lines.extend(
        [
            "## AI Lab boundary",
            "",
            "This snapshot is descriptive only.",
            "",
            "It does not select parameters, "
            "modify locked experiments, or "
            "declare a hypothesis confirmed.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """CLI entry point for generating a state snapshot."""
    state = build_research_state()

    print(
        json.dumps(
            state,
            indent=2,
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
