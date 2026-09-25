"""AI Lab experiment runner.

Location:
    ml/ai-lab/ai_lab_runner.py

Input:
    data/ai_lab/inbox/<experiment>.json

Output:
    data/ai_lab/results/<experiment-id>/
        results.json
        report.md

The runner uses an allowlist of registered experiments.
AI-generated specifications cannot execute arbitrary shell commands.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_ROOT = (
    ROOT
    / "data"
    / "ai_lab"
    / "results"
)


def smoke_test(
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Minimal end-to-end test for the AI Lab execution pipeline."""
    return {
        "message": (
            "AI Lab smoke test completed "
            "successfully."
        ),
        "experiment_id": spec[
            "experiment_id"
        ],
        "runner": (
            "ml/ai-lab/ai_lab_runner.py"
        ),
    }


EXPERIMENT_REGISTRY: dict[
    str,
    tuple[str, str],
] = {
    "smoke_test": (
        __name__,
        "smoke_test",
    ),
    "research_state": (
        "@file:ml/ai-lab/research_state.py",
        "build_research_state",
    ),
    "experiment_planner": (
        "@file:ml/ai-lab/experiment_planner.py",
        "main",
    ),
}


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def load_spec(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        spec = json.load(handle)

    if not isinstance(
        spec,
        dict,
    ):
        raise ValueError(
            "Experiment specification "
            "must be a JSON object."
        )

    return spec


def validate_spec(
    spec: dict[str, Any],
) -> None:
    required = [
        "experiment_id",
        "description",
        "experiment",
    ]

    missing = [
        key
        for key in required
        if key not in spec
    ]

    if missing:
        raise ValueError(
            "Missing required experiment fields: "
            + ", ".join(missing)
        )

    if not isinstance(
        spec["experiment_id"],
        str,
    ):
        raise ValueError(
            "experiment_id must be a string."
        )

    if not isinstance(
        spec["description"],
        str,
    ):
        raise ValueError(
            "description must be a string."
        )

    if not isinstance(
        spec["experiment"],
        str,
    ):
        raise ValueError(
            "experiment must be a string."
        )

    if not spec["experiment"]:
        raise ValueError(
            "experiment must not be empty."
        )

    if (
        spec["experiment"]
        not in EXPERIMENT_REGISTRY
    ):
        allowed = ", ".join(
            sorted(
                EXPERIMENT_REGISTRY
            )
        )

        raise ValueError(
            f"Unknown experiment "
            f"'{spec['experiment']}'. "
            f"Registered experiments: "
            f"{allowed or 'none'}"
        )


def load_module_from_file(
    file_reference: str,
) -> ModuleType:
    """Load a Python module from a repository-relative file path.

    File references use the format:

        @file:path/to/module.py

    This is needed for AI Lab modules because the directory
    ml/ai-lab contains a hyphen and therefore cannot be imported
    as a normal Python package name.
    """
    if not file_reference.startswith(
        "@file:"
    ):
        raise ValueError(
            "Invalid file module reference: "
            f"{file_reference}"
        )

    relative_path = file_reference[
        len("@file:") :
    ]

    module_path = (
        ROOT
        / relative_path
    ).resolve()

    try:
        module_path.relative_to(
            ROOT
        )
    except ValueError as exc:
        raise ValueError(
            "Module path must remain "
            "inside the repository: "
            f"{relative_path}"
        ) from exc

    if not module_path.is_file():
        raise ValueError(
            "Registered module file does "
            "not exist: "
            f"{relative_path}"
        )

    module_name = (
        "blankdiss_ai_lab_"
        + module_path.stem
    )

    module_spec = (
        importlib.util.spec_from_file_location(
            module_name,
            module_path,
        )
    )

    if module_spec is None:
        raise ValueError(
            "Could not create import "
            "specification for: "
            f"{relative_path}"
        )

    if module_spec.loader is None:
        raise ValueError(
            "Could not create module "
            "loader for: "
            f"{relative_path}"
        )

    module = (
        importlib.util.module_from_spec(
            module_spec
        )
    )

    sys.modules[
        module_name
    ] = module

    module_spec.loader.exec_module(
        module
    )

    return module


def load_registered_module(
    module_name: str,
) -> ModuleType:
    """Load a registered module safely.

    Normal Python modules use importlib.

    AI Lab modules located below ml/ai-lab can use the
    explicit @file: form because that directory contains
    a hyphen and is not a valid Python package name.
    """
    if module_name.startswith(
        "@file:"
    ):
        return load_module_from_file(
            module_name
        )

    return importlib.import_module(
        module_name
    )


def load_experiment_runner(
    experiment_name: str,
) -> Callable[..., Any]:
    module_name, function_name = (
        EXPERIMENT_REGISTRY[
            experiment_name
        ]
    )

    module = load_registered_module(
        module_name
    )

    try:
        runner = getattr(
            module,
            function_name,
        )
    except AttributeError as exc:
        raise ValueError(
            f"Registered experiment "
            f"'{experiment_name}' does not "
            f"expose '{function_name}'."
        ) from exc

    if not callable(runner):
        raise ValueError(
            f"Registered runner "
            f"'{module_name}.{function_name}' "
            "is not callable."
        )

    return runner


def run_experiment(
    spec: dict[str, Any],
) -> dict[str, Any]:
    experiment_name = spec[
        "experiment"
    ]

    runner = load_experiment_runner(
        experiment_name
    )

    started = utc_now()

    result = runner(spec)

    finished = utc_now()

    if result is None:
        result = {}

    if not isinstance(
        result,
        dict,
    ):
        raise ValueError(
            "Experiment runner must "
            "return a dictionary."
        )

    return {
        "started_at": started,
        "finished_at": finished,
        "success": True,
        "experiment": experiment_name,
        "result": result,
    }


def write_results(
    output_dir: Path,
    spec: dict[str, Any],
    execution: dict[str, Any],
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "experiment_id": spec[
            "experiment_id"
        ],
        "description": spec[
            "description"
        ],
        "spec": spec,
        "execution": execution,
        "runner": {
            "name": "ai_lab_runner",
            "version": 6,
        },
    }

    with (
        output_dir / "results.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    status = (
        "PASS"
        if execution["success"]
        else "FAIL"
    )

    result_json = json.dumps(
        execution.get(
            "result",
            {},
        ),
        indent=2,
        ensure_ascii=False,
    )

    report = "\n".join(
        [
            "# AI Lab experiment",
            "",
            "## Experiment",
            "",
            f'**ID:** '
            f'`{spec["experiment_id"]}`',
            "",
            f'**Type:** '
            f'`{spec["experiment"]}`',
            "",
            f'**Description:** '
            f'{spec["description"]}',
            "",
            "## Status",
            "",
            f"**{status}**",
            "",
            "## Execution",
            "",
            f'- Started: '
            f'`{execution["started_at"]}`',
            f'- Finished: '
            f'`{execution["finished_at"]}`',
            "",
            "## Result",
            "",
            "```json",
            result_json,
            "```",
            "",
        ]
    )

    (
        output_dir / "report.md"
    ).write_text(
        report,
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a registered AI Lab "
            "experiment."
        )
    )

    parser.add_argument(
        "spec",
        type=Path,
        help=(
            "Path to the experiment "
            "JSON specification."
        ),
    )

    args = parser.parse_args()

    spec = load_spec(
        args.spec
    )

    validate_spec(
        spec
    )

    experiment_id = (
        safe_experiment_id(
            spec["experiment_id"]
        )
    )

    execution = run_experiment(
        spec
    )

    output_dir = (
        RESULTS_ROOT
        / experiment_id
    )

    write_results(
        output_dir,
        spec,
        execution,
    )

    print(
        json.dumps(
            {
                "experiment_id": (
                    spec["experiment_id"]
                ),
                "experiment": (
                    spec["experiment"]
                ),
                "success": (
                    execution["success"]
                ),
                "output_dir": str(
                    output_dir
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    return 0


def safe_experiment_id(
    value: str,
) -> str:
    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "_-"
    )

    cleaned = "".join(
        character
        if character in allowed
        else "_"
        for character in value
    )

    if not cleaned:
        raise ValueError(
            "experiment_id is empty "
            "after sanitization."
        )

    return cleaned


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
