"""
AI Lab experiment runner.

Location:
    ml/ai-lab/ai_lab_runner.py

Input:
    data/ai_lab/inbox/<experiment>.json

Output:
    data/ai_lab/results/<experiment-id>/
        results.json
        report.md

The runner intentionally uses a registry of allowed experiments.
AI-generated specifications cannot execute arbitrary shell commands.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "data" / "ai_lab" / "results"


EXPERIMENT_REGISTRY: dict[str, tuple[str, str]] = {
    # Example:
    # "momentum_si_interaction": (
    #     "ml.research.momentum_si_interaction",
    #     "run",
    # ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)

    if not isinstance(spec, dict):
        raise ValueError(
            "Experiment specification must be a JSON object."
        )

    return spec


def validate_spec(spec: dict[str, Any]) -> None:
    required = [
        "experiment_id",
        "description",
        "experiment",
    ]

    missing = [key for key in required if key not in spec]

    if missing:
        raise ValueError(
            "Missing required experiment fields: "
            + ", ".join(missing)
        )

    if not isinstance(spec["experiment_id"], str):
        raise ValueError("experiment_id must be a string.")

    if not isinstance(spec["description"], str):
        raise ValueError("description must be a string.")

    if not isinstance(spec["experiment"], str):
        raise ValueError("experiment must be a string.")

    if not spec["experiment"]:
        raise ValueError("experiment must not be empty.")

    if spec["experiment"] not in EXPERIMENT_REGISTRY:
        allowed = ", ".join(sorted(EXPERIMENT_REGISTRY))

        raise ValueError(
            f"Unknown experiment '{spec['experiment']}'. "
            f"Registered experiments: {allowed or 'none'}"
        )


def safe_experiment_id(value: str) -> str:
    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "_-"
    )

    cleaned = "".join(
        character if character in allowed else "_"
        for character in value
    )

    if not cleaned:
        raise ValueError(
            "experiment_id is empty after sanitization."
        )

    return cleaned


def load_experiment_runner(experiment_name: str):
    module_name, function_name = EXPERIMENT_REGISTRY[
        experiment_name
    ]

    module = importlib.import_module(module_name)

    try:
        runner = getattr(module, function_name)
    except AttributeError as exc:
        raise ValueError(
            f"Registered experiment '{experiment_name}' "
            f"does not expose '{function_name}'."
        ) from exc

    if not callable(runner):
        raise ValueError(
            f"Registered runner '{module_name}.{function_name}' "
            "is not callable."
        )

    return runner


def run_experiment(spec: dict[str, Any]) -> dict[str, Any]:
    experiment_name = spec["experiment"]
    runner = load_experiment_runner(experiment_name)

    started = utc_now()
    result = runner(spec)
    finished = utc_now()

    if result is None:
        result = {}

    if not isinstance(result, dict):
        raise ValueError(
            "Experiment runner must return a dictionary."
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
        "experiment_id": spec["experiment_id"],
        "description": spec["description"],
        "spec": spec,
        "execution": execution,
        "runner": {
            "name": "ai_lab_runner",
            "version": 2,
        },
    }

    with (output_dir / "results.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    status = "PASS" if execution["success"] else "FAIL"

    report = f"""# AI Lab experiment

## Experiment

**ID:** `{spec["experiment_id"]}`

**Type:** `{spec["experiment"]}`

**Description:** {spec["description"]}

## Status

**{status}**

## Execution

- Started: `{execution["started_at"]}`
- Finished: `{execution["finished_at"]}`

## Result

```json
{json.dumps(
    execution.get("result", {}),
    indent=2,
    ensure_ascii=False,
)}
  with (output_dir / "report.md").open(
    "w",
    encoding="utf-8",
) as handle:
    handle.write(report)
  def main() -> int:
parser = argparse.ArgumentParser(
description=“Run a registered AI Lab experiment.”
)
  parser.add_argument(
    "spec",
    type=Path,
    help="Path to experiment JSON specification.",
)

args = parser.parse_args()
spec_path = args.spec

if not spec_path.exists():
    print(
        f"Experiment specification not found: {spec_path}",
        file=sys.stderr,
    )
    return 1

try:
    spec = load_spec(spec_path)
    validate_spec(spec)

    experiment_id = safe_experiment_id(
        spec["experiment_id"]
    )

    output_dir = RESULTS_ROOT / experiment_id

    print(
        f"Running AI Lab experiment: {experiment_id}"
    )

    print(
        f"Experiment type: {spec['experiment']}"
    )

    print(
        f"Description: {spec['description']}"
    )

    execution = run_experiment(spec)

    write_results(
        output_dir=output_dir,
        spec=spec,
        execution=execution,
    )

    print(
        f"Results written to: {output_dir}"
    )

    return 0

except Exception as exc:
    print(
        f"AI Lab runner failed: {exc}",
        file=sys.stderr,
    )
    return 1
  if name == “main”:
raise SystemExit(main())
