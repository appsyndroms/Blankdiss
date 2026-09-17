"""
Blankdiss experiment manager.

Runs registered research experiments, captures their output, and stores
machine-readable experiment metadata alongside raw logs.

The manager does not modify source code or repository state.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "experiments"
)

RUNS_DIR = OUTPUT_DIR / "runs"
LATEST_DIR = OUTPUT_DIR / "latest"


@dataclass(frozen=True)
class Experiment:
    name: str
    command: tuple[str, ...]
    description: str


EXPERIMENTS = (
    Experiment(
        name="ml_diagnostics",
        command=(
            sys.executable,
            "-m",
            "ml.diagnostics",
        ),
        description=(
            "Standard Blankdiss ML diagnostics."
        ),
    ),
    Experiment(
        name="fi_short_interest_event_risk_interaction",
        command=(
            sys.executable,
            "-m",
            "ml.diagnostics."
            "fi_short_interest_event_risk_interaction_diagnostic",
        ),
        description=(
            "Short-interest change × extreme event-risk interaction."
        ),
    ),
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def run_experiment(
    experiment: Experiment,
) -> dict:
    started = utc_now()

    completed = subprocess.run(
        experiment.command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
        },
    )

    finished = utc_now()

    return {
        "name": experiment.name,
        "description": experiment.description,
        "command": list(experiment.command),
        "started_at": started,
        "finished_at": finished,
        "return_code": completed.returncode,
        "success": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_sha256": sha256_text(
            completed.stdout
        ),
        "stderr_sha256": sha256_text(
            completed.stderr
        ),
    }


def write_json(
    path: Path,
    data: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def write_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        text,
        encoding="utf-8",
    )


def main() -> int:
    run_id = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    run_dir = RUNS_DIR / run_id

    results = []

    for experiment in EXPERIMENTS:
        print(
            f"=== Running {experiment.name} ===",
            flush=True,
        )

        result = run_experiment(
            experiment
        )

        results.append(result)

        write_text(
            run_dir
            / f"{experiment.name}.stdout.log",
            result["stdout"],
        )

        write_text(
            run_dir
            / f"{experiment.name}.stderr.log",
            result["stderr"],
        )

        print(
            f"{experiment.name}: "
            f"{'OK' if result['success'] else 'FAILED'}",
            flush=True,
        )

        if result["stderr"]:
            print(
                result["stderr"],
                file=sys.stderr,
                flush=True,
            )

    summary = {
        "run_id": run_id,
        "created_at": utc_now(),
        "experiment_count": len(
            results
        ),
        "successful_experiments": sum(
            result["success"]
            for result in results
        ),
        "failed_experiments": sum(
            not result["success"]
            for result in results
        ),
        "experiments": results,
    }

    write_json(
        run_dir / "experiment_run.json",
        summary,
    )

    write_json(
        LATEST_DIR / "experiment_run.json",
        summary,
    )

    print()
    print(
        f"Experiment run: {run_id}"
    )
    print(
        f"Experiments: {len(results)}"
    )
    print(
        "Successful:",
        summary["successful_experiments"],
    )
    print(
        "Failed:",
        summary["failed_experiments"],
    )

    return (
        0
        if summary["failed_experiments"] == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
