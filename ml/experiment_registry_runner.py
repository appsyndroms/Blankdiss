from __future__ import annotations

import contextlib
import importlib
import io
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REGISTRY_PATH = ROOT / "ml" / "experiment_registry.json"

RESEARCH_DIR = (
    ROOT
    / "data"
    / "processed"
    / "ml"
    / "research"
)

LATEST_DIR = RESEARCH_DIR / "latest"


def load_registry() -> list[dict]:
    registry = json.loads(
        REGISTRY_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(registry, dict):
        raise ValueError(
            "Experiment registry måste vara ett objekt."
        )

    experiments = registry.get("experiments")

    if not isinstance(experiments, list):
        raise ValueError(
            "Experiment registry saknar en lista 'experiments'."
        )

    return experiments


def _run_timestamp() -> str:
    """
    Use the timestamp already created by the main research runner
    when available. This keeps generic research and diagnostics
    belonging to the same workflow run together.
    """
    metadata_path = LATEST_DIR / "metadata.json"

    if metadata_path.exists():
        try:
            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            created_at = metadata.get(
                "created_at_utc"
            )

            if created_at:
                return str(created_at)
        except Exception:
            pass

    return datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )


def _diagnostic_payload(
    experiment: dict,
    status: str,
    output: str,
    started_at: str,
    finished_at: str,
) -> dict:
    """
    Store the complete diagnostic stdout as machine-readable JSON.

    The diagnostic itself remains responsible for its statistical
    calculations. We deliberately do not parse/recalculate its
    statistics here.
    """
    return {
        "experiment_id": experiment["id"],
        "question": experiment.get(
            "question",
            "",
        ),
        "module": experiment["module"],
        "status": status,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "output": output,
    }


def _write_json(
    path: Path,
    value: object,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run_experiment(
    experiment: dict,
    output_dir: Path,
) -> tuple[bool, dict]:
    experiment_id = experiment["id"]
    module_name = experiment["module"]

    started_at = datetime.now(
        timezone.utc
    ).isoformat()

    buffer = io.StringIO()

    try:
        module = importlib.import_module(
            module_name
        )

        run = getattr(
            module,
            "main",
            None
        )

        if not callable(run):
            raise RuntimeError(
                f"Experiment module {module_name} "
                "saknar main()."
            )

        # Diagnostics can remain verbose internally.
        # Their output is captured and written to a result file
        # instead of flooding the Actions log.
        with contextlib.redirect_stdout(
            buffer
        ):
            with contextlib.redirect_stderr(
                buffer
            ):
                run()

        status = "completed"
        success = True

    except Exception as exc:
        buffer.write(
            f"\nERROR: {type(exc).__name__}: {exc}\n"
        )

        status = "failed"
        success = False

    finished_at = datetime.now(
        timezone.utc
    ).isoformat()

    payload = _diagnostic_payload(
        experiment=experiment,
        status=status,
        output=buffer.getvalue(),
        started_at=started_at,
        finished_at=finished_at,
    )

    result_path = (
        output_dir
        / f"{experiment_id}.json"
    )

    _write_json(
        result_path,
        payload,
    )

    return success, payload


def _copy_directory(
    source: Path,
    destination: Path,
) -> None:
    if destination.exists():
        shutil.rmtree(
            destination
        )

    shutil.copytree(
        source,
        destination,
    )


def main() -> None:
    experiments = load_registry()

    active = [
        experiment
        for experiment in experiments
        if experiment.get("status") == "active"
    ]

    active.sort(
        key=lambda experiment: (
            experiment.get(
                "priority",
                9999,
            ),
            experiment.get(
                "id",
                "",
            ),
        )
    )

    print(
        f"Registered experiments: {len(experiments):,}",
        flush=True,
    )

    print(
        f"Active experiments: {len(active):,}",
        flush=True,
    )

    if not active:
        print(
            "No active registered experiments.",
            flush=True,
        )
        return

    run_timestamp = _run_timestamp()

    timestamp_dir = (
        RESEARCH_DIR
        / run_timestamp
        / "diagnostic"
    )

    latest_dir = (
        LATEST_DIR
        / "diagnostic"
    )

    timestamp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = 0
    failed: list[str] = []
    results: list[dict] = []

    for experiment in active:
        experiment_id = experiment["id"]

        print(
            f"Running: {experiment_id}",
            flush=True,
        )

        success, payload = run_experiment(
            experiment=experiment,
            output_dir=timestamp_dir,
        )

        results.append(
            {
                "experiment_id": experiment_id,
                "status": payload["status"],
                "result_file": (
                    f"diagnostic/{experiment_id}.json"
                ),
            }
        )

        if success:
            completed += 1
            print(
                f"Completed: {experiment_id}",
                flush=True,
            )
        else:
            failed.append(
                experiment_id
            )
            print(
                f"Failed: {experiment_id}",
                flush=True,
            )

    manifest = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "research_run_timestamp": run_timestamp,
        "registered_experiments": len(
            experiments
        ),
        "active_experiments": len(
            active
        ),
        "completed": completed,
        "failed": len(failed),
        "results": results,
    }

    _write_json(
        timestamp_dir.parent
        / "diagnostics.json",
        manifest,
    )

    # Mirror diagnostics into latest/, exactly like the main
    # research runner does for its result files.
    _copy_directory(
        timestamp_dir,
        latest_dir,
    )

    _write_json(
        LATEST_DIR / "diagnostics.json",
        manifest,
    )

    print()
    print(
        f"Diagnostic results: {len(results):,}",
        flush=True,
    )

    print(
        f"Completed: {completed:,}",
        flush=True,
    )

    print(
        f"Failed: {len(failed):,}",
        flush=True,
    )

    print(
        f"Results: {LATEST_DIR / 'diagnostics.json'}",
        flush=True,
    )

    if failed:
        print(
            "Failed experiments:",
            flush=True,
        )

        for experiment_id in failed:
            print(
                f"  - {experiment_id}",
                flush=True,
            )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
