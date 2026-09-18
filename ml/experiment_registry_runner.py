from __future__ import annotations

import importlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.config import WALK_FORWARD_WINDOWS
from ml.dataset import load_features
from ml.diagnostics.framework import (
    DiagnosticExperiment,
    ExperimentContext,
    ExperimentResult,
)


ROOT = Path(__file__).resolve().parents[1]

REGISTRY_PATH = (
    ROOT
    / "ml"
    / "experiment_registry.json"
)

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
            "Experiment registry saknar en lista "
            "'experiments'."
        )

    return experiments


def _run_timestamp() -> str:
    """
    Använd samma timestamp som research-runnern när
    den finns.
    """
    metadata_path = (
        LATEST_DIR
        / "metadata.json"
    )

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


def _load_experiment(
    module_name: str,
) -> DiagnosticExperiment:
    module = importlib.import_module(
        module_name
    )

    experiment_classes = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(
            value,
            DiagnosticExperiment,
        )
        and value is not DiagnosticExperiment
    ]

    if not experiment_classes:
        raise RuntimeError(
            f"Modulen {module_name} saknar en "
            "DiagnosticExperiment-klass."
        )

    if len(experiment_classes) > 1:
        names = ", ".join(
            cls.__name__
            for cls in experiment_classes
        )

        raise RuntimeError(
            f"Modulen {module_name} innehåller flera "
            "DiagnosticExperiment-klasser: "
            f"{names}"
        )

    return experiment_classes[0]()


def _run_experiment(
    experiment: dict,
    features,
    output_dir: Path,
) -> tuple[bool, dict[str, Any]]:
    experiment_id = experiment["id"]
    module_name = experiment["module"]

    started_at = datetime.now(
        timezone.utc
    ).isoformat()

    results = []

    try:
        instance = _load_experiment(
            module_name
        )

        for window_index, window in enumerate(
            WALK_FORWARD_WINDOWS,
            start=1,
        ):
            context = ExperimentContext(
                data=features,
                train_end=window.train_end,
                validation_end=window.validation_end,
                test_end=window.test_end,
            )

            result = instance.execute(
                context
            )

            window_result = {
                "window": (
                    f"window_{window_index}"
                ),
                "train_end": window.train_end,
                "validation_end": (
                    window.validation_end
                ),
                "test_end": window.test_end,
                "result": result,
            }

            results.append(
                window_result
            )

        finished_at = datetime.now(
            timezone.utc
        ).isoformat()

        payload = {
            "experiment_id": experiment_id,
            "question": experiment.get(
                "question",
                "",
            ),
            "module": module_name,
            "status": "completed",
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "windows": results,
        }

        _write_result(
            payload,
            output_dir
            / f"{experiment_id}.json",
        )

        return True, payload

    except Exception as exc:
        finished_at = datetime.now(
            timezone.utc
        ).isoformat()

        payload = {
            "experiment_id": experiment_id,
            "question": experiment.get(
                "question",
                "",
            ),
            "module": module_name,
            "status": "failed",
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }

        _write_result(
            payload,
            output_dir
            / f"{experiment_id}.json",
        )

        return False, payload


def _serialise_result(
    result: ExperimentResult,
) -> dict[str, Any]:
    """
    Konverterar ExperimentResult till JSON-kompatibelt
    format.
    """
    return {
        "name": result.name,
        "description": result.description,
        "metrics": result.metrics,
        "metadata": result.metadata,
        "tables": {
            name: table.to_dict(
                orient="records"
            )
            for name, table in result.tables.items()
        },
    }


def _write_result(
    payload: dict[str, Any],
    path: Path,
) -> None:
    """
    Serialiserar ExperimentResult-objekt i payload.
    """

    def convert(value):
        if isinstance(
            value,
            dict,
        ):
            return {
                key: convert(item)
                for key, item in value.items()
            }

        if isinstance(
            value,
            list,
        ):
            return [
                convert(item)
                for item in value
            ]

        if isinstance(
            value,
            tuple,
        ):
            return [
                convert(item)
                for item in value
            ]

        if hasattr(
            value,
            "to_pydatetime",
        ):
            return value.isoformat()

        if hasattr(
            value,
            "item",
        ):
            try:
                return value.item()
            except Exception:
                pass

        if isinstance(
            value,
            ExperimentResult,
        ):
            return _serialise_result(
                value
            )

        return value

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            convert(payload),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


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
        if experiment.get("status")
        == "active"
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
        f"Registered experiments: "
        f"{len(experiments):,}",
        flush=True,
    )

    print(
        f"Active experiments: "
        f"{len(active):,}",
        flush=True,
    )

    if not active:
        print(
            "No active registered experiments.",
            flush=True,
        )
        return

    print(
        "Loading features...",
        flush=True,
    )

    features = load_features()

    print(
        f"Loaded {len(features):,} feature rows",
        flush=True,
    )

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

        success, payload = _run_experiment(
            experiment=experiment,
            features=features,
            output_dir=timestamp_dir,
        )

        results.append(
            {
                "experiment_id": experiment_id,
                "status": payload["status"],
                "result_file": (
                    f"diagnostic/"
                    f"{experiment_id}.json"
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

            error = payload.get(
                "error"
            )

            if error:
                print(
                    f"  {error['type']}: "
                    f"{error['message']}",
                    flush=True,
                )

    manifest = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "research_run_timestamp": (
            run_timestamp
        ),
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

    _write_result(
        manifest,
        timestamp_dir.parent
        / "diagnostics.json",
    )

    _copy_directory(
        timestamp_dir,
        latest_dir,
    )

    _write_result(
        manifest,
        LATEST_DIR
        / "diagnostics.json",
    )

    print()
    print(
        f"Diagnostic results: "
        f"{len(results):,}",
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
        f"Results: "
        f"{LATEST_DIR / 'diagnostics.json'}",
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
