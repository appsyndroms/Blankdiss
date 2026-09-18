from __future__ import annotations

import importlib
import json
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REGISTRY_PATH = (
    ROOT
    / "ml"
    / "experiment_registry.json"
)


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


def run_experiment(experiment: dict) -> bool:
    experiment_id = experiment["id"]
    module_name = experiment["module"]

    print()
    print("=" * 80)
    print(
        f"REGISTERED EXPERIMENT: {experiment_id}"
    )
    print("=" * 80)
    print(
        f"Question: {experiment.get('question', '')}"
    )
    print(
        f"Module:   {module_name}"
    )

    try:
        module = importlib.import_module(
            module_name
        )
    except Exception:
        print(
            f"Could not import {module_name}"
        )
        traceback.print_exc()
        return False

    run = getattr(
        module,
        "main",
        None,
    )

    if not callable(run):
        print(
            f"Experiment module {module_name} "
            "saknar main()."
        )
        return False

    try:
        run()
    except Exception:
        print(
            f"Experiment {experiment_id} failed."
        )
        traceback.print_exc()
        return False

    print()
    print(
        f"Experiment {experiment_id} completed."
    )

    return True


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
        f"Registered experiments: "
        f"{len(experiments)}"
    )
    print(
        f"Active experiments: "
        f"{len(active)}"
    )

    if not active:
        print(
            "No active registered experiments."
        )
        return

    failed: list[str] = []

    for experiment in active:
        success = run_experiment(
            experiment
        )

        if not success:
            failed.append(
                experiment["id"]
            )

    print()
    print("=" * 80)
    print("REGISTERED EXPERIMENT SUMMARY")
    print("=" * 80)

    print(
        f"Completed: "
        f"{len(active) - len(failed)}"
    )
    print(
        f"Failed:    "
        f"{len(failed)}"
    )

    if failed:
        print()
        print("Failed experiments:")

        for experiment_id in failed:
            print(
                f"  - {experiment_id}"
            )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
