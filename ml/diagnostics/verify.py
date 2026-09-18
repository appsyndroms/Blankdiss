from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    base = Path(
        "data/processed/ml/research/latest"
    )

    manifest_path = base / "diagnostics.json"
    registry_path = Path(
        "ml/experiment_registry.json"
    )

    if not manifest_path.exists():
        raise SystemExit(
            f"Saknar diagnostic manifest: {manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    registry = json.loads(
        registry_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(manifest, dict):
        raise SystemExit(
            "Diagnostic manifest är inte ett objekt."
        )

    if not isinstance(registry, dict):
        raise SystemExit(
            "Experiment registry är inte ett objekt."
        )

    results = manifest.get("results")

    if not isinstance(results, list):
        raise SystemExit(
            "Diagnostic manifest saknar results."
        )

    if not results:
        raise SystemExit(
            "Inga diagnostic results hittades."
        )

    experiments = registry.get("experiments")

    if not isinstance(experiments, list):
        raise SystemExit(
            "Experiment registry saknar experiments."
        )

    required_results = {
        experiment["id"]
        for experiment in experiments
        if experiment.get("status") == "active"
    }

    actual_results = {
        result["experiment_id"]
        for result in results
    }

    missing_results = (
        required_results
        - actual_results
    )

    unexpected_results = (
        actual_results
        - required_results
    )

    if missing_results:
        raise SystemExit(
            "Saknar aktiva diagnostic results: "
            + ", ".join(
                sorted(missing_results)
            )
        )

    if unexpected_results:
        print(
            "Varning: diagnostic results innehåller "
            "experiment som inte längre är aktiva: "
            + ", ".join(
                sorted(unexpected_results)
            )
        )

    for result in results:
        experiment_id = result[
            "experiment_id"
        ]

        result_path = (
            base
            / "diagnostic"
            / f"{experiment_id}.json"
        )

        if not result_path.exists():
            raise SystemExit(
                "Saknar diagnostic result: "
                f"{result_path}"
            )

        payload = json.loads(
            result_path.read_text(
                encoding="utf-8"
            )
        )

        if payload.get("status") != "completed":
            raise SystemExit(
                "Diagnostic failed: "
                f"{experiment_id}"
            )

    completed = manifest.get(
        "completed",
        0,
    )

    failed = manifest.get(
        "failed",
        0,
    )

    if failed != 0:
        raise SystemExit(
            "Diagnostic manifest rapporterar "
            f"{failed} failed experiments."
        )

    if completed != len(required_results):
        raise SystemExit(
            "Antalet completed diagnostics "
            "matchar inte antalet aktiva experiment: "
            f"{completed} != {len(required_results)}"
        )


if __name__ == "__main__":
    main()
