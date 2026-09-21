from __future__ import annotations
import json
from pathlib import Path
from typing import Any
def _validate_payload(
    experiment_id: str,
    payload: dict[str, Any],
) -> None:
    if payload.get("status") != "completed":
        raise SystemExit(
            "Diagnostic failed: "
            f"{experiment_id}"
        )
    tables = payload.get(
        "tables"
    )
    if tables is None:
        return
    if not isinstance(tables, dict):
        raise SystemExit(
            "Diagnostic result har ogiltigt "
            f"'tables': {experiment_id}"
        )
    for table_name, table in tables.items():
        if table is None:
            continue
        if not isinstance(table, list):
            continue
        if not table:
            raise SystemExit(
                "Diagnostic result innehåller en "
                "tom tabell: "
                f"{experiment_id}/{table_name}"
            )
        for row_index, row in enumerate(table):
            if not isinstance(row, dict):
                raise SystemExit(
                    "Diagnostic table innehåller en "
                    "ogiltig rad: "
                    f"{experiment_id}/{table_name}"
                    f"[{row_index}]"
                )
    # Sector-relative analysis must contain actual
    # observations. The experiment itself also validates
    # this, but keeping the invariant here protects the
    # persisted diagnostic result from silent regressions.
    if experiment_id == "sector_relative_return":
        table = tables.get(
            "sector_relative_returns"
        )
        if not isinstance(table, list) or not table:
            raise SystemExit(
                "sector_relative_return saknar "
                "sector_relative_returns-data."
            )
        invalid_rows = []
        for row in table:
            signal_n = row.get(
                "signal_n"
            )
            control_n = row.get(
                "control_n"
            )
            if not isinstance(
                signal_n,
                (int, float),
            ) or not isinstance(
                control_n,
                (int, float),
            ):
                invalid_rows.append(
                    (
                        row.get(
                            "change_cutoff"
                        ),
                        row.get(
                            "horizon_days"
                        ),
                        signal_n,
                        control_n,
                    )
                )
                continue
            if signal_n <= 0 or control_n <= 0:
                invalid_rows.append(
                    (
                        row.get(
                            "change_cutoff"
                        ),
                        row.get(
                            "horizon_days"
                        ),
                        signal_n,
                        control_n,
                    )
                )
        if invalid_rows:
            preview = ", ".join(
                (
                    f"cutoff={cutoff}, "
                    f"horizon={horizon}d, "
                    f"signal_n={signal_n}, "
                    f"control_n={control_n}"
                )
                for cutoff, horizon, signal_n, control_n
                in invalid_rows[:5]
            )
            raise SystemExit(
                "sector_relative_return innehåller "
                "rader utan användbara observationer: "
                + preview
            )
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
        if not isinstance(
            payload,
            dict,
        ):
            raise SystemExit(
                "Diagnostic result är inte ett "
                "JSON-objekt: "
                f"{experiment_id}"
            )
        _validate_payload(
            experiment_id,
            payload,
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
