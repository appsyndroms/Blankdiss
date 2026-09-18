from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .base import ExperimentResult


def _json_default(value: Any):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (pd.Int64Dtype,)):
        return int(value)

    if hasattr(value, "item"):
        return value.item()

    return str(value)


def print_result(
    result: ExperimentResult,
) -> None:
    print()
    print("=" * 80)
    print(result.name)
    print("=" * 80)

    if result.description:
        print(result.description)

    if result.metrics:
        print()
        print("METRICS")
        print("-" * 80)

        for key, value in result.metrics.items():
            print(f"{key}: {value}")

    for name, table in result.tables.items():
        print()
        print(name.upper())
        print("-" * 80)
        print(table.to_string(index=False))

    if result.metadata:
        print()
        print("METADATA")
        print("-" * 80)

        for key, value in result.metadata.items():
            print(f"{key}: {value}")


def save_result_json(
    result: ExperimentResult,
    path: str | Path,
) -> None:
    payload = {
        "name": result.name,
        "description": result.description,
        "metrics": result.metrics,
        "metadata": result.metadata,
        "tables": {
            name: table.to_dict(orient="records")
            for name, table in result.tables.items()
        },
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )


def save_result_csv(
    result: ExperimentResult,
    directory: str | Path,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    for name, table in result.tables.items():
        filename = (
            name.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        table.to_csv(
            directory / f"{filename}.csv",
            index=False,
        )
