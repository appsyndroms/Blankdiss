from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.research.spec import ResearchSpec, load_spec

from config import (
    RUNS_DIR,
    SPEC_DIR,
)


EXECUTION_METADATA_KEY = "ai_lab_execution"
ENABLED_KEY = "enabled"
MODE_KEY = "mode"

SUPPORTED_MODES = {
    "once",
}


def discover_controlled_specs() -> list[ResearchSpec]:
    """
    Discover research specs explicitly opted in to AI Lab execution.

    A spec becomes executable by declaring:

        metadata:
          ai_lab_execution:
            enabled: true
            mode: once

    The AI Lab does not know or care about the spec's id.
    """

    specs: list[ResearchSpec] = []

    for path in sorted(
        SPEC_DIR.glob("*.yaml")
    ):
        spec = load_spec(
            path
        )

        execution = spec.metadata.get(
            EXECUTION_METADATA_KEY,
            {},
        )

        if not isinstance(
            execution,
            dict,
        ):
            continue

        if execution.get(
            ENABLED_KEY
        ) is not True:
            continue

        mode = str(
            execution.get(
                MODE_KEY,
                "",
            )
        ).lower()

        if mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported AI Lab execution mode "
                f"'{mode}' in spec '{spec.id}'. "
                f"Supported modes: "
                f"{sorted(SUPPORTED_MODES)}"
            )

        specs.append(
            spec
        )

    return specs


def completed_spec_ids() -> set[str]:
    """
    Find research specs that already have persisted results.

    A persisted manifest is treated as the durable execution record.
    This means mode=once survives a new Python process and does not
    depend solely on in-memory state.
    """

    completed: set[str] = set()

    if not RUNS_DIR.is_dir():
        return completed

    for manifest_path in RUNS_DIR.glob(
        "*/manifest.json"
    ):
        try:
            import json

            payload = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ):
            continue

        if not isinstance(
            payload,
            dict,
        ):
            continue

        specs = payload.get(
            "specs",
            [],
        )

        if not isinstance(
            specs,
            list,
        ):
            continue

        for item in specs:
            if not isinstance(
                item,
                dict,
            ):
                continue

            spec_id = item.get(
                "id"
            )

            if spec_id:
                completed.add(
                    str(spec_id)
                )

    return completed


def pending_controlled_specs() -> list[ResearchSpec]:
    """
    Return opted-in specs that have not yet been executed.
    """

    completed = completed_spec_ids()

    return [
        spec
        for spec
        in discover_controlled_specs()
        if spec.id not in completed
    ]


def summarize_controlled_specs(
    specs: list[ResearchSpec],
) -> list[dict[str, Any]]:
    """
    Return a small serializable description of discovered specs.
    """

    return [
        {
            "id": spec.id,
            "mode": spec.metadata.get(
                EXECUTION_METADATA_KEY,
                {},
            ).get(
                MODE_KEY
            ),
        }
        for spec in specs
    ]
