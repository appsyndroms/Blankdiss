from __future__ import annotations

from pathlib import Path

from config import (
    LOCKED_CONFIRMATION_ID,
    SPEC_DIR,
)
from experiments import load_yaml
from spec import load_spec


def discover_controlled_specs():
    specs = []

    for path in sorted(
        SPEC_DIR.glob("*.yaml")
    ):
        payload = load_yaml(path)

        metadata = payload.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            continue

        execution = metadata.get(
            "ai_lab_execution",
            {},
        )

        if not isinstance(
            execution,
            dict,
        ):
            continue

        if execution.get(
            "enabled"
        ) is not True:
            continue

        if str(
            payload.get("id", "")
        ) == LOCKED_CONFIRMATION_ID:
            raise ValueError(
                "Locked confirmation spec cannot "
                "be registered as an AI Lab controlled spec."
            )

        spec = load_spec(path)

        specs.append(
            (
                path,
                spec,
            )
        )

    return specs
