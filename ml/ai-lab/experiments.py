from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adaptive_config import (
    AI_LAB_RESULTS_DIR,
    ROOT,
)
from config import RUNS_DIR
from research.engine import run_spec
from research.spec import ResearchSpec
from state import (
    read_json,
    write_json,
)


def _write_experiment(
    path: Path,
    payload: dict[str, Any],
) -> None:
    write_json(
        path,
        payload,
    )


def run_experiment(
    session,
    experiment,
) -> tuple[Path, Any]:
    path = (
        AI_LAB_RESULTS_DIR
        / f"{experiment.id}.json"
    )

    payload = {
        "id": experiment.id,
        "question": experiment.question,
        "mode": experiment.mode,
        "code": experiment.code,
    }

    _write_experiment(
        path,
        payload,
    )

    parsed = read_json(
        path
    )

    if parsed is None:
        raise ValueError(
            "Experiment could not be read back: "
            f"{path}"
        )

    if parsed.get(
        "id"
    ) != payload[
        "id"
    ]:
        raise ValueError(
            "Experiment read-back changed the "
            "experiment id."
        )

    return (
        path,
        parsed,
    )


def run_research_spec(
    session,
    spec,
) -> Path:
    run_timestamp = (
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
    )

    run_dir = (
        RUNS_DIR
        / run_timestamp
    )

    result = run_spec(
        session.cache,
        spec,
    )

    result_path = (
        run_dir
        / f"{spec.id}.json"
    )

    write_json(
        result_path,
        result,
    )

    manifest = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "feature_rows": int(
            len(session.frame)
        ),
        "specs": [
            {
                "id": spec.id,
                "mode": spec.mode,
                "question": spec.question,
                "result": str(
                    result_path.relative_to(
                        ROOT
                    )
                ),
                "rows": len(
                    result.get(
                        "results",
                        [],
                    )
                ),
            }
        ],
    }

    write_json(
        run_dir
        / "manifest.json",
        manifest,
    )

    # Read the persisted result back.
    persisted = read_json(
        result_path
    )

    if persisted is None:
        raise ValueError(
            "Research result could not be read back: "
            f"{result_path}"
        )

    if persisted.get(
        "spec_id"
    ) != spec.id:
        raise ValueError(
            "Research result read-back has unexpected spec_id."
        )

    return result_path
