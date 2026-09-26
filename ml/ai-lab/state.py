from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import (
    LOCKED_CONFIRMATION_ID,
    ROOT,
    STATE_PATH,
)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def read_json(
    path: Path,
) -> dict[str, Any] | None:
    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    if not isinstance(
        payload,
        dict,
    ):
        return None

    return payload


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def read_state() -> dict[str, Any] | None:
    return read_json(
        STATE_PATH
    )


def build_state(
    phase: str,
    history: list[dict[str, Any]],
    next_candidate: tuple[
        float,
        float,
        str,
    ] | None,
    extension_family: str | None = None,
) -> dict[str, Any]:
    return {
        "state_version": 1,
        "updated_at_utc": utc_now(),
        "phase": phase,
        "history": history,
        "next_candidate": (
            {
                "baseline_fraction": (
                    next_candidate[0]
                ),
                "incremental_fraction": (
                    next_candidate[1]
                ),
                "target": (
                    next_candidate[2]
                ),
            }
            if next_candidate
            else None
        ),
        "extension_family": extension_family,
        "locked_boundary": {
            "confirmation_spec": (
                LOCKED_CONFIRMATION_ID
            ),
            "confirmation_may_not_be_executed_or_modified": True,
        },
    }


def write_state(
    payload: dict[str, Any],
) -> None:
    write_json(
        STATE_PATH,
        payload,
    )
