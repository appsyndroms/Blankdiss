"""Lagring av FI-rådata och manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .config import (
    MANIFEST_PATH,
    RAW_DIR,
    SNAPSHOT_DIR,
)
from .errors import FIError
from .normalize import now_stockholm


def sha256_bytes(
    data: bytes,
) -> str:

    return hashlib.sha256(
        data
    ).hexdigest()


def load_manifest() -> dict:

    if not MANIFEST_PATH.exists():
        return {
            "source": "FI",
            "dataset": "aggregate_short_positions",
            "last_checked": None,
            "files": {},
        }

    try:
        return json.loads(
            MANIFEST_PATH.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise FIError(
            "Kunde inte läsa FI-manifestet."
        ) from exc


def save_manifest(
    manifest: dict,
) -> None:

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_snapshot(
    records: list[dict],
) -> Path:

    SNAPSHOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if records and records[0].get(
        "fetched_at"
    ):
        fetched = datetime.fromisoformat(
            records[0]["fetched_at"]
        )
    else:
        fetched = now_stockholm()

    timestamp = fetched.strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    path = (
        SNAPSHOT_DIR
        / (
            "fi_aggregate_"
            f"{timestamp}.jsonl"
        )
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    return path


def write_source(
    data: bytes,
    extension: str,
    source_date: str,
) -> Path:

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SOURCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    digest = sha256_bytes(data)

    filename = (
        f"aggregate_"
        f"{source_date}_"
        f"{digest[:12]}"
        f"{extension}"
    )

    path = SOURCE_DIR / filename

    if not path.exists():
        path.write_bytes(data)

    return path
