"""Lagring av FI-rådata och snapshots."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .config import (
    MANIFEST_PATH,
    RAW_DIR,
    SNAPSHOT_DIR,
    SOURCE_DIR,
)
from .errors import FIError
from .normalize import now_stockholm


def sha256_bytes(
    data: bytes,
) -> str:
    """Beräknar SHA-256 för rådata."""

    return hashlib.sha256(
        data
    ).hexdigest()


def load_manifest() -> dict:
    """Läser manifestet eller skapar ett tomt."""

    if not MANIFEST_PATH.exists():
        return {
            "source": "FI",
            "dataset": (
                "aggregate_short_positions"
            ),
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
    """Skriver manifestet."""

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
    """
    Skriver en tidsstämplad JSONL-snapshot.

    Varje körning får ett eget filnamn.
    """

    if not records:
        raise FIError(
            "Kan inte skriva tom FI-snapshot."
        )

    SNAPSHOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fetched_value = records[0].get(
        "fetched_at"
    )

    if fetched_value:
        try:
            fetched = datetime.fromisoformat(
                fetched_value
            )
        except ValueError as exc:
            raise FIError(
                "Ogiltig fetched_at i FI-data: "
                f"{fetched_value}"
            ) from exc
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
    """
    Sparar en rå FI-fil med innehållsbaserat namn.

    Funktionen används inte av den aktuella
    HTML-baserade hämtningen men finns kvar för
    framtida FI-filer.
    """

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SOURCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    digest = sha256_bytes(
        data
    )

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
