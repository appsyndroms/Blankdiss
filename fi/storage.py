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
            "snapshots": [],
        }

    try:
        manifest = json.loads(
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

    if not isinstance(manifest, dict):
        raise FIError(
            "FI-manifestet har ogiltigt format."
        )

    snapshots = manifest.get("snapshots")

    if not isinstance(snapshots, list):
        manifest["snapshots"] = []

    return manifest


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

    Varje körning får ett eget filnamn och manifestet
    uppdateras så att FI-historiken kan följas utan att
    tidigare snapshots skrivs över.
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
        fetched_value = fetched.isoformat(
            timespec="seconds"
        )

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

    if path.exists():
        raise FIError(
            "FI-snapshoten finns redan: "
            f"{path}"
        )

    lines = [
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
        for record in records
    ]

    data = "".join(lines).encode("utf-8")

    path.write_bytes(data)

    digest = sha256_bytes(data)

    source_dates = sorted(
        {
            str(record["source_date"])
            for record in records
            if record.get("source_date")
        }
    )

    manifest = load_manifest()

    snapshots = manifest.setdefault(
        "snapshots",
        [],
    )

    relative_path = path.relative_to(
        RAW_DIR
    ).as_posix()

    snapshot_entry = {
        "file": relative_path,
        "fetched_at": fetched_value,
        "source_dates": source_dates,
        "observations": len(records),
        "sha256": digest,
    }

    existing_files = {
        item.get("file")
        for item in snapshots
        if isinstance(item, dict)
    }

    if relative_path not in existing_files:
        snapshots.append(snapshot_entry)

    snapshots.sort(
        key=lambda item: item.get(
            "fetched_at",
            "",
        )
    )

    manifest["last_checked"] = fetched_value
    manifest["source"] = "FI"
    manifest["dataset"] = (
        "aggregate_short_positions"
    )

    save_manifest(manifest)

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
