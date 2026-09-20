"""Kontrollerar om feature-källorna har förändrats."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from analysis.feature_config import (
    FI_PATH,
    METADATA_PATH,
    PRICE_DIR,
)
ROOT = Path(__file__).resolve().parents[1]
def file_sha256(
    path: Path,
) -> str:
    """Beräknar SHA-256 för en fil."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()
def build_source_fingerprint() -> dict[str, object]:
    """
    Skapar ett deterministiskt fingerprint av alla
    källfiler som används av feature-bygget.
    """
    if not FI_PATH.exists():
        raise FileNotFoundError(
            f"Saknar FI-data: {FI_PATH}"
        )
    price_files = sorted(
        PRICE_DIR.glob(
            "prices_*.jsonl"
        )
    )
    source_files = [
        FI_PATH,
        *price_files,
    ]
    if not price_files:
        raise FileNotFoundError(
            f"Saknar prisdata i: {PRICE_DIR}"
        )
    entries: list[dict[str, object]] = []
    for path in source_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Saknar datafil: {path}"
            )
        entries.append(
            {
                "path": str(
                    path.relative_to(ROOT)
                ),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    canonical = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    fingerprint = hashlib.sha256(
        canonical
    ).hexdigest()
    return {
        "algorithm": "sha256",
        "fingerprint": fingerprint,
        "files": entries,
    }
def get_previous_fingerprint() -> str | None:
    """Läser tidigare source fingerprint från feature-metadata."""
    if not METADATA_PATH.exists():
        return None
    try:
        metadata = json.loads(
            METADATA_PATH.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            "Kunde inte läsa "
            f"{METADATA_PATH}: {exc}"
        ) from exc
    source_fingerprint = metadata.get(
        "source_fingerprint"
    )
    if not isinstance(
        source_fingerprint,
        dict,
    ):
        return None
    fingerprint = source_fingerprint.get(
        "fingerprint"
    )
    if isinstance(
        fingerprint,
        str,
    ):
        return fingerprint
    return None
def sources_changed() -> bool:
    """
    Returnerar True om feature-källorna har ändrats
    sedan senaste feature-bygget.
    """
    current = build_source_fingerprint()
    previous = get_previous_fingerprint()
    current_fingerprint = current[
        "fingerprint"
    ]
    print()
    print(
        "=== Feature source check ==="
    )
    print(
        "Metadata exists:",
        METADATA_PATH.exists(),
    )
    print(
        "Source files:",
        len(current["files"]),
    )
    print(
        "Previous:",
        previous,
    )
    print(
        "Current: ",
        current_fingerprint,
    )
    changed = (
        previous
        != current_fingerprint
    )
    print(
        "Changed:",
        changed,
    )
    print(
        "=== End feature source check ==="
    )
    return changed
def main() -> int:
    changed = sources_changed()
    print()
    print(
        "FEATURES_CHANGED="
        + (
            "true"
            if changed
            else "false"
        )
    )
    return 0
if __name__ == "__main__":
    raise SystemExit(
        main()
    )
