"""Gemensam fingerprint-logik för feature-källor."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
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
def build_source_fingerprint(
    *,
    fi_path: Path,
    price_files: list[Path],
) -> dict[str, object]:
    """
    Skapar ett deterministiskt fingerprint av det underlag
    som används för att bygga feature-datasetet.
    """
    files = [
        fi_path,
        *sorted(price_files),
    ]
    entries: list[dict[str, object]] = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(
                f"Saknar feature source-fil: {path}"
            )
        entries.append(
            {
                "path": str(
                    path.relative_to(ROOT)
                ),
                "sha256": file_sha256(
                    path
                ),
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
def load_previous_fingerprint(
    metadata_path: Path,
) -> dict[str, object] | None:
    """Läser tidigare source fingerprint från feature-metadata."""
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Kunde inte läsa {metadata_path}: {exc}"
        ) from exc
    source_fingerprint = metadata.get(
        "source_fingerprint"
    )
    if not isinstance(
        source_fingerprint,
        dict,
    ):
        return None
    return source_fingerprint
def sources_changed(
    *,
    current_fingerprint: dict[str, object],
    previous_fingerprint: dict[str, object] | None,
) -> bool:
    """Returnerar True om feature-underlaget har ändrats."""
    if previous_fingerprint is None:
        return True
    return (
        previous_fingerprint.get(
            "fingerprint"
        )
        != current_fingerprint.get(
            "fingerprint"
        )
    )
def print_source_comparison(
    *,
    current_fingerprint: dict[str, object],
    previous_fingerprint: dict[str, object] | None,
) -> None:
    """Skriver en tydlig jämförelse för CI/loggar."""
    previous_fingerprint_value = None
    if previous_fingerprint is not None:
        previous_fingerprint_value = (
            previous_fingerprint.get(
                "fingerprint"
            )
        )
    print()
    print("=== Feature source check ===")
    print(
        "Previous:",
        previous_fingerprint_value,
    )
    print(
        "Current: ",
        current_fingerprint["fingerprint"],
    )
    old_entries = (
        previous_fingerprint.get(
            "files",
            [],
        )
        if previous_fingerprint
        else []
    )
    current_entries = current_fingerprint.get(
        "files",
        [],
    )
    old_by_path = {
        item["path"]: item
        for item in old_entries
    }
    current_by_path = {
        item["path"]: item
        for item in current_entries
    }
    all_paths = sorted(
        set(old_by_path)
        | set(current_by_path)
    )
    if not all_paths:
        print(
            "No source files found."
        )
        return
    print()
    print("=== Source file differences ===")
    differences = 0
    for path in all_paths:
        old = old_by_path.get(path)
        current = current_by_path.get(path)
        if old is None:
            print(
                f"ADDED:   {path}"
            )
            differences += 1
            continue
        if current is None:
            print(
                f"REMOVED: {path}"
            )
            differences += 1
            continue
        if (
            old.get("sha256")
            != current.get("sha256")
        ):
            print(
                f"CHANGED: {path}"
            )
            print(
                "  Previous SHA256:",
                old.get("sha256"),
            )
            print(
                "  Current SHA256: ",
                current.get("sha256"),
            )
            differences += 1
        elif (
            old.get("size_bytes")
            != current.get("size_bytes")
        ):
            print(
                f"SIZE CHANGED: {path}"
            )
            print(
                "  Previous size:",
                old.get("size_bytes"),
            )
            print(
                "  Current size: ",
                current.get("size_bytes"),
            )
            differences += 1
    if differences == 0:
        print(
            "No source file differences."
        )
