"""Kontrollerar om canonical feature-data behöver byggas om."""
from __future__ import annotations
import os
from analysis.feature_config import (
    FI_PATH,
    METADATA_PATH,
    PRICE_DIR,
)
from analysis.feature_source import (
    build_source_fingerprint,
    load_previous_fingerprint,
    print_source_comparison,
    sources_changed,
)
def write_github_env(
    *,
    changed: bool,
    fingerprint: str,
) -> None:
    """Skriver resultatet till GitHub Actions environment."""
    github_env = os.environ.get(
        "GITHUB_ENV"
    )
    if not github_env:
        return
    with open(
        github_env,
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "FEATURES_CHANGED="
            + (
                "true"
                if changed
                else "false"
            )
            + "\n"
        )
        handle.write(
            "FEATURE_SOURCE_FINGERPRINT="
            + fingerprint
            + "\n"
        )
def main() -> None:
    """Kontrollerar feature source fingerprint."""
    if not FI_PATH.exists():
        raise SystemExit(
            f"Saknar FI-data: {FI_PATH}"
        )
    price_files = sorted(
        PRICE_DIR.glob(
            "prices_*.jsonl"
        )
    )
    print(
        f"Feature source: "
        f"{len(price_files)} prisfiler."
    )
    current_fingerprint = (
        build_source_fingerprint(
            fi_path=FI_PATH,
            price_files=price_files,
        )
    )
    previous_fingerprint = (
        load_previous_fingerprint(
            METADATA_PATH
        )
    )
    print_source_comparison(
        current_fingerprint=current_fingerprint,
        previous_fingerprint=previous_fingerprint,
    )
    changed = sources_changed(
        current_fingerprint=current_fingerprint,
        previous_fingerprint=previous_fingerprint,
    )
    fingerprint = str(
        current_fingerprint[
            "fingerprint"
        ]
    )
    print()
    print(
        "Changed:",
        changed,
    )
    print(
        "FEATURES_CHANGED="
        + (
            "true"
            if changed
            else "false"
        )
    )
    print(
        "FEATURE_SOURCE_FINGERPRINT="
        + fingerprint
    )
    print(
        "=== End feature source check ==="
    )
    write_github_env(
        changed=changed,
        fingerprint=fingerprint,
    )
if __name__ == "__main__":
    main()
