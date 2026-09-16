"""CI-hjälpfunktioner för Blankdiss feature-pipeline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
FEATURES_DIR = Path(
    "data/processed/analysis"
)
FEATURE_METADATA = (
    FEATURES_DIR / "features_metadata.json"
)
FEATURE_QC = (
    FEATURES_DIR / "features_qc.json"
)
FEATURE_SIZE_LIMIT = 20 * 1024 * 1024
def feature_paths() -> list[Path]:
    """Returnera alla feature-chunks."""
    return sorted(
        FEATURES_DIR.glob(
            "features_*.jsonl"
        )
    )
def inspect_features() -> None:
    """Skriv en kort översikt av feature-datasetet."""
    paths = feature_paths()
    print("================================")
    print("FEATURE DATASET")
    print("================================")
    try:
        import subprocess
        commit = subprocess.check_output(
            [
                "git",
                "rev-parse",
                "--short",
                "HEAD",
            ],
            text=True,
        ).strip()
    except Exception:
        commit = "unknown"
    print(f"Commit: {commit}")
    print(
        f"Feature-chunks: {len(paths)}"
    )
    if FEATURE_METADATA.exists():
        print("Metadata: OK")
    else:
        print("Metadata: SAKNAS")
    if FEATURE_QC.exists():
        print("QC: OK")
    else:
        print("QC: SAKNAS ännu")
def verify_features() -> None:
    """Verifiera att feature-datasetet finns."""
    if not FEATURE_METADATA.is_file():
        raise SystemExit(
            "features_metadata.json saknas."
        )
    paths = feature_paths()
    if not paths:
        raise SystemExit(
            "Inga feature-chunks hittades."
        )
    total_size = sum(
        path.stat().st_size
        for path in paths
    )
    total_mb = (
        total_size
        / 1024
        / 1024
    )
    print(
        "Feature-dataset finns."
    )
    print(
        f"Feature-chunks: {len(paths)}"
    )
    print(
        f"Total storlek: {total_mb:.2f} MB"
    )
    for path in paths:
        size = path.stat().st_size
        if size > FEATURE_SIZE_LIMIT:
            size_mb = (
                size
                / 1024
                / 1024
            )
            raise SystemExit(
                f"{path.name} är större "
                f"än 20 MB "
                f"({size_mb:.2f} MB)."
            )
    max_size_mb = (
        max(
            path.stat().st_size
            for path in paths
        )
        / 1024
        / 1024
    )
    print(
        "Feature-storlek OK: "
        f"{len(paths)} filer, "
        f"max {max_size_mb:.2f} MB"
    )
def verify_feature_qc() -> None:
    """Verifiera status från Feature QC."""
    if not FEATURE_QC.is_file():
        raise SystemExit(
            "features_qc.json saknas."
        )
    data = json.loads(
        FEATURE_QC.read_text(
            encoding="utf-8"
        )
    )
    status = data.get("status")
    print(
        f"Feature QC: {status}"
    )
    if status == "FAIL":
        raise SystemExit(
            "Feature QC failed "
            "with status: FAIL"
        )
    if status == "WARN":
        print(
            "Varningar tillåts — "
            "workflow fortsätter."
        )
    elif status == "PASS":
        print(
            "Feature QC OK."
        )
    else:
        raise SystemExit(
            "Feature QC returnerade "
            f"okänd status: {status}"
        )
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "inspect-features",
            "verify-features",
            "verify-feature-qc",
        ],
    )
    args = parser.parse_args()
    if args.command == "inspect-features":
        inspect_features()
    elif args.command == "verify-features":
        verify_features()
    elif args.command == "verify-feature-qc":
        verify_feature_qc()
if __name__ == "__main__":
    main()
