from __future__ import annotations
import json
from pathlib import Path
from analysis.feature_config import (
    FI_PATH,
    METADATA_PATH,
    OUTPUT_DIR,
    PRICE_DIR,
)
from analysis.feature_source import build_source_fingerprint
ROOT = Path(__file__).resolve().parents[1]
def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
def find_price_files() -> list[Path]:
    return sorted(
        PRICE_DIR.glob("prices_*.jsonl")
    )
def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    price_files = find_price_files()
    if not FI_PATH.exists():
        raise FileNotFoundError(
            f"FI source file does not exist: {FI_PATH}"
        )
    if not price_files:
        raise FileNotFoundError(
            f"No price files found in: {PRICE_DIR}"
        )
    source_fingerprint = build_source_fingerprint(
        fi_path=FI_PATH,
        price_files=price_files,
    )
    # ------------------------------------------------------------------
    # Existing feature-building logic starts here.
    #
    # Keep the existing implementation from this point onward exactly
    # as it is today, including:
    #
    # - loading FI data
    # - loading price data
    # - constructing feature rows
    # - writing feature chunks
    # - feature QC data
    # - metadata
    #
    # The only change required in the existing implementation is that
    # source_fingerprint is now supplied by analysis.feature_source.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # Replace the block above with the existing feature-building body
    # from the current build_features.py.
    #
    # At the point where features_metadata.json is written, retain:
    #
    #     "source_fingerprint": source_fingerprint
    #
    # so that the next workflow run can compare the current source data
    # with the fingerprint from the previous build.
    # ------------------------------------------------------------------
    raise NotImplementedError(
        "Insert the existing feature-building implementation here. "
        "Only the source fingerprint implementation has changed."
    )
if __name__ == "__main__":
    main()
