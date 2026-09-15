"""Bygger och uppdaterar analysklar FI + pris-data inkrementellt."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pandas as pd
from analysis.feature_config import (
    FI_PATH,
    METADATA_PATH,
    OUTPUT_DIR,
    PRICE_DIR,
)
from analysis.feature_fi import (
    add_fi_features,
    load_fi,
)
from analysis.feature_incremental import (
    build_context_rows,
    find_new_fi_rows,
    load_existing,
    merge_features,
    refresh_incomplete_returns,
)
from analysis.feature_prices import (
    attach_prices,
    find_price_files,
    load_prices,
)
from analysis.feature_returns import (
    add_forward_returns,
)
from analysis.feature_utils import (
    clean_for_json,
    feature_key,
)
CHUNK_SIZE_MB = 20
CHUNK_SIZE_BYTES = (
    CHUNK_SIZE_MB * 1024 * 1024
)
CHUNK_PREFIX = "features_"
def feature_chunk_paths() -> list[Path]:
    """Returnerar alla genererade feature-chunks."""
    return sorted(
        OUTPUT_DIR.glob(
            f"{CHUNK_PREFIX}*.jsonl"
        )
    )
def remove_feature_chunks() -> None:
    """Tar bort alla tidigare feature-chunks."""
    for path in feature_chunk_paths():
        path.unlink()
def write_features(
    frame: pd.DataFrame,
) -> list[Path]:
    """
    Skriver feature-datasetet som JSONL-chunks.
    Alla tidigare chunks rensas först.
    Ingen enskild fil blir större än
    CHUNK_SIZE_MB.
    """
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    remove_feature_chunks()
    clean = clean_for_json(
        frame
    )
    paths: list[Path] = []
    chunk_number = 1
    rows: list[str] = []
    current_size = 0
    def flush() -> None:
        nonlocal chunk_number
        nonlocal rows
        nonlocal current_size
        if not rows:
            return
        path = (
            OUTPUT_DIR
            / (
                f"{CHUNK_PREFIX}"
                f"{chunk_number:04d}.jsonl"
            )
        )
        path.write_text(
            "".join(rows),
            encoding="utf-8",
        )
        paths.append(
            path
        )
        chunk_number += 1
        rows = []
        current_size = 0
    for record in clean.to_dict(
        orient="records"
    ):
        line = (
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )
        line_size = len(
            line.encode("utf-8")
        )
        if (
            rows
            and current_size + line_size
            > CHUNK_SIZE_BYTES
        ):
            flush()
        rows.append(
            line
        )
        current_size += line_size
    flush()
    return paths
def write_metadata(
    stats: dict[str, Any],
    frame: pd.DataFrame,
    price_files: list[Path],
    feature_chunks: list[Path],
) -> None:
    metadata = {
        "feature_rows": int(
            len(frame)
        ),
        "columns": list(
            frame.columns
        ),
        "chunk_size_mb": CHUNK_SIZE_MB,
        "chunk_count": int(
            len(feature_chunks)
        ),
        "chunks": [
            path.name
            for path in feature_chunks
        ],
        "price_files": [
            path.name
            for path in price_files
        ],
        "price_file_count": int(
            len(price_files)
        ),
        "price_file_paths": [
            str(path)
            for path in price_files
        ],
        "stats": stats,
    }
    METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            ensure_ascii=False,
            indent=2,
        )
def load_existing_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}
    try:
        with METADATA_PATH.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(
                handle
            )
        if isinstance(
            data,
            dict,
        ):
            return data
    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
    ):
        pass
    return {}
def requires_full_rebuild(
    existing: pd.DataFrame,
    price_files: list[Path],
) -> tuple[bool, str]:
    """
    Avgör om befintligt feature-arkiv måste byggas om.
    """
    if existing.empty:
        return (
            True,
            "Inget befintligt feature-arkiv finns.",
        )
    metadata = load_existing_metadata()
    stored_files = metadata.get(
        "price_files"
    )
    if not isinstance(
        stored_files,
        list,
    ):
        return (
            True,
            "Feature-metadata saknar prisfil-lista.",
        )
    stored_files = sorted(
        str(name)
        for name in stored_files
    )
    current_files = sorted(
        path.name
        for path in price_files
    )
    if stored_files != current_files:
        return (
            True,
            "Feature-arkivet är byggt med en annan "
            "uppsättning prisfiler än den aktuella "
            "lokala historiken.",
        )
    return (
        False,
        "Feature-arkivet är byggt med aktuell "
        "prisfiluppsättning.",
    )
def build_full_history(
    fi: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
]:
    enriched = add_fi_features(
        fi
    )
    features, stats = attach_prices(
        enriched,
        prices,
    )
    features = add_forward_returns(
        features,
        prices,
    )
    return (
        features,
        stats,
    )
def build_incremental(
    fi: pd.DataFrame,
    prices: pd.DataFrame,
    existing: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
]:
    new_fi = find_new_fi_rows(
        fi,
        existing,
    )
    if new_fi.empty:
        features = refresh_incomplete_returns(
            existing,
            prices,
        )
        stats = {
            "fi_rows": int(
                len(fi)
            ),
            "new_fi_rows": 0,
            "matched_rows": 0,
            "unmatched_rows": 0,
        }
        return (
            features,
            stats,
        )
    context = build_context_rows(
        fi,
        new_fi,
    )
    work = pd.concat(
        [
            context,
            new_fi,
        ],
        ignore_index=True,
    )
    work = add_fi_features(
        work
    )
    new_features, stats = attach_prices(
        work,
        prices,
    )
    new_features = add_forward_returns(
        new_features,
        prices,
    )
    new_keys = set(
        feature_key(
            new_fi
        )
    )
    new_features = (
        new_features.loc[
            feature_key(
                new_features
            ).isin(
                new_keys
            )
        ]
        .copy()
    )
    existing = refresh_incomplete_returns(
        existing,
        prices,
    )
    features = merge_features(
        existing,
        new_features,
    )
    stats[
        "new_fi_rows"
    ] = int(
        len(new_fi)
    )
    return (
        features,
        stats,
    )
def main() -> None:
    print(
        "Featurejobb: startar."
    )
    fi = load_fi(
        FI_PATH
    )
    print(
        f"{len(fi):,} FI-observationer"
    )
    price_files = find_price_files(
        PRICE_DIR
    )
    print(
        "Featurejobb: "
        f"{len(price_files)} prisfiler hittades."
    )
    prices = load_prices(
        price_files
    )
    print(
        f"{len(prices):,} prisobservationer"
    )
    # Chunk-formatet är nu den enda källan
    # till befintliga features. Ingen OUTPUT_PATH
    # eller gammal monolitisk features-fil används.
    existing = load_existing()
    full_rebuild, reason = requires_full_rebuild(
        existing,
        price_files,
    )
    print(
        "Featurejobb: "
        f"{reason}"
    )
    if full_rebuild:
        print(
            "Featurejobb: "
            "bygger om hela feature-historiken."
        )
        features, stats = build_full_history(
            fi,
            prices,
        )
        stats[
            "full_rebuild"
        ] = 1
    else:
        print(
            "Featurejobb: "
            "fortsätter inkrementellt."
        )
        features, stats = build_incremental(
            fi,
            prices,
            existing,
        )
        stats[
            "full_rebuild"
        ] = 0
    features = (
        features.sort_values(
            [
                "snapshot_date",
                "security_key",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )
    feature_chunks = write_features(
        features
    )
    write_metadata(
        stats,
        features,
        price_files,
        feature_chunks,
    )
    print(
        f"{len(features):,} rader -> "
        f"{len(feature_chunks)} feature-chunks"
    )
    for path in feature_chunks:
        print(
            f"  {path.name}: "
            f"{path.stat().st_size / 1024 / 1024:.1f} MB"
        )
    print(
        "Featurejobb: klart."
    )
if __name__ == "__main__":
    main()
