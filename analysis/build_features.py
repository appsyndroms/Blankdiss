"""Bygger och uppdaterar analysklar FI + pris-data inkrementellt."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any
import pandas as pd
from analysis.feature_config import (
    FI_PATH,
    METADATA_PATH,
    OUTPUT_DIR,
    OUTPUT_PATH,
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
    find_price_file,
    load_prices,
)
from analysis.feature_returns import (
    add_forward_returns,
)
from analysis.feature_utils import (
    clean_for_json,
    feature_key,
)
def _timed(label: str, function, *args):
    started = time.perf_counter()
    result = function(*args)
    elapsed = time.perf_counter() - started
    print(
        f"FEATURE-TID {label}: {elapsed:.2f} s"
    )
    return result
def write_features(
    frame: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    started = time.perf_counter()
    clean = clean_for_json(
        frame
    )
    clean.to_json(
        OUTPUT_PATH,
        orient="records",
        lines=True,
        force_ascii=False,
    )
    print(
        f"FEATURE-TID write_features: "
        f"{time.perf_counter() - started:.2f} s"
    )
def write_metadata(
    stats: dict[str, Any],
    frame: pd.DataFrame,
    price_file: Path,
) -> None:
    metadata = {
        "feature_rows": int(
            len(frame)
        ),
        "columns": list(
            frame.columns
        ),
        "price_file": price_file.name,
        "price_file_path": str(
            price_file
        ),
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
def build_full_history(
    fi: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
]:
    enriched = _timed(
        "add_fi_features",
        add_fi_features,
        fi,
    )
    features, stats = _timed(
        "attach_prices",
        attach_prices,
        enriched,
        prices,
    )
    features = _timed(
        "add_forward_returns",
        add_forward_returns,
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
    new_fi = _timed(
        "find_new_fi_rows",
        find_new_fi_rows,
        fi,
        existing,
    )
    if new_fi.empty:
        features = _timed(
            "refresh_incomplete_returns",
            refresh_incomplete_returns,
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
    context = _timed(
        "build_context_rows",
        build_context_rows,
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
    work = _timed(
        "add_fi_features",
        add_fi_features,
        work,
    )
    new_features, stats = _timed(
        "attach_prices",
        attach_prices,
        work,
        prices,
    )
    new_features = _timed(
        "add_forward_returns",
        add_forward_returns,
        new_features,
        prices,
    )
    started = time.perf_counter()
    new_keys = set(
        feature_key(
            new_fi
        )
    )
    new_features = (
        new_features.loc[
            feature_key(
                new_features
            ).isin(new_keys)
        ]
        .copy()
    )
    print(
        f"FEATURE-TID filter_new_features: "
        f"{time.perf_counter() - started:.2f} s"
    )
    existing = _timed(
        "refresh_incomplete_returns",
        refresh_incomplete_returns,
        existing,
        prices,
    )
    features = _timed(
        "merge_features",
        merge_features,
        existing,
        new_features,
    )
    stats["new_fi_rows"] = int(
        len(new_fi)
    )
    return (
        features,
        stats,
    )
def main() -> None:
    total_started = time.perf_counter()
    print(
        "Featurejobb: startar."
    )
    fi = _timed(
        "load_fi",
        load_fi,
        FI_PATH,
    )
    print(
        f"{len(fi):,} FI-observationer"
    )
    price_file = _timed(
        "find_price_file",
        find_price_file,
        PRICE_DIR,
    )
    prices = _timed(
        "load_prices",
        load_prices,
        price_file,
    )
    print(
        f"{len(prices):,} prisobservationer"
    )
    existing = _timed(
        "load_existing",
        load_existing,
        OUTPUT_PATH,
    )
    if existing.empty:
        features, stats = build_full_history(
            fi,
            prices,
        )
    else:
        features, stats = build_incremental(
            fi,
            prices,
            existing,
        )
    features = _timed(
        "sort_features",
        lambda frame: frame.sort_values(
            [
                "snapshot_date",
                "security_key",
            ],
            kind="mergesort",
        ).reset_index(
            drop=True
        ),
        features,
    )
    write_features(
        features
    )
    write_metadata(
        stats,
        features,
        price_file,
    )
    print(
        f"{len(features):,} rader -> "
        f"{OUTPUT_PATH.name}"
    )
    print(
        f"FEATURE-TID TOTAL: "
        f"{time.perf_counter() - total_started:.2f} s"
    )
    print(
        "Featurejobb: klart."
    )
if __name__ == "__main__":
    main()
