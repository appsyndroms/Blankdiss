"""Bygger den kanoniska analys-/ML-dataseten för Blankdiss."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.feature_config import (
    FEATURE_GLOB,
    FI_PATH,
    METADATA_PATH,
    OUTPUT_DIR,
    PRICE_DIR,
    RETURN_HORIZONS,
)
from analysis.feature_fi import (
    add_fi_features,
    load_fi,
)
from analysis.feature_prices import (
    SEVERITY_HORIZON,
    attach_prices,
    find_price_files,
    load_prices,
)
from analysis.feature_returns import (
    add_forward_returns,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    OUTPUT_DIR
    / "fi_price_features.jsonl"
)

OUTPUT_METADATA_PATH = (
    OUTPUT_DIR
    / "fi_price_features_metadata.json"
)

# Håll varje JSONL-fil tydligt under CI-gränsen på 20 MB.
CHUNK_SIZE = 10_000


def clean_for_json(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Gör DataFrame säker för JSONL."""

    result = frame.copy()

    for column in result.columns:
        if "date" in column.lower():
            result[column] = (
                pd.to_datetime(
                    result[column],
                    errors="coerce",
                ).dt.strftime(
                    "%Y-%m-%d"
                )
            )

    result = result.replace(
        {
            np.nan: None,
            np.inf: None,
            -np.inf: None,
        }
    )

    return result


def remove_old_feature_chunks() -> None:
    """Tar bort gamla genererade featurefiler."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in OUTPUT_DIR.glob(
        FEATURE_GLOB
    ):
        path.unlink()

    if METADATA_PATH.exists():
        METADATA_PATH.unlink()


def write_jsonl(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """Skriver DataFrame som JSONL."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in frame.to_dict(
            orient="records"
        ):
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def write_feature_chunks(
    frame: pd.DataFrame,
) -> list[dict[str, object]]:
    """Skriver feature-datasetet i mindre JSONL-chunks."""

    remove_old_feature_chunks()

    chunks: list[
        dict[str, object]
    ] = []

    total_rows = len(frame)

    for start in range(
        0,
        total_rows,
        CHUNK_SIZE,
    ):
        end = min(
            start + CHUNK_SIZE,
            total_rows,
        )

        chunk_number = (
            start // CHUNK_SIZE
            + 1
        )

        path = (
            OUTPUT_DIR
            / (
                f"features_"
                f"{chunk_number:04d}.jsonl"
            )
        )

        chunk = frame.iloc[
            start:end
        ].copy()

        write_jsonl(
            chunk,
            path,
        )

        size_bytes = path.stat().st_size

        chunks.append(
            {
                "path": str(
                    path.relative_to(ROOT)
                ),
                "rows": int(
                    len(chunk)
                ),
                "size_bytes": int(
                    size_bytes
                ),
            }
        )

        if size_bytes >= 20 * 1024 * 1024:
            raise ValueError(
                f"{path.name} blev "
                f"{size_bytes / 1024 / 1024:.2f} MB. "
                "Minska CHUNK_SIZE."
            )

    return chunks


def validate_feature_dataset(
    frame: pd.DataFrame,
    chunks: list[dict[str, object]],
) -> None:
    """Validerar det kanoniska feature-schemat."""

    required = {
        # Identitet / FI
        "snapshot_date",
        "issuer",
        "isin",
        "security_key",
        "short_interest_pct",
        "active_holders",
        "max_individual_position_pct",
        "max_position_share_pct",

        # FI-historik
        "previous_snapshot_date",
        "previous_short_interest_pct",
        "previous_active_holders",
        "previous_max_individual_position_pct",
        "previous_max_position_share_pct",
        "fi_observation_gap_days",
        "short_interest_delta_pp",
        "holder_delta",
        "max_position_delta_pp",
        "concentration_delta_pp",
        "short_interest_relative_change",
        "short_interest_acceleration_pp",
        "new_visible_observation",

        # Threshold-features.
        "above_1_0pct",
        "entered_above_1_0pct",
        "exited_below_1_0pct",
        "above_2_0pct",
        "entered_above_2_0pct",
        "exited_below_2_0pct",
        "above_3_0pct",
        "entered_above_3_0pct",
        "exited_below_3_0pct",
        "above_5_0pct",
        "entered_above_5_0pct",
        "exited_below_5_0pct",

        # Prisidentitet
        "price_date",
        "close",
        "close_on_signal_date",
        "days_from_fi_to_price",
        "price_match_available",
        "yahoo_symbol",
        "price_mapping_source",

        # Prisfeatures.
        "price_return_5d",
        "price_return_20d",
        "price_return_60d",
        "price_volatility_20d",
        "price_distance_from_20d_high",
        "price_distance_from_60d_high",

        # Forward returns / targets.
        "forward_return_1d",
        "forward_return_5d",
        "forward_return_20d",
        "forward_return_60d",

        # Severity targets.
        "min_return_5d",
        "max_return_5d",
        "min_return_5d_date",
        "max_return_5d_date",
    }

    missing = sorted(
        required.difference(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            "Feature-dataset saknar "
            "kolumner:\n"
            + "\n".join(
                f"- {column}"
                for column in missing
            )
        )

    if frame.empty:
        raise ValueError(
            "Feature-datasetet är tomt."
        )

    if not chunks:
        raise ValueError(
            "Feature-dataset skapade inga chunks."
        )

    chunk_rows = sum(
        int(chunk["rows"])
        for chunk in chunks
    )

    if chunk_rows != len(frame):
        raise ValueError(
            "Chunk-rader stämmer inte "
            "med feature-rader: "
            f"{chunk_rows} != {len(frame)}"
        )

    duplicates = frame.duplicated(
        subset=[
            "security_key",
            "snapshot_date",
        ],
        keep=False,
    )

    if duplicates.any():
        raise ValueError(
            "Feature-datasetet innehåller "
            "dubbletter på "
            "security_key + snapshot_date: "
            f"{int(duplicates.sum())} rader."
        )

    severity_available = (
        pd.to_numeric(
            frame["min_return_5d"],
            errors="coerce",
        ).notna()
        & pd.to_numeric(
            frame["max_return_5d"],
            errors="coerce",
        ).notna()
    )

    if not severity_available.any():
        raise ValueError(
            "Severity-targets saknar helt "
            "giltiga observationer."
        )


def write_metadata(
    *,
    fi: pd.DataFrame,
    prices: pd.DataFrame,
    result: pd.DataFrame,
    stats: dict[str, int],
    price_files: list[Path],
    chunks: list[dict[str, object]],
) -> None:
    """Skriver metadata för feature-datasetet."""

    metadata = {
        "dataset": (
            "Blankdiss canonical "
            "FI + price feature dataset"
        ),
        "source": {
            "fi_file": str(
                FI_PATH.relative_to(ROOT)
            ),
            "price_files": [
                str(
                    path.relative_to(ROOT)
                )
                for path in price_files
            ],
        },
        "fi_rows": int(
            len(fi)
        ),
        "price_rows": int(
            len(prices)
        ),
        "feature_rows": int(
            len(result)
        ),
        "matched_fi_rows": int(
            stats.get(
                "matched_rows",
                0,
            )
        ),
        "unmatched_fi_rows": int(
            stats.get(
                "unmatched_rows",
                0,
            )
        ),
        "matched_by_isin": int(
            stats.get(
                "matched_by_isin",
                0,
            )
        ),
        "matched_by_issuer": int(
            stats.get(
                "matched_by_issuer",
                0,
            )
        ),
        "security_keys_fi": int(
            fi["security_key"].nunique()
        ),
        "security_keys_prices": int(
            prices["security_key"].nunique()
        ),
        "feature_columns": [
            str(column)
            for column in result.columns
        ],
        "return_horizons_trading_days": [
            int(value)
            for value in RETURN_HORIZONS
        ],
        "severity_horizon_trading_days": int(
            SEVERITY_HORIZON
        ),
        "severity_window": (
            "trading_days_1_to_5_after_entry"
        ),
        "entry_rule": (
            "first available trading-day "
            "close on or after FI snapshot date"
        ),
        "severity_columns": [
            "min_return_5d",
            "max_return_5d",
            "min_return_5d_date",
            "max_return_5d_date",
        ],
        "price_feature_columns": [
            "price_return_5d",
            "price_return_20d",
            "price_return_60d",
            "price_volatility_20d",
            "price_distance_from_20d_high",
            "price_distance_from_60d_high",
        ],
        "chunk_size": int(
            CHUNK_SIZE
        ),
        "chunks": chunks,
    }

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

    legacy_metadata = dict(
        metadata
    )

    legacy_metadata[
        "feature_dataset"
    ] = "fi_price_features.jsonl"

    with OUTPUT_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            legacy_metadata,
            handle,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    """Bygg hela det kanoniska feature-datasetet."""

    print(
        "Featurejobb: startar."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 1. FI
    # ---------------------------------------------------------

    fi = load_fi(
        FI_PATH
    )

    print(
        f"Featurejobb: "
        f"{len(fi):,} FI-observationer lästa."
    )

    fi = add_fi_features(
        fi
    )

    # ---------------------------------------------------------
    # 2. Prisdata
    # ---------------------------------------------------------

    price_files = find_price_files(
        PRICE_DIR
    )

    print(
        "Featurejobb: hittade "
        f"{len(price_files)} prisfiler."
    )

    for price_file in price_files:
        print(
            f"  - {price_file.name}"
        )

    prices = load_prices(
        price_files
    )

    print(
        f"Featurejobb: "
        f"{len(prices):,} prisobservationer lästa."
    )

    # ---------------------------------------------------------
    # 3. Matcha FI → pris
    #
    # feature_prices.py äger:
    # - price_date
    # - close
    # - prisfeatures
    # - severity targets
    # - mapping source
    # ---------------------------------------------------------

    result, stats = attach_prices(
        fi,
        prices,
    )

    if result.empty:
        raise RuntimeError(
            "Featurejobb gav 0 "
            "matchade FI-observationer."
        )

    print(
        "Featurejobb: "
        f"{stats.get('matched_rows', 0):,} "
        "matchade FI-observationer."
    )

    print(
        "Featurejobb: "
        f"{stats.get('unmatched_rows', 0):,} "
        "FI-observationer utan pris."
    )

    # ---------------------------------------------------------
    # 4. Forward returns
    #
    # Dessa är targets och beräknas separat från
    # samtidiga prisfeatures.
    # ---------------------------------------------------------

    result = add_forward_returns(
        result,
        prices,
    )

    # ---------------------------------------------------------
    # 5. JSON-normalisering
    # ---------------------------------------------------------

    result = clean_for_json(
        result
    )

    # ---------------------------------------------------------
    # 6. Skriv legacy-dataset
    # ---------------------------------------------------------

    write_jsonl(
        result,
        OUTPUT_PATH,
    )

    # ---------------------------------------------------------
    # 7. Skriv kanoniska chunks
    # ---------------------------------------------------------

    chunks = write_feature_chunks(
        result
    )

    # ---------------------------------------------------------
    # 8. Validera innan metadata skrivs
    # ---------------------------------------------------------

    validate_feature_dataset(
        result,
        chunks,
    )

    # ---------------------------------------------------------
    # 9. Metadata
    # ---------------------------------------------------------

    write_metadata(
        fi=fi,
        prices=prices,
        result=result,
        stats=stats,
        price_files=price_files,
        chunks=chunks,
    )

    # ---------------------------------------------------------
    # 10. Sammanfattning
    # ---------------------------------------------------------

    print(
        "Featurejobb: klart."
    )

    print(
        f"Feature-rader: "
        f"{len(result):,}"
    )

    print(
        f"Feature-kolumner: "
        f"{len(result.columns)}"
    )

    print(
        f"Chunks: "
        f"{len(chunks)}"
    )

    print(
        "Severity: "
        "min_return_5d, "
        "max_return_5d, "
        "min_return_5d_date, "
        "max_return_5d_date"
    )

    print(
        "Skrivet:"
    )

    print(
        f"  {OUTPUT_PATH.relative_to(ROOT)}"
    )

    print(
        f"  {METADATA_PATH.relative_to(ROOT)}"
    )

    for chunk in chunks:
        size_mb = (
            Path(
                ROOT / chunk["path"]
            ).stat().st_size
            / 1024
            / 1024
        )

        print(
            f"  {chunk['path']} "
            f"({chunk['rows']:,} rader, "
            f"{size_mb:.2f} MB)"
        )


if __name__ == "__main__":
    main()
