"""Bygger den kanoniska analys-/ML-dataseten för Blankdiss."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from analysis.feature_config import (
    FEATURE_GLOB,
    FI_PATH,
    MARKET_PATH,
    METADATA_PATH,
    OUTPUT_DIR,
    PRICE_DIR,
    RELATIVE_HORIZONS,
    RETURN_HORIZONS,
    SECTOR_MAP_PATH,
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
from analysis.feature_relative import (
    add_relative_features,
)
from analysis.feature_returns import (
    add_forward_returns,
)
from analysis.feature_source import (
    build_source_fingerprint,
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
CHUNK_SIZE = 7_500

STOCKHOLM_TZ = ZoneInfo("Europe/Stockholm")

JSON_DATE_COLUMNS = {
    "snapshot_date",
    "previous_snapshot_date",
    "price_date",
    "min_return_5d_date",
    "max_return_5d_date",
}


def parse_args() -> argparse.Namespace:
    """Parsar kommandoradsargument."""

    parser = argparse.ArgumentParser(
        description=(
            "Bygger det kanoniska "
            "FI + price feature-datasetet."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Tvinga full rebuild även om "
            "dagens snapshot redan finns."
        ),
    )

    return parser.parse_args()


def current_feature_date() -> date:
    """Returnerar dagens datum i Europe/Stockholm."""

    return datetime.now(
        STOCKHOLM_TZ
    ).date()


def feature_dataset_has_date(
    target_date: date,
) -> bool:
    """Kontrollerar om feature-datasetet redan innehåller target_date."""

    if not OUTPUT_DIR.exists():
        return False

    feature_chunks = sorted(
        OUTPUT_DIR.glob(FEATURE_GLOB)
    )

    if not feature_chunks:
        return False

    target = target_date.isoformat()

    for path in feature_chunks:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                line = line.strip()

                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                snapshot_date = row.get(
                    "snapshot_date"
                )

                if snapshot_date is None:
                    continue

                if (
                    str(snapshot_date)[:10]
                    == target
                ):
                    return True

    return False


def clean_for_json(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Gör DataFrame säker för JSONL."""

    result = frame.copy()

    for column in JSON_DATE_COLUMNS:
        if column not in result.columns:
            continue

        result[column] = (
            pd.to_datetime(
                result[column],
                errors="coerce",
            )
            .dt.strftime(
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

        # Threshold-features
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

        # Prisfeatures
        "price_return_5d",
        "price_return_20d",
        "price_return_60d",
        "price_volatility_20d",
        "price_distance_from_20d_high",
        "price_distance_from_60d_high",

        # Marknads-/sektorkontext
        "sector",
        "market_return_5d",
        "market_return_20d",
        "market_return_60d",
        "sector_return_5d",
        "sector_return_20d",
        "sector_return_60d",
        "price_return_5d_relative_market",
        "price_return_20d_relative_market",
        "price_return_60d_relative_market",
        "price_return_5d_relative_sector",
        "price_return_20d_relative_sector",
        "price_return_60d_relative_sector",

        # Forward returns / targets
        "forward_return_1d",
        "forward_return_5d",
        "forward_return_20d",
        "forward_return_60d",

        # Severity targets
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
    source_fingerprint: dict[str, object],
) -> None:
    """Skriver metadata för feature-datasetet."""

    metadata = {
        "dataset": (
            "Blankdiss canonical "
            "FI + price + market + sector "
            "feature dataset"
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
            "market_file": str(
                MARKET_PATH.relative_to(ROOT)
            ),
            "sector_map_file": str(
                SECTOR_MAP_PATH.relative_to(ROOT)
            ),
        },
        "source_fingerprint": source_fingerprint,
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
        "relative_horizons_trading_days": [
            int(value)
            for value in RELATIVE_HORIZONS
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
        "relative_feature_semantics": {
            "market_return": (
                "OMXSPI return over N prior "
                "trading observations"
            ),
            "sector_return": (
                "median return over N prior "
                "trading observations for "
                "sector constituents"
            ),
            "relative_market": (
                "stock return minus OMXSPI return"
            ),
            "relative_sector": (
                "stock return minus sector return"
            ),
        },
        "relative_horizons": [
            int(value)
            for value in RELATIVE_HORIZONS
        ],
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
        "relative_feature_columns": [
            "market_return_5d",
            "market_return_20d",
            "market_return_60d",
            "sector_return_5d",
            "sector_return_20d",
            "sector_return_60d",
            "price_return_5d_relative_market",
            "price_return_20d_relative_market",
            "price_return_60d_relative_market",
            "price_return_5d_relative_sector",
            "price_return_20d_relative_sector",
            "price_return_60d_relative_sector",
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

    # Behåll denna metadatafil tills legacy-städningen görs.
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


def main(
    *,
    force: bool = False,
) -> None:
    """Bygg hela det kanoniska feature-datasetet."""

    today = current_feature_date()

    print(
        "Featurejobb: startar."
    )

    if force:
        print(
            "Featurejobb: --force är aktiverat "
            f"– full rebuild för {today.isoformat()}."
        )
    else:
        print(
            "Featurejobb: kontrollerar befintliga "
            "feature-chunks för "
            f"{today.isoformat()}."
        )

        if feature_dataset_has_date(
            today
        ):
            print(
                "Feature dataset finns redan för "
                f"{today.isoformat()} – "
                "hoppar över build."
            )
            return

        print(
            "Featurejobb: dagens snapshot saknas "
            "– full rebuild."
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
    # 3. Source fingerprint
    # ---------------------------------------------------------

    source_fingerprint = build_source_fingerprint(
        fi_path=FI_PATH,
        price_files=price_files,
        market_path=MARKET_PATH,
        sector_map_path=SECTOR_MAP_PATH,
    )

    print(
        "Featurejobb: source fingerprint "
        f"{source_fingerprint['fingerprint']}"
    )

    # ---------------------------------------------------------
    # 4. Matcha FI → pris
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
    # 5. Forward returns
    # ---------------------------------------------------------

    result = add_forward_returns(
        result,
        prices,
    )

    # ---------------------------------------------------------
    # 6. Marknads- och sektorkontext
    # ---------------------------------------------------------

    result = add_relative_features(
        result,
        prices,
    )

    print(
        "Featurejobb: marknads- och "
        "sektorrelativa features skapade."
    )

    print(
        "  Horisonter:",
        ", ".join(
            f"{value}d"
            for value in RELATIVE_HORIZONS
        )
    )

    # ---------------------------------------------------------
    # 7. JSON-normalisering
    # ---------------------------------------------------------

    result = clean_for_json(
        result
    )

    # ---------------------------------------------------------
    # 8. Skriv legacy-dataset
    # ---------------------------------------------------------

    write_jsonl(
        result,
        OUTPUT_PATH,
    )

    # ---------------------------------------------------------
    # 9. Skriv kanoniska chunks
    # ---------------------------------------------------------

    chunks = write_feature_chunks(
        result
    )

    # ---------------------------------------------------------
    # 10. Validera
    # ---------------------------------------------------------

    validate_feature_dataset(
        result,
        chunks,
    )

    # ---------------------------------------------------------
    # 11. Metadata
    # ---------------------------------------------------------

    write_metadata(
        fi=fi,
        prices=prices,
        result=result,
        stats=stats,
        price_files=price_files,
        chunks=chunks,
        source_fingerprint=source_fingerprint,
    )

    # ---------------------------------------------------------
    # 12. Sammanfattning
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
        "Relativa horisonter:",
        ", ".join(
            f"{value}d"
            for value in RELATIVE_HORIZONS
        )
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
    args = parse_args()

    main(
        force=args.force,
    )
