"""Bygger den kanoniska analys-/ML-dataseten för Blankdiss."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FI_PATH = (
    ROOT
    / "data"
    / "processed"
    / "fi"
    / "aggregate"
    / "reconstructed.jsonl"
)

PRICE_DIR = (
    ROOT
    / "data"
    / "raw"
    / "prices"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "analysis"
)

FEATURE_GLOB = "features_*.jsonl"

FEATURE_METADATA_PATH = (
    OUTPUT_DIR
    / "features_metadata.json"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "fi_price_features.jsonl"
)

OUTPUT_METADATA_PATH = (
    OUTPUT_DIR
    / "fi_price_features_metadata.json"
)

RETURN_HORIZONS = (
    1,
    5,
    20,
    60,
)

SEVERITY_HORIZON = 5

# Håll varje JSONL-fil tydligt under GitHub/CI-gränsen på 20 MB.
CHUNK_SIZE = 10_000


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().upper()

    text = re.sub(
        r"[^A-Z0-9ÅÄÖÉÜÆØ]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def security_key(
    isin: Any,
    issuer: Any,
) -> str:
    isin_text = normalize_text(isin)

    if isin_text:
        return f"ISIN:{isin_text}"

    return f"ISSUER:{normalize_text(issuer)}"


def load_fi() -> pd.DataFrame:
    if not FI_PATH.exists():
        raise FileNotFoundError(
            f"Saknar FI-data: {FI_PATH}"
        )

    frame = pd.read_json(
        FI_PATH,
        lines=True,
    )

    required = {
        "snapshot_date",
        "issuer",
        "isin",
        "short_interest_pct",
        "active_holders",
        "max_individual_position_pct",
        "max_position_share_pct",
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            "FI-data saknar kolumner: "
            + ", ".join(sorted(missing))
        )

    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    frame = frame.loc[
        frame["snapshot_date"].notna()
    ].copy()

    frame["issuer"] = (
        frame["issuer"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    frame["isin"] = frame["isin"].where(
        frame["isin"].notna(),
        None,
    )

    frame["security_key"] = [
        security_key(isin, issuer)
        for isin, issuer in zip(
            frame["isin"],
            frame["issuer"],
        )
    ]

    for column in (
        "short_interest_pct",
        "active_holders",
        "max_individual_position_pct",
        "max_position_share_pct",
    ):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.sort_values(
        [
            "security_key",
            "snapshot_date",
        ],
        kind="mergesort",
    )

    duplicates = frame.duplicated(
        [
            "security_key",
            "snapshot_date",
        ],
        keep="last",
    )

    return frame.loc[
        ~duplicates
    ].copy()


def find_price_files() -> list[Path]:
    files = sorted(
        PRICE_DIR.glob("prices_*.jsonl")
    )

    if not files:
        raise FileNotFoundError(
            f"Inga prices_*.jsonl hittades i {PRICE_DIR}"
        )

    return files


def load_prices(
    paths: list[Path],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for path in paths:
        print(
            f"Featurejobb: läser prisfil = "
            f"{path.name}"
        )

        frame = pd.read_json(
            path,
            lines=True,
        )

        required = {
            "date",
            "isin",
            "issuer",
            "yahoo_symbol",
            "close",
        }

        missing = required.difference(
            frame.columns
        )

        if missing:
            raise ValueError(
                f"Prisdata saknar kolumner i "
                f"{path.name}: "
                + ", ".join(sorted(missing))
            )

        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="coerce",
        )

        frame["close"] = pd.to_numeric(
            frame["close"],
            errors="coerce",
        )

        frame["isin"] = frame["isin"].where(
            frame["isin"].notna(),
            None,
        )

        frame["issuer"] = (
            frame["issuer"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        frame["yahoo_symbol"] = (
            frame["yahoo_symbol"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        frame = frame.loc[
            frame["date"].notna()
            & frame["close"].notna()
            & np.isfinite(frame["close"])
            & (frame["close"] > 0)
        ].copy()

        frame["security_key"] = [
            security_key(isin, issuer)
            for isin, issuer in zip(
                frame["isin"],
                frame["issuer"],
            )
        ]

        frames.append(frame)

    if not frames:
        raise RuntimeError(
            "Prisfiler hittades men innehöll "
            "inga användbara observationer."
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = combined.sort_values(
        [
            "security_key",
            "date",
            "yahoo_symbol",
        ],
        kind="mergesort",
    )

    combined = combined.drop_duplicates(
        subset=[
            "security_key",
            "date",
        ],
        keep="last",
    )

    return combined.reset_index(
        drop=True
    )


def add_fi_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.sort_values(
        [
            "security_key",
            "snapshot_date",
        ],
        kind="mergesort",
    ).copy()

    grouped = frame.groupby(
        "security_key",
        sort=False,
    )

    frame["previous_snapshot_date"] = (
        grouped["snapshot_date"].shift(1)
    )

    frame["previous_short_interest_pct"] = (
        grouped["short_interest_pct"].shift(1)
    )

    frame["previous_active_holders"] = (
        grouped["active_holders"].shift(1)
    )

    frame[
        "previous_max_individual_position_pct"
    ] = (
        grouped[
            "max_individual_position_pct"
        ].shift(1)
    )

    frame[
        "previous_max_position_share_pct"
    ] = (
        grouped[
            "max_position_share_pct"
        ].shift(1)
    )

    frame["fi_observation_gap_days"] = (
        frame["snapshot_date"]
        - frame["previous_snapshot_date"]
    ).dt.days

    frame["short_interest_delta_pp"] = (
        frame["short_interest_pct"]
        - frame["previous_short_interest_pct"]
    )

    frame["holder_delta"] = (
        frame["active_holders"]
        - frame["previous_active_holders"]
    )

    frame["max_position_delta_pp"] = (
        frame["max_individual_position_pct"]
        - frame[
            "previous_max_individual_position_pct"
        ]
    )

    frame["concentration_delta_pp"] = (
        frame["max_position_share_pct"]
        - frame[
            "previous_max_position_share_pct"
        ]
    )

    valid_relative = (
        frame["previous_short_interest_pct"]
        >= 0.5
    )

    frame["short_interest_relative_change"] = (
        np.where(
            valid_relative,
            frame["short_interest_delta_pp"]
            / frame["previous_short_interest_pct"],
            np.nan,
        )
    )

    frame["short_interest_acceleration_pp"] = (
        grouped[
            "short_interest_delta_pp"
        ].diff()
    )

    for threshold in (
        1,
        2,
        3,
        5,
    ):
        current = (
            frame["short_interest_pct"]
            >= threshold
        )

        previous = (
            frame["previous_short_interest_pct"]
        )

        frame[
            f"short_interest_ge_{threshold}pp"
        ] = current

        frame[
            f"entered_ge_{threshold}pp"
        ] = (
            previous.notna()
            & (previous < threshold)
            & current
        )

        frame[
            f"exited_below_{threshold}pp"
        ] = (
            previous.notna()
            & (previous >= threshold)
            & (~current)
        )

    frame["new_visible_observation"] = (
        frame["previous_snapshot_date"].isna()
    )

    return frame


def build_price_lookup(
    prices: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        key: group.sort_values(
            "date",
            kind="mergesort",
        ).reset_index(drop=True)
        for key, group in prices.groupby(
            "security_key",
            sort=False,
        )
    }


def attach_prices(
    fi: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
]:
    lookup = build_price_lookup(
        prices
    )

    rows: list[
        dict[str, Any]
    ] = []

    matched = 0
    unmatched = 0

    for row in fi.itertuples(
        index=False
    ):
        series = lookup.get(
            row.security_key
        )

        if (
            series is None
            and not normalize_text(row.isin)
        ):
            series = lookup.get(
                f"ISSUER:{normalize_text(row.issuer)}"
            )

        if series is None or series.empty:
            unmatched += 1
            continue

        dates = series[
            "date"
        ].to_numpy(
            dtype="datetime64[ns]"
        )

        snapshot = np.datetime64(
            row.snapshot_date.to_datetime64(),
            "ns",
        )

        entry_idx = int(
            np.searchsorted(
                dates,
                snapshot,
                side="left",
            )
        )

        if entry_idx >= len(series):
            unmatched += 1
            continue

        matched += 1

        entry = series.iloc[
            entry_idx
        ]

        entry_price = float(
            entry["close"]
        )

        result = row._asdict()

        result["price_date"] = (
            entry["date"]
        )

        result["close"] = (
            entry_price
        )

        # Explicita QC-/legacyfält.
        # Dessa beskriver samma matchning som
        # price_date/close men gör datasetet
        # kompatibelt med features_qc.py.
        result["close_on_signal_date"] = (
            entry_price
        )

        result["days_from_fi_to_price"] = (
            entry["date"]
            - row.snapshot_date
        ).days

        result["price_match_available"] = True

        result["yahoo_symbol"] = (
            entry["yahoo_symbol"]
        )

        result["price_mapping_source"] = (
            entry.get("mapping_source")
        )

        for horizon in RETURN_HORIZONS:
            target_idx = (
                entry_idx
                + horizon
            )

            if target_idx < len(series):
                target = series.iloc[
                    target_idx
                ]

                result[
                    f"forward_return_{horizon}d"
                ] = (
                    float(target["close"])
                    / entry_price
                    - 1.0
                )

                result[
                    f"forward_price_date_{horizon}d"
                ] = target["date"]

            else:
                result[
                    f"forward_return_{horizon}d"
                ] = np.nan

                result[
                    f"forward_price_date_{horizon}d"
                ] = pd.NaT

        severity_end_idx = (
            entry_idx
            + SEVERITY_HORIZON
        )

        if severity_end_idx < len(series):
            severity_window = series.iloc[
                entry_idx
                + 1 : severity_end_idx
                + 1
            ]

            severity_returns = (
                severity_window["close"]
                .to_numpy(dtype=float)
                / entry_price
                - 1.0
            )

            result["min_return_5d"] = (
                float(
                    np.min(
                        severity_returns
                    )
                )
            )

            result["max_return_5d"] = (
                float(
                    np.max(
                        severity_returns
                    )
                )
            )

            min_idx = int(
                np.argmin(
                    severity_returns
                )
            )

            max_idx = int(
                np.argmax(
                    severity_returns
                )
            )

            result[
                "min_return_5d_date"
            ] = severity_window.iloc[
                min_idx
            ]["date"]

            result[
                "max_return_5d_date"
            ] = severity_window.iloc[
                max_idx
            ]["date"]

        else:
            result["min_return_5d"] = np.nan
            result["max_return_5d"] = np.nan
            result["min_return_5d_date"] = pd.NaT
            result["max_return_5d_date"] = pd.NaT

        rows.append(result)

    return (
        pd.DataFrame(rows),
        {
            "fi_rows": int(len(fi)),
            "matched_rows": matched,
            "unmatched_rows": unmatched,
        },
    )


def clean_for_json(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()

    for column in frame.columns:
        if "date" in column:
            frame[column] = (
                pd.to_datetime(
                    frame[column],
                    errors="coerce",
                ).dt.strftime(
                    "%Y-%m-%d"
                )
            )

    return frame.replace(
        {
            np.nan: None
        }
    )


def remove_old_feature_chunks() -> None:
    for path in OUTPUT_DIR.glob(
        FEATURE_GLOB
    ):
        path.unlink()

    if FEATURE_METADATA_PATH.exists():
        FEATURE_METADATA_PATH.unlink()


def write_jsonl(
    frame: pd.DataFrame,
    path: Path,
) -> None:
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
) -> list[dict[str, Any]]:
    remove_old_feature_chunks()

    chunks: list[
        dict[str, Any]
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
            / f"features_{chunk_number:04d}.jsonl"
        )

        chunk = frame.iloc[
            start:end
        ].copy()

        write_jsonl(
            chunk,
            path,
        )

        chunks.append(
            {
                "path": str(
                    path.relative_to(ROOT)
                ),
                "rows": int(len(chunk)),
            }
        )

    return chunks


def validate_feature_dataset(
    frame: pd.DataFrame,
    chunks: list[dict[str, Any]],
) -> None:
    required = {
        "snapshot_date",
        "security_key",
        "short_interest_pct",
        "forward_return_5d",
        "min_return_5d",
        "max_return_5d",
        "min_return_5d_date",
        "max_return_5d_date",
        "close_on_signal_date",
        "days_from_fi_to_price",
        "price_match_available",
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Feature-dataset saknar kolumner: "
            + ", ".join(sorted(missing))
        )

    if not chunks:
        raise ValueError(
            "Feature-dataset skapade inga chunks."
        )

    chunk_rows = sum(
        chunk["rows"]
        for chunk in chunks
    )

    if chunk_rows != len(frame):
        raise ValueError(
            "Chunk-rader stämmer inte med "
            "feature-rader: "
            f"{chunk_rows} != {len(frame)}"
        )

    severity_available = (
        frame["min_return_5d"].notna()
        & frame["max_return_5d"].notna()
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
    chunks: list[dict[str, Any]],
) -> None:
    metadata = {
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
        "fi_rows": int(len(fi)),
        "price_rows": int(len(prices)),
        "feature_rows": int(len(result)),
        "matched_fi_rows": int(
            stats["matched_rows"]
        ),
        "unmatched_fi_rows": int(
            stats["unmatched_rows"]
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
        "return_horizons_trading_days": list(
            RETURN_HORIZONS
        ),
        "severity_horizon_trading_days": (
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
        "chunk_size": CHUNK_SIZE,
        "chunks": chunks,
    }

    with FEATURE_METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    legacy_metadata = dict(metadata)

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
    print(
        "Featurejobb: startar."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fi = load_fi()

    print(
        f"Featurejobb: {len(fi):,} "
        "FI-observationer lästa."
    )

    price_files = find_price_files()

    print(
        "Featurejobb: hittade "
        f"{len(price_files)} prisfiler."
    )

    prices = load_prices(
        price_files
    )

    print(
        f"Featurejobb: {len(prices):,} "
        "unika prisobservationer lästa."
    )

    print(
        "Featurejobb: prisfiler:"
    )

    for price_file in price_files:
        print(
            f"  - {price_file.name}"
        )

    fi = add_fi_features(
        fi
    )

    result, stats = attach_prices(
        fi,
        prices,
    )

    if result.empty:
        raise RuntimeError(
            "Featurejobb gav 0 "
            "matchade FI-observationer."
        )

    result = clean_for_json(
        result
    )

    write_jsonl(
        result,
        OUTPUT_PATH,
    )

    chunks = write_feature_chunks(
        result
    )

    validate_feature_dataset(
        result,
        chunks,
    )

    write_metadata(
        fi=fi,
        prices=prices,
        result=result,
        stats=stats,
        price_files=price_files,
        chunks=chunks,
    )

    print(
        "Featurejobb: klart."
    )

    print(
        f"Feature-rader: {len(result):,}"
    )

    print(
        f"Feature-kolumner: {len(result.columns)}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        "Severity: min_return_5d, "
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
        f"  {FEATURE_METADATA_PATH.relative_to(ROOT)}"
    )

    for chunk in chunks:
        print(
            f"  {chunk['path']} "
            f"({chunk['rows']:,} rader)"
        )


if __name__ == "__main__":
    main()
