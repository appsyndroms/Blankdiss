"""Bygger och uppdaterar analysklar FI + pris-data inkrementellt."""
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

OUTPUT_PATH = (
    OUTPUT_DIR
    / "features.jsonl"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "features_metadata.json"
)

RETURN_HORIZONS = (
    1,
    5,
    20,
    60,
)


def normalize_text(
    value: Any,
) -> str:
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
    isin_text = normalize_text(
        isin
    )

    if isin_text:
        return f"ISIN:{isin_text}"

    return (
        "ISSUER:"
        f"{normalize_text(issuer)}"
    )


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
            + ", ".join(
                sorted(missing)
            )
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
        security_key(
            isin,
            issuer,
        )
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


def find_price_file() -> Path:
    files = sorted(
        PRICE_DIR.glob(
            "prices_*.jsonl"
        )
    )

    if not files:
        raise FileNotFoundError(
            "Ingen prices_*.jsonl "
            f"hittades i {PRICE_DIR}"
        )

    return files[-1]


def load_prices(
    path: Path,
) -> pd.DataFrame:
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
            "Prisdata saknar kolumner: "
            + ", ".join(
                sorted(missing)
            )
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
        security_key(
            isin,
            issuer,
        )
        for isin, issuer in zip(
            frame["isin"],
            frame["issuer"],
        )
    ]

    return frame.sort_values(
        [
            "security_key",
            "date",
            "yahoo_symbol",
        ],
        kind="mergesort",
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
    ] = grouped[
        "max_individual_position_pct"
    ].shift(1)

    frame[
        "previous_max_position_share_pct"
    ] = grouped[
        "max_position_share_pct"
    ].shift(1)

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

    previous = frame[
        "previous_short_interest_pct"
    ]

    frame["short_interest_relative_change"] = (
        np.where(
            previous >= 0.5,
            frame["short_interest_delta_pp"]
            / previous,
            np.nan,
        )
    )

    frame["short_interest_acceleration_pp"] = (
        grouped[
            "short_interest_delta_pp"
        ].diff()
    )

    for threshold in (
        1.0,
        2.0,
        3.0,
        5.0,
    ):
        label = (
            f"{threshold:.1f}".replace(
                ".",
                "_",
            )
        )

        current = (
            frame["short_interest_pct"]
            >= threshold
        )

        frame[
            f"above_{label}pct"
        ] = current

        frame[
            f"entered_above_{label}pct"
        ] = (
            previous.notna()
            & (previous < threshold)
            & current
        )

        frame[
            f"exited_below_{label}pct"
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
        ).reset_index(
            drop=True
        )
        for key, group in prices.groupby(
            "security_key",
            sort=False,
        )
    }


def add_price_history_features(
    result: dict[str, Any],
    series: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
) -> None:
    """
    Beräknar prisfeatures enbart från observationer
    på eller före signal-dagen.

    Ingen framtida prisinformation används.
    """

    closes = pd.to_numeric(
        series["close"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    def previous_return(
        days: int,
    ) -> float:
        previous_idx = entry_idx - days

        if previous_idx < 0:
            return np.nan

        previous_price = closes[
            previous_idx
        ]

        if (
            not np.isfinite(previous_price)
            or previous_price <= 0
        ):
            return np.nan

        return (
            entry_price
            / previous_price
            - 1.0
        )

    result["price_return_5d"] = (
        previous_return(5)
    )

    result["price_return_20d"] = (
        previous_return(20)
    )

    result["price_return_60d"] = (
        previous_return(60)
    )

    start_20 = max(
        0,
        entry_idx - 20,
    )

    history_20 = closes[
        start_20:entry_idx + 1
    ]

    if len(history_20) >= 2:
        result["price_volatility_20d"] = (
            float(
                pd.Series(
                    history_20
                ).pct_change().std()
            )
        )
    else:
        result["price_volatility_20d"] = (
            np.nan
        )

    high_20 = (
        np.nanmax(history_20)
        if len(history_20)
        else np.nan
    )

    result["price_distance_from_20d_high"] = (
        entry_price / high_20 - 1.0
        if np.isfinite(high_20)
        and high_20 > 0
        else np.nan
    )

    start_60 = max(
        0,
        entry_idx - 60,
    )

    history_60 = closes[
        start_60:entry_idx + 1
    ]

    high_60 = (
        np.nanmax(history_60)
        if len(history_60)
        else np.nan
    )

    result["price_distance_from_60d_high"] = (
        entry_price / high_60 - 1.0
        if np.isfinite(high_60)
        and high_60 > 0
        else np.nan
    )


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

    rows: list[dict[str, Any]] = []

    stats = {
        "fi_rows": int(len(fi)),
        "matched_rows": 0,
        "unmatched_rows": 0,
        "matched_by_isin": 0,
        "matched_by_issuer": 0,
    }

    for row in fi.itertuples(
        index=False
    ):
        mapping_source = None

        series = lookup.get(
            row.security_key
        )

        if (
            series is not None
            and not series.empty
        ):
            mapping_source = (
                "isin"
                if normalize_text(row.isin)
                else "issuer"
            )

        if (
            series is None
            and not normalize_text(row.isin)
        ):
            series = lookup.get(
                "ISSUER:"
                + normalize_text(row.issuer)
            )

            if (
                series is not None
                and not series.empty
            ):
                mapping_source = "issuer"

        if (
            series is None
            or series.empty
        ):
            stats["unmatched_rows"] += 1
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
            stats["unmatched_rows"] += 1
            continue

        stats["matched_rows"] += 1

        if mapping_source == "isin":
            stats["matched_by_isin"] += 1
        else:
            stats["matched_by_issuer"] += 1

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

        result["close"] = entry_price

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
            mapping_source
        )

        add_price_history_features(
            result,
            series,
            entry_idx,
            entry_price,
        )

        for horizon in RETURN_HORIZONS:
            target_idx = (
                entry_idx + horizon
            )

            if target_idx < len(series):
                result[
                    f"forward_return_{horizon}d"
                ] = (
                    float(
                        series.iloc[
                            target_idx
                        ]["close"]
                    )
                    / entry_price
                    - 1.0
                )
            else:
                result[
                    f"forward_return_{horizon}d"
                ] = np.nan

        rows.append(
            result
        )

    return (
        pd.DataFrame(rows),
        stats,
    )


def clean_for_json(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()

    for column in {
        "snapshot_date",
        "previous_snapshot_date",
        "price_date",
    }:
        if column in frame.columns:
            frame[column] = (
                pd.to_datetime(
                    frame[column],
                    errors="coerce",
                )
                .dt.strftime(
                    "%Y-%m-%d"
                )
            )

    frame = frame.astype(object)

    return frame.where(
        pd.notna(frame),
        None,
    )


def load_existing() -> pd.DataFrame:
    if not OUTPUT_PATH.exists():
        return pd.DataFrame()

    frame = pd.read_json(
        OUTPUT_PATH,
        lines=True,
    )

    if frame.empty:
        return frame

    for column in (
        "snapshot_date",
        "previous_snapshot_date",
        "price_date",
    ):
        if column in frame.columns:
            frame[column] = pd.to_datetime(
                frame[column],
                errors="coerce",
            )

    return frame


def feature_key(
    frame: pd.DataFrame,
) -> pd.Series:
    return (
        frame["security_key"].astype(str)
        + "|"
        + frame["snapshot_date"].dt.strftime(
            "%Y-%m-%d"
        )
    )


def refresh_incomplete_returns(
    existing: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    if existing.empty:
        return existing

    return_columns = [
        f"forward_return_{h}d"
        for h in RETURN_HORIZONS
    ]

    incomplete = existing[
        return_columns
    ].isna().any(axis=1)

    if not incomplete.any():
        return existing

    subset = existing.loc[
        incomplete
    ].copy()

    refreshed, _ = attach_prices(
        subset,
        prices,
    )

    if refreshed.empty:
        return existing

    refreshed = refreshed.set_index(
        [
            "security_key",
            "snapshot_date",
        ]
    )

    current = existing.set_index(
        [
            "security_key",
            "snapshot_date",
        ]
    )

    update_columns = [
        "price_date",
        "close",
        "close_on_signal_date",
        "days_from_fi_to_price",
        "price_match_available",
        "yahoo_symbol",
        "price_mapping_source",
        "price_return_5d",
        "price_return_20d",
        "price_return_60d",
        "price_volatility_20d",
        "price_distance_from_20d_high",
        "price_distance_from_60d_high",
        "forward_return_1d",
        "forward_return_5d",
        "forward_return_20d",
        "forward_return_60d",
    ]

    for column in update_columns:
        if column in refreshed.columns:
            current.loc[
                refreshed.index,
                column,
            ] = refreshed[column]

    return current.reset_index()


def write_features(
    frame: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean = clean_for_json(
        frame
    )

    clean.to_json(
        OUTPUT_PATH,
        orient="records",
        lines=True,
        force_ascii=False,
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


def main() -> None:
    print(
        "Featurejobb: startar."
    )

    fi = load_fi()

    print(
        f"{len(fi):,} FI-observationer"
    )

    price_file = find_price_file()

    prices = load_prices(
        price_file
    )

    print(
        f"{len(prices):,} prisobservationer"
    )

    existing = load_existing()

    # Första körningen: bygg hela historiken.
    if existing.empty:
        enriched = add_fi_features(
            fi
        )

        features, stats = attach_prices(
            enriched,
            prices,
        )

    else:
        existing_keys = set(
            feature_key(existing)
        )

        fi_keys = feature_key(
            fi
        )

        new_mask = ~fi_keys.isin(
            existing_keys
        )

        new_fi = fi.loc[
            new_mask
        ].copy()

        if new_fi.empty:
            features = existing.copy()

            features = (
                refresh_incomplete_returns(
                    features,
                    prices,
                )
            )

            stats = {
                "fi_rows": int(
                    len(fi)
                ),
                "new_fi_rows": 0,
                "matched_rows": 0,
                "unmatched_rows": 0,
            }

        else:
            # Viktigt:
            # vi måste hämta senaste gamla observation
            # FÖRE den första nya observationen per bolag.
            first_new_dates = (
                new_fi.groupby(
                    "security_key"
                )["snapshot_date"]
                .min()
            )

            context_mask = (
                fi["security_key"]
                .isin(
                    first_new_dates.index
                )
                & fi.apply(
                    lambda row: (
                        row["snapshot_date"]
                        < first_new_dates.get(
                            row["security_key"],
                            pd.Timestamp.max,
                        )
                    ),
                    axis=1,
                )
            )

            context = (
                fi.loc[
                    context_mask
                ]
                .sort_values(
                    [
                        "security_key",
                        "snapshot_date",
                    ],
                    kind="mergesort",
                )
                .groupby(
                    "security_key",
                    sort=False,
                )
                .tail(1)
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

            existing = (
                refresh_incomplete_returns(
                    existing,
                    prices,
                )
            )

            features = pd.concat(
                [
                    existing,
                    new_features,
                ],
                ignore_index=True,
            )

    features = features.sort_values(
        [
            "snapshot_date",
            "security_key",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
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
        "Featurejobb: klart."
    )


if __name__ == "__main__":
    main()
