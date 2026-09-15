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
        *return_columns,
    ]

    for column in update_columns:
        if column in refreshed.columns:
            current.loc[
                refreshed.index,
                column,
            ] = refreshed[column]

    return current.reset_index()


def validate_output_columns(
    frame: pd.DataFrame,
) -> None:
    required = {
        "snapshot_date",
        "issuer",
        "isin",
        "security_key",
        "yahoo_symbol",
        "price_mapping_source",
        "price_date",
        "close",
        "close_on_signal_date",
        "days_from_fi_to_price",
        "price_match_available",
        "short_interest_pct",
        "active_holders",
        "max_individual_position_pct",
        "max_position_share_pct",
        "forward_return_1d",
        "forward_return_5d",
        "forward_return_20d",
        "forward_return_60d",
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
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise RuntimeError(
            "Featurejobb saknar "
            "förväntade outputkolumner: "
            + ", ".join(
                sorted(missing)
            )
        )


def write_jsonl(
    frame: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
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


def main() -> None:
    print(
        "Featurejobb: startar inkrementellt."
    )

    fi = load_fi()

    price_file = find_price_file()

    prices = load_prices(
        price_file
    )

    existing = load_existing()

    print(
        "Featurejobb: "
        f"{len(fi):,} "
        "FI-observationer lästa."
    )

    print(
        "Featurejobb: "
        f"{len(prices):,} "
        "prisobservationer lästa."
    )

    print(
        "Featurejobb: prisfil = "
        f"{price_file.name}"
    )

    if existing.empty:
        print(
            "Featurejobb: ingen befintlig "
            "feature-fil -> bygger "
            "initial historik."
        )

        result, stats = attach_prices(
            add_fi_features(fi),
            prices,
        )

    else:
        existing_keys = set(
            feature_key(existing)
        )

        candidates = fi.loc[
            ~feature_key(fi).isin(
                existing_keys
            )
        ].copy()

        print(
            "Featurejobb: "
            f"{len(existing):,} "
            "befintliga feature-rader."
        )

        print(
            "Featurejobb: "
            f"{len(candidates):,} "
            "nya FI-observationer."
        )

        if candidates.empty:
            result = (
                refresh_incomplete_returns(
                    existing,
                    prices,
                )
            )

            stats = {
                "matched_rows": 0,
                "unmatched_rows": 0,
                "matched_by_isin": 0,
                "matched_by_issuer": 0,
            }

        else:
            context_keys = (
                candidates[
                    "security_key"
                ].unique()
            )

            context = (
                fi.loc[
                    fi["security_key"].isin(
                        context_keys
                    )
                ]
                .sort_values(
                    [
                        "security_key",
                        "snapshot_date",
                    ]
                )
                .groupby(
                    "security_key",
                    sort=False,
                )
                .tail(1)
            )

            context = context.loc[
                ~feature_key(
                    context
                ).isin(
                    set(
                        feature_key(
                            candidates
                        )
                    )
                )
            ]

            combined = pd.concat(
                [
                    context,
                    candidates,
                ],
                ignore_index=True,
            )

            built = add_fi_features(
                combined
            )

            candidate_keys = set(
                feature_key(candidates)
            )

            built = built.loc[
                feature_key(
                    built
                ).isin(candidate_keys)
            ].copy()

            new_rows, stats = attach_prices(
                built,
                prices,
            )

            result = pd.concat(
                [
                    existing,
                    new_rows,
                ],
                ignore_index=True,
            )

            result = (
                refresh_incomplete_returns(
                    result,
                    prices,
                )
            )

    if result.empty:
        raise RuntimeError(
            "Featurejobb gav 0 "
            "matchade FI-observationer."
        )

    validate_output_columns(
        result
    )

    result = (
        result
        .sort_values(
            [
                "security_key",
                "snapshot_date",
            ],
            kind="mergesort",
        )
        .drop_duplicates(
            [
                "security_key",
                "snapshot_date",
            ],
            keep="last",
        )
    )

    json_result = clean_for_json(
        result
    )

    write_jsonl(
        json_result
    )

    metadata = {
        "source": {
            "fi_file": str(
                FI_PATH.relative_to(
                    ROOT
                )
            ),
            "price_file": str(
                price_file.relative_to(
                    ROOT
                )
            ),
        },
        "build_mode": "incremental",
        "fi_rows": len(fi),
        "price_rows": len(prices),
        "feature_rows": len(result),
        "matched_fi_rows": stats.get(
            "matched_rows",
            0,
        ),
        "unmatched_fi_rows": stats.get(
            "unmatched_rows",
            0,
        ),
        "matched_by_isin": stats.get(
            "matched_by_isin",
            0,
        ),
        "matched_by_issuer": stats.get(
            "matched_by_issuer",
            0,
        ),
        "security_keys_fi": int(
            fi["security_key"].nunique()
        ),
        "security_keys_prices": int(
            prices["security_key"].nunique()
        ),
        "return_horizons_trading_days": list(
            RETURN_HORIZONS
        ),
        "entry_price_rule": (
            "first available "
            "trading-day close on "
            "or after FI snapshot date"
        ),
        "forward_return_rule": (
            "close at Nth subsequent "
            "trading day divided by "
            "entry close minus 1"
        ),
        "identity_rule": (
            "ISIN first; exact "
            "normalized issuer fallback "
            "only when ISIN is absent"
        ),
        "missing_fi_observation_rule": (
            "not interpreted as zero"
        ),
        "feature_columns": list(
            result.columns
        ),
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

    print(
        "Featurejobb: "
        f"{len(result):,} rader "
        f"-> {OUTPUT_PATH}"
    )

    print(
        "Featurejobb: färdig."
    )


if __name__ == "__main__":
    main()
