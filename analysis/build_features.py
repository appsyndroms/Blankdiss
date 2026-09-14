"""Bygger en analysklar FI + pris-dataset för Blankdiss."""
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
        grouped["short_interest_delta_pp"].diff()
    )

    for threshold in (
        1.0,
        2.0,
        3.0,
        5.0,
    ):
        current = (
            frame["short_interest_pct"]
            >= threshold
        )

        previous = frame[
            "previous_short_interest_pct"
        ]

        threshold_label = (
            f"{threshold:.1f}".replace(
                ".",
                "_",
            )
        )

        frame[
            f"above_{threshold_label}pct"
        ] = current

        frame[
            f"entered_above_{threshold_label}pct"
        ] = (
            previous.notna()
            & (previous < threshold)
            & current
        )

        frame[
            f"exited_below_{threshold_label}pct"
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
) -> tuple[pd.DataFrame, dict[str, int]]:
    lookup = build_price_lookup(
        prices
    )

    rows: list[dict[str, Any]] = []

    matched = 0
    unmatched = 0
    matched_by_isin = 0
    matched_by_issuer = 0

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
            if normalize_text(row.isin):
                mapping_source = "isin"
            else:
                mapping_source = "issuer"

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
            unmatched += 1
            continue

        dates = series["date"].to_numpy(
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

        if mapping_source == "isin":
            matched_by_isin += 1
        elif mapping_source == "issuer":
            matched_by_issuer += 1

        entry = series.iloc[
            entry_idx
        ]

        entry_price = float(
            entry["close"]
        )

        result = row._asdict()

        result["price_date"] = entry["date"]

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

        rows.append(
            result
        )

    return (
        pd.DataFrame(rows),
        {
            "fi_rows": int(len(fi)),
            "matched_rows": matched,
            "unmatched_rows": unmatched,
            "matched_by_isin": matched_by_isin,
            "matched_by_issuer": matched_by_issuer,
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
                )
                .dt.strftime(
                    "%Y-%m-%d"
                )
            )

    frame = frame.astype(object)

    frame = frame.where(
        pd.notna(frame),
        None,
    )

    return frame


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
        "Featurejobb: startar."
    )

    fi = load_fi()

    print(
        "Featurejobb: "
        f"{len(fi):,} "
        "FI-observationer lästa."
    )

    price_file = find_price_file()

    prices = load_prices(
        price_file
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

    validate_output_columns(
        result
    )

    result = clean_for_json(
        result
    )

    write_jsonl(
        result
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
        "fi_rows": len(fi),
        "price_rows": len(prices),
        "feature_rows": len(result),
        "matched_fi_rows": stats[
            "matched_rows"
        ],
        "unmatched_fi_rows": stats[
            "unmatched_rows"
        ],
        "matched_by_isin": stats[
            "matched_by_isin"
        ],
        "matched_by_issuer": stats[
            "matched_by_issuer"
        ],
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
        f"{stats['matched_rows']:,} "
        "matchade, "
        f"{stats['unmatched_rows']:,} "
        "omatchade."
    )

    print(
        "Featurejobb: "
        f"{stats['matched_by_isin']:,} "
        "matchade via ISIN, "
        f"{stats['matched_by_issuer']:,} "
        "via issuer."
    )

    print(
        "Featurejobb: "
        f"{len(result):,} rader "
        f"-> {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
