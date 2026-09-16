"""
QC för Blankdiss FI + pris-feature-dataset.

Kontrollerar bland annat:

- FI -> pris-tidsriktning
- avstånd mellan FI-datum och prisdatum
- FI-observationer före första tillgängliga pris
- instrumentidentitet
- Yahoo-symboler
- forward returns
- potentiell data leakage
- dubbletter
- saknade värden
- extrema matchningsavstånd

Jobbet ändrar inte feature-dataseten.

Output:
    data/processed/analysis/features_qc.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FEATURES_DIR = (
    ROOT
    / "data"
    / "processed"
    / "analysis"
)

FEATURES_GLOB = "features_*.jsonl"

QC_PATH = (
    ROOT
    / "data"
    / "processed"
    / "analysis"
    / "features_qc.json"
)

MAPPING_PATH = (
    ROOT
    / "data"
    / "analysis"
    / "instrument_map.json"
)

PRICE_DIR = (
    ROOT
    / "data"
    / "raw"
    / "prices"
)

FORWARD_WINDOWS = (
    1,
    5,
    20,
    60,
)


def load_jsonl(
    path: Path,
) -> pd.DataFrame:
    """Läser en JSONL-fil."""

    if not path.exists():
        raise FileNotFoundError(
            f"Saknar fil: {path}"
        )

    frame = pd.read_json(
        path,
        lines=True,
    )

    if frame.empty:
        raise ValueError(
            f"Filen är tom: {path}"
        )

    return frame


def load_feature_chunks() -> pd.DataFrame:
    """
    Läser hela feature-datasetet från alla chunks.

    Datasetet består av:
        features_0001.jsonl
        features_0002.jsonl
        ...

    Alla chunks slås ihop till ett DataFrame för QC.
    """

    paths = sorted(
        FEATURES_DIR.glob(
            FEATURES_GLOB
        )
    )

    if not paths:
        raise FileNotFoundError(
            "Saknar feature-chunks i "
            f"{FEATURES_DIR}: "
            f"{FEATURES_GLOB}"
        )

    frames: list[pd.DataFrame] = []

    for path in paths:
        print(
            f"  Läser {path.name}"
        )

        frame = load_jsonl(
            path
        )

        frames.append(
            frame
        )

    features = pd.concat(
        frames,
        ignore_index=True,
    )

    if features.empty:
        raise ValueError(
            "Feature-datasetet är tomt."
        )

    print(
        f"  Läste {len(paths)} feature-chunks."
    )

    return features


def load_mapping() -> dict[str, dict[str, Any]]:
    """Läser instrumentmappningen."""

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Saknar mapping: {MAPPING_PATH}"
        )

    with MAPPING_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            "instrument_map.json måste "
            "innehålla ett objekt."
        )

    return data


def normalise_text(
    value: Any,
) -> str | None:
    """Normaliserar textvärden."""

    if value is None:
        return None

    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def finite_numeric(
    value: Any,
) -> bool:
    """Returnerar True om värdet är ett ändligt tal."""

    if value is None:
        return False

    if pd.isna(value):
        return False

    try:
        return math.isfinite(
            float(value)
        )
    except (
        TypeError,
        ValueError,
    ):
        return False


def check_required_columns(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Kontrollerar att alla förväntade kolumner finns."""

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
    }

    required.update(
        f"forward_return_{window}d"
        for window in FORWARD_WINDOWS
    )

    missing = sorted(
        required.difference(
            frame.columns
        )
    )

    return {
        "status": (
            "FAIL"
            if missing
            else "PASS"
        ),
        "missing_columns": missing,
    }


def prepare_dates(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Konverterar datumkolumner."""

    result = frame.copy()

    result[
        "snapshot_date"
    ] = pd.to_datetime(
        result[
            "snapshot_date"
        ],
        errors="coerce",
    )

    result[
        "price_date"
    ] = pd.to_datetime(
        result[
            "price_date"
        ],
        errors="coerce",
    )

    return result


def check_date_integrity(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Kontrollerar datum och FI -> pris-riktning."""

    snapshot_invalid = int(
        frame[
            "snapshot_date"
        ].isna().sum()
    )

    matched = frame.loc[
        frame[
            "price_match_available"
        ].fillna(False)
    ].copy()

    # price_date är endast obligatoriskt för matchade rader.
    # Omatchade FI-rader har avsiktligt price_date = NaT.
    invalid_matched_price_dates = int(
        matched[
            "price_date"
        ].isna().sum()
    )

    wrong_direction = matched.loc[
        matched[
            "price_date"
        ]
        < matched[
            "snapshot_date"
        ]
    ]

    negative_days = matched.loc[
        matched[
            "days_from_fi_to_price"
        ] < 0
    ]

    status = "PASS"

    if (
        snapshot_invalid
        or invalid_matched_price_dates > 0
        or len(wrong_direction) > 0
        or len(negative_days) > 0
    ):
        status = "FAIL"

    return {
        "status": status,
        "invalid_snapshot_dates": snapshot_invalid,
        "invalid_price_dates": (
            invalid_matched_price_dates
        ),
        "price_before_fi": int(
            len(wrong_direction)
        ),
        "negative_days_from_fi_to_price": int(
            len(negative_days)
        ),
    }


def check_match_distances(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Kontrollerar hur långt efter FI-datum
    priset ligger.

    Detta är en diagnostik.

    Långa avstånd är inte automatiskt fel.
    """

    matched = frame.loc[
        frame[
            "price_match_available"
        ].fillna(False)
    ].copy()

    if matched.empty:
        return {
            "status": "WARN",
            "matched_rows": 0,
            "same_day": 0,
            "one_day": 0,
            "two_days": 0,
            "three_days": 0,
            "four_to_five_days": 0,
            "more_than_five_days": 0,
            "more_than_ten_days": 0,
            "max_calendar_days": None,
            "median_calendar_days": None,
        }

    days = pd.to_numeric(
        matched[
            "days_from_fi_to_price"
        ],
        errors="coerce",
    )

    return {
        "status": "PASS",
        "matched_rows": int(
            len(matched)
        ),
        "same_day": int(
            (days == 0).sum()
        ),
        "one_day": int(
            (days == 1).sum()
        ),
        "two_days": int(
            (days == 2).sum()
        ),
        "three_days": int(
            (days == 3).sum()
        ),
        "four_to_five_days": int(
            days.between(
                4,
                5,
            ).sum()
        ),
        "more_than_five_days": int(
            (days > 5).sum()
        ),
        "more_than_ten_days": int(
            (days > 10).sum()
        ),
        "max_calendar_days": int(
            days.max()
        ),
        "median_calendar_days": float(
            days.median()
        ),
    }


def check_first_price_problem(
    frame: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict[str, Any]:
    """
    Identifierar FI-observationer som ligger före
    första tillgängliga pris för samma Yahoo-symbol.

    Detta är viktigt eftersom merge_asof(direction=forward)
    annars kan ge ett pris långt efter FI-datumet.
    """

    first_prices = (
        prices.groupby(
            "yahoo_symbol",
            as_index=True,
        )[
            "date"
        ]
        .min()
    )

    check = frame.loc[
        frame[
            "yahoo_symbol"
        ].notna()
    ].copy()

    check[
        "first_price_date"
    ] = check[
        "yahoo_symbol"
    ].map(
        first_prices
    )

    affected = check.loc[
        check[
            "first_price_date"
        ].notna()
        & (
            check[
                "snapshot_date"
            ]
            < check[
                "first_price_date"
            ]
        )
    ]

    missing_symbol_in_prices = check.loc[
        check[
            "first_price_date"
        ].isna()
    ]

    return {
        "status": (
            "WARN"
            if (
                len(affected) > 0
                or len(missing_symbol_in_prices) > 0
            )
            else "PASS"
        ),
        "fi_rows_before_first_price": int(
            len(affected)
        ),
        "symbols_without_price_history": int(
            missing_symbol_in_prices[
                "yahoo_symbol"
            ].nunique()
        ),
        "rows_without_price_history": int(
            len(missing_symbol_in_prices)
        ),
    }


def check_mapping_integrity(
    frame: pd.DataFrame,
    mapping: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Kontrollerar att Yahoo-symboler i features
    faktiskt finns i instrumentmappningen.
    """

    mapping_symbols = {
        normalise_text(
            entry.get(
                "yahoo_symbol"
            )
        )
        for entry in mapping.values()
    }

    mapping_symbols.discard(None)

    feature_symbols = {
        normalise_text(value)
        for value in frame[
            "yahoo_symbol"
        ].dropna()
    }

    unknown_symbols = sorted(
        feature_symbols
        - mapping_symbols
    )

    missing_mapping = frame.loc[
        frame[
            "yahoo_symbol"
        ].notna()
        & ~frame[
            "yahoo_symbol"
        ].isin(
            mapping_symbols
        )
    ]

    return {
        "status": (
            "FAIL"
            if unknown_symbols
            else "PASS"
        ),
        "feature_symbols": int(
            len(feature_symbols)
        ),
        "mapping_symbols": int(
            len(mapping_symbols)
        ),
        "unknown_yahoo_symbols": unknown_symbols,
        "rows_with_unknown_symbols": int(
            len(missing_mapping)
        ),
    }


def check_mapping_sources(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Kontrollerar hur prisidentiteterna matchades."""

    source = (
        frame[
            "price_mapping_source"
        ]
        .fillna("unmatched")
        .astype(str)
    )

    counts = (
        source.value_counts()
        .to_dict()
    )

    unexpected = sorted(
        set(counts)
        - {
            "isin",
            "issuer",
            "unmatched",
        }
    )

    return {
        "status": (
            "FAIL"
            if unexpected
            else "PASS"
        ),
        "sources": {
            str(key): int(value)
            for key, value in counts.items()
        },
        "unexpected_sources": unexpected,
    }


def check_identity_consistency(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Kontrollerar om samma security_key har
    flera Yahoo-symboler.

    Det kan vara legitimt vid symbolbyten,
    så detta är WARN och inte FAIL.
    """

    working = frame.loc[
        frame[
            "yahoo_symbol"
        ].notna()
    ].copy()

    symbol_counts = (
        working.groupby(
            "security_key"
        )[
            "yahoo_symbol"
        ]
        .nunique()
    )

    multiple = symbol_counts.loc[
        symbol_counts > 1
    ]

    examples: list[dict[str, Any]] = []

    for security_key in (
        multiple.index[:20]
    ):
        symbols = sorted(
            working.loc[
                working[
                    "security_key"
                ]
                == security_key,
                "yahoo_symbol",
            ]
            .dropna()
            .unique()
            .tolist()
        )

        examples.append(
            {
                "security_key": str(
                    security_key
                ),
                "yahoo_symbols": symbols,
            }
        )

    return {
        "status": (
            "WARN"
            if len(multiple) > 0
            else "PASS"
        ),
        "security_keys_with_multiple_yahoo_symbols": int(
            len(multiple)
        ),
        "examples": examples,
    }


def check_duplicate_features(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Kontrollerar dubbletter på
    security_key + snapshot_date.
    """

    duplicates = frame.duplicated(
        subset=[
            "security_key",
            "snapshot_date",
        ],
        keep=False,
    )

    duplicate_rows = frame.loc[
        duplicates
    ]

    return {
        "status": (
            "FAIL"
            if duplicates.any()
            else "PASS"
        ),
        "duplicate_rows": int(
            len(duplicate_rows)
        ),
        "duplicate_groups": int(
            duplicate_rows[
                [
                    "security_key",
                    "snapshot_date",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        ),
    }


def check_price_values(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Kontrollerar close-värden."""

    close = pd.to_numeric(
        frame[
            "close_on_signal_date"
        ],
        errors="coerce",
    )

    matched = frame[
        "price_match_available"
    ].fillna(False)

    missing_close = int(
        (
            matched
            & close.isna()
        ).sum()
    )

    nonpositive_close = int(
        (
            matched
            & close.notna()
            & (close <= 0)
        ).sum()
    )

    nonfinite_close = 0

    for value in close.loc[
        matched
        & close.notna()
    ]:
        if not finite_numeric(value):
            nonfinite_close += 1

    status = "PASS"

    if (
        missing_close
        or nonpositive_close
        or nonfinite_close
    ):
        status = "FAIL"

    return {
        "status": status,
        "missing_close_with_price_match": missing_close,
        "nonpositive_close": nonpositive_close,
        "nonfinite_close": nonfinite_close,
    }


def check_forward_returns(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Kontrollerar forward returns.

    Vi förväntar oss:

    - None/NaN för slutet av varje prisserie
    - ändliga värden annars
    - inga forward returns utan pris
    """

    results: dict[str, Any] = {}

    overall_status = "PASS"

    for window in FORWARD_WINDOWS:
        column = (
            f"forward_return_{window}d"
        )

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        matched = frame[
            "price_match_available"
        ].fillna(False)

        nonfinite = 0

        for value in values.loc[
            values.notna()
        ]:
            if not finite_numeric(value):
                nonfinite += 1

        invalid = int(
            (
                matched
                & values.notna()
                & ~values.map(
                    finite_numeric
                )
            ).sum()
        )

        if nonfinite or invalid:
            overall_status = "FAIL"

        results[column] = {
            "non_null": int(
                values.notna().sum()
            ),
            "null": int(
                values.isna().sum()
            ),
            "nonfinite": int(
                nonfinite
            ),
            "invalid_with_price": int(
                invalid
            ),
        }

    return {
        "status": overall_status,
        "windows": results,
    }


def check_forward_return_alignment(
    frame: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict[str, Any]:
    """
    Oberoende kontroll av forward-return-logiken.

    För varje signalpris hämtas de framtida
    handelsdagarna direkt från prisserien och
    jämförs med de lagrade forward returns.

    Detta fångar fel i shift-logiken.
    """

    price_frame = prices.copy()

    price_frame["date"] = pd.to_datetime(
        price_frame["date"],
        errors="coerce",
    )

    price_frame["close"] = pd.to_numeric(
        price_frame["close"],
        errors="coerce",
    )

    price_frame = price_frame.sort_values(
        [
            "yahoo_symbol",
            "date",
        ],
        kind="mergesort",
    )

    mismatches: list[dict[str, Any]] = []

    matched = frame.loc[
        frame[
            "price_match_available"
        ].fillna(False)
        & frame[
            "yahoo_symbol"
        ].notna()
        & frame[
            "price_date"
        ].notna()
    ].copy()

    price_groups = {
        symbol: group.reset_index(
            drop=True
        )
        for symbol, group
        in price_frame.groupby(
            "yahoo_symbol",
            sort=False,
        )
    }

    for row in matched.itertuples(
        index=False
    ):
        symbol = normalise_text(
            getattr(
                row,
                "yahoo_symbol",
                None,
            )
        )

        price_date = pd.Timestamp(
            getattr(
                row,
                "price_date",
            )
        )

        group = price_groups.get(
            symbol
        )

        if group is None:
            continue

        positions = group.index[
            group["date"]
            == price_date
        ].tolist()

        if not positions:
            mismatches.append(
                {
                    "reason": "price_date_not_found",
                    "yahoo_symbol": symbol,
                    "price_date": (
                        price_date
                        .date()
                        .isoformat()
                    ),
                }
            )

            if len(mismatches) >= 20:
                break

            continue

        position = positions[0]

        base_price = float(
            group.loc[
                position,
                "close",
            ]
        )

        for window in FORWARD_WINDOWS:
            target_position = (
                position + window
            )

            column = (
                f"forward_return_{window}d"
            )

            stored = getattr(
                row,
                column,
                None,
            )

            if (
                target_position
                >= len(group)
            ):
                continue

            future_price = float(
                group.loc[
                    target_position,
                    "close",
                ]
            )

            expected = (
                future_price
                / base_price
                - 1.0
            )

            if pd.isna(stored):
                mismatches.append(
                    {
                        "reason": "missing_forward_return",
                        "yahoo_symbol": symbol,
                        "price_date": (
                            price_date
                            .date()
                            .isoformat()
                        ),
                        "window": window,
                    }
                )
            elif not math.isclose(
                float(stored),
                expected,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                mismatches.append(
                    {
                        "reason": "forward_return_mismatch",
                        "yahoo_symbol": symbol,
                        "price_date": (
                            price_date
                            .date()
                            .isoformat()
                        ),
                        "window": window,
                        "stored": float(
                            stored
                        ),
                        "expected": expected,
                    }
                )

            if len(mismatches) >= 20:
                break

        if len(mismatches) >= 20:
            break

    return {
        "status": (
            "FAIL"
            if mismatches
            else "PASS"
        ),
        "mismatches_found": int(
            len(mismatches)
        ),
        "examples": mismatches[:20],
    }


def check_price_leakage(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Kontrollerar att framtida prisfält inte används
    som samtidiga features.

    Forward-return-kolumnerna är tillåtna som targets.

    Alla andra prisrelaterade kolumner måste representera
    signalpris eller information som fanns senast vid
    FI-observationen.
    """

    forbidden_patterns = (
        "future_",
        "next_",
        "forward_price",
    )

    forbidden_columns = [
        column
        for column in frame.columns
        if any(
            pattern in column.lower()
            for pattern in forbidden_patterns
        )
        and not column.startswith(
            "forward_return_"
        )
    ]

    return {
        "status": (
            "FAIL"
            if forbidden_columns
            else "PASS"
        ),
        "forbidden_future_price_columns": sorted(
            forbidden_columns
        ),
    }


def check_unmatched_rows(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Summerar FI-rader utan pris."""

    matched = frame[
        "price_match_available"
    ].fillna(False)

    return {
        "status": (
            "WARN"
            if (~matched).any()
            else "PASS"
        ),
        "total_rows": int(
            len(frame)
        ),
        "rows_with_price": int(
            matched.sum()
        ),
        "rows_without_price": int(
            (~matched).sum()
        ),
        "symbols_without_price": int(
            frame.loc[
                ~matched,
                "yahoo_symbol",
            ]
            .dropna()
            .nunique()
        ),
    }


def check_feature_numeric_values(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Kontrollerar centrala numeriska featurefält."""

    columns = [
        "short_interest_pct",
        "active_holders",
        "max_individual_position_pct",
        "max_position_share_pct",
        "short_interest_delta_pp",
        "short_interest_relative_change",
        "short_interest_acceleration_pp",
        "holder_delta",
        "max_position_delta_pp",
        "concentration_delta_pp",
        "days_since_previous_fi_observation",
    ]

    invalid_by_column: dict[str, int] = {}

    for column in columns:
        if column not in frame.columns:
            continue

        values = frame[column]

        invalid = 0

        for value in values.loc[
            values.notna()
        ]:
            if not finite_numeric(value):
                invalid += 1

        if invalid:
            invalid_by_column[
                column
            ] = invalid

    return {
        "status": (
            "FAIL"
            if invalid_by_column
            else "PASS"
        ),
        "invalid_by_column": invalid_by_column,
    }


def check_threshold_logic(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """
    Kontrollerar att entered/exited-flaggor
    faktiskt följer threshold-logiken.

    Vi testar endast rader där både aktuell och
    föregående short-interest finns.

    Viktigt:
    itertuples(name=None) används här eftersom
    pandas annars kan ändra kolumnnamn som börjar
    med underscore, exempelvis "_current".
    """

    failures: list[dict[str, Any]] = []

    for threshold in (
        1.0,
        2.0,
        3.0,
        5.0,
    ):
        suffix = str(
            threshold
        ).replace(
            ".",
            "_",
        )

        above_column = (
            f"above_{suffix}pct"
        )

        entered_column = (
            f"entered_above_{suffix}pct"
        )

        exited_column = (
            f"exited_below_{suffix}pct"
        )

        working = frame.copy()

        working[
            "_current"
        ] = pd.to_numeric(
            working[
                "short_interest_pct"
            ],
            errors="coerce",
        )

        working = working.sort_values(
            [
                "security_key",
                "snapshot_date",
            ],
            kind="mergesort",
        )

        working[
            "_previous"
        ] = working.groupby(
            "security_key",
            sort=False,
        )[
            "_current"
        ].shift(1)

        columns_to_check = [
            "security_key",
            "snapshot_date",
            "_current",
            "_previous",
            above_column,
            entered_column,
            exited_column,
        ]

        for (
            security_key,
            snapshot_date,
            current,
            previous,
            actual_above,
            actual_entered,
            actual_exited,
        ) in working[
            columns_to_check
        ].itertuples(
            index=False,
            name=None,
        ):
            if (
                pd.isna(current)
                or pd.isna(previous)
            ):
                continue

            expected_above = (
                current >= threshold
            )

            expected_entered = (
                current >= threshold
                and previous < threshold
            )

            expected_exited = (
                current < threshold
                and previous >= threshold
            )

            actual_above = bool(
                actual_above
            )

            actual_entered = bool(
                actual_entered
            )

            actual_exited = bool(
                actual_exited
            )

            if (
                actual_above
                != expected_above
                or actual_entered
                != expected_entered
                or actual_exited
                != expected_exited
            ):
                failures.append(
                    {
                        "threshold": threshold,
                        "security_key": str(
                            security_key
                        ),
                        "snapshot_date": str(
                            snapshot_date
                        ),
                        "current": float(
                            current
                        ),
                        "previous": float(
                            previous
                        ),
                        "expected_above": bool(
                            expected_above
                        ),
                        "actual_above": bool(
                            actual_above
                        ),
                        "expected_entered": bool(
                            expected_entered
                        ),
                        "actual_entered": bool(
                            actual_entered
                        ),
                        "expected_exited": bool(
                            expected_exited
                        ),
                        "actual_exited": bool(
                            actual_exited
                        ),
                    }
                )

                if len(failures) >= 20:
                    break

        if len(failures) >= 20:
            break

    return {
        "status": (
            "FAIL"
            if failures
            else "PASS"
        ),
        "failures_found": int(
            len(failures)
        ),
        "examples": failures[:20],
    }


def load_price_files() -> list[Path]:
    """Hittar alla lokala prisfiler."""

    files = sorted(
        PRICE_DIR.glob(
            "prices_*.jsonl"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"Inga prisfiler hittades i "
            f"{PRICE_DIR}"
        )

    return files


def load_prices(
    paths: list[Path],
) -> pd.DataFrame:
    """Läser all prisdata för oberoende QC."""

    frames: list[pd.DataFrame] = []

    for path in paths:
        print(
            f"  Läser prisfil {path.name}"
        )

        prices = load_jsonl(
            path
        )

        required = {
            "date",
            "yahoo_symbol",
            "close",
        }

        missing = required.difference(
            prices.columns
        )

        if missing:
            raise ValueError(
                f"Prisfil {path} saknar "
                "kolumner: "
                + ", ".join(
                    sorted(missing)
                )
            )

        frames.append(
            prices
        )

    prices = pd.concat(
        frames,
        ignore_index=True,
    )

    prices["date"] = pd.to_datetime(
        prices["date"],
        errors="coerce",
    )

    prices["close"] = pd.to_numeric(
        prices["close"],
        errors="coerce",
    )

    prices["yahoo_symbol"] = (
        prices[
            "yahoo_symbol"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    prices = prices.loc[
        prices["date"].notna()
        & prices[
            "yahoo_symbol"
        ].ne("")
        & prices["close"].notna()
    ].copy()

    # Om samma symbol + datum finns i mer än en
    # prisfil ska samma observation inte räknas flera gånger.
    prices = (
        prices
        .sort_values(
            [
                "yahoo_symbol",
                "date",
            ],
            kind="mergesort",
        )
        .drop_duplicates(
            subset=[
                "yahoo_symbol",
                "date",
            ],
            keep="last",
        )
    )

    return prices.sort_values(
        [
            "yahoo_symbol",
            "date",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )


def build_summary(
    checks: dict[str, dict[str, Any]],
) -> str:
    """Bestämmer övergripande QC-status."""

    statuses = [
        value.get(
            "status"
        )
        for value in checks.values()
    ]

    if "FAIL" in statuses:
        return "FAIL"

    if "WARN" in statuses:
        return "WARN"

    return "PASS"


def main() -> None:
    print(
        "Feature-QC: startar."
    )

    feature_paths = sorted(
        FEATURES_DIR.glob(
            FEATURES_GLOB
        )
    )

    print(
        f"Feature-chunks: "
        f"{len(feature_paths)}"
    )

    features = load_feature_chunks()

    print(
        f"Feature-rader: {len(features):,}"
    )

    column_check = (
        check_required_columns(
            features
        )
    )

    if column_check[
        "status"
    ] == "FAIL":
        report = {
            "dataset": (
                "Blankdiss FI + price features"
            ),
            "status": "FAIL",
            "checks": {
                "required_columns": column_check,
            },
        }

        QC_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with QC_PATH.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                report,
                handle,
                ensure_ascii=False,
                indent=2,
            )

        raise SystemExit(
            "Feature-QC FAIL: "
            "obligatoriska kolumner saknas."
        )

    features = prepare_dates(
        features
    )

    mapping = load_mapping()

    price_files = load_price_files()

    prices = load_prices(
        price_files
    )

    checks: dict[
        str,
        dict[str, Any],
    ] = {}

    checks[
        "required_columns"
    ] = column_check

    checks[
        "date_integrity"
    ] = check_date_integrity(
        features
    )

    checks[
        "match_distances"
    ] = check_match_distances(
        features
    )

    checks[
        "first_price_problem"
    ] = check_first_price_problem(
        features,
        prices,
    )

    checks[
        "mapping_integrity"
    ] = check_mapping_integrity(
        features,
        mapping,
    )

    checks[
        "mapping_sources"
    ] = check_mapping_sources(
        features
    )

    checks[
        "identity_consistency"
    ] = check_identity_consistency(
        features
    )

    checks[
        "duplicate_features"
    ] = check_duplicate_features(
        features
    )

    checks[
        "price_values"
    ] = check_price_values(
        features
    )

    checks[
        "forward_returns"
    ] = check_forward_returns(
        features
    )

    checks[
        "forward_return_alignment"
    ] = check_forward_return_alignment(
        features,
        prices,
    )

    checks[
        "price_leakage"
    ] = check_price_leakage(
        features
    )

    checks[
        "unmatched_rows"
    ] = check_unmatched_rows(
        features
    )

    checks[
        "feature_numeric_values"
    ] = check_feature_numeric_values(
        features
    )

    checks[
        "threshold_logic"
    ] = check_threshold_logic(
        features
    )

    status = build_summary(
        checks
    )

    report = {
        "dataset": (
            "Blankdiss FI + price features"
        ),
        "status": status,
        "feature_files": [
            str(
                path.relative_to(
                    ROOT
                )
            )
            for path in feature_paths
        ],
        "price_files": [
            str(
                path.relative_to(
                    ROOT
                )
            )
            for path in price_files
        ],
        "rows": int(
            len(features)
        ),
        "symbols": int(
            features[
                "yahoo_symbol"
            ]
            .dropna()
            .nunique()
        ),
        "fi_period": {
            "start": (
                features[
                    "snapshot_date"
                ]
                .min()
                .date()
                .isoformat()
                if features[
                    "snapshot_date"
                ].notna().any()
                else None
            ),
            "end": (
                features[
                    "snapshot_date"
                ]
                .max()
                .date()
                .isoformat()
                if features[
                    "snapshot_date"
                ].notna().any()
                else None
            ),
        },
        "checks": checks,
    }

    QC_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with QC_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "Feature-QC"
    )
    print(
        f"Status: {status}"
    )

    for name, result in checks.items():
        print(
            f"{name}: "
            f"{result.get('status')}"
        )

    print()
    print(
        f"QC-rapport: {QC_PATH}"
    )

    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
