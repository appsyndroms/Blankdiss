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

    Ett target som ligger efter den aktuella
    prisseriens slut kan inte verifieras ännu.
    Det klassificeras därför som
    "not_yet_verifiable" och är inte ett QC-fel.

    Ett lagrat target som borde kunna verifieras
    men saknas är däremot ett fel.

    Ett numeriskt avvikande target är också ett fel.

    Returnerar:

        verified_match
        not_yet_verifiable
        value_mismatch
        price_date_not_found
        missing_forward_return
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

    price_frame = price_frame.loc[
        price_frame["date"].notna()
        & price_frame["close"].notna()
    ].copy()

    price_frame = price_frame.sort_values(
        [
            "yahoo_symbol",
            "date",
        ],
        kind="mergesort",
    )

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

    counts = {
        "verified_match": 0,
        "not_yet_verifiable": 0,
        "value_mismatch": 0,
        "price_date_not_found": 0,
        "missing_forward_return": 0,
    }

    examples = {
        "value_mismatch": [],
        "price_date_not_found": [],
        "missing_forward_return": [],
        "not_yet_verifiable": [],
    }

    max_examples = 20

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
            counts[
                "price_date_not_found"
            ] += 1

            if len(
                examples[
                    "price_date_not_found"
                ]
            ) < max_examples:
                examples[
                    "price_date_not_found"
                ].append(
                    {
                        "yahoo_symbol": symbol,
                        "price_date": (
                            price_date
                            .date()
                            .isoformat()
                        ),
                        "reason": (
                            "symbol_not_found_in_price_data"
                        ),
                    }
                )

            continue

        positions = group.index[
            group["date"] == price_date
        ].tolist()

        if not positions:
            counts[
                "price_date_not_found"
            ] += 1

            if len(
                examples[
                    "price_date_not_found"
                ]
            ) < max_examples:
                examples[
                    "price_date_not_found"
                ].append(
                    {
                        "yahoo_symbol": symbol,
                        "price_date": (
                            price_date
                            .date()
                            .isoformat()
                        ),
                        "reason": (
                            "price_date_not_found"
                        ),
                    }
                )

            continue

        position = positions[0]

        base_price = float(
            group.loc[
                position,
                "close",
            ]
        )

        if (
            not math.isfinite(
                base_price
            )
            or base_price <= 0
        ):
            counts[
                "value_mismatch"
            ] += 1

            if len(
                examples[
                    "value_mismatch"
                ]
            ) < max_examples:
                examples[
                    "value_mismatch"
                ].append(
                    {
                        "yahoo_symbol": symbol,
                        "price_date": (
                            price_date
                            .date()
                            .isoformat()
                        ),
                        "reason": (
                            "invalid_base_price"
                        ),
                    }
                )

            continue

        for window in FORWARD_WINDOWS:
            column = (
                f"forward_return_{window}d"
            )

            stored = getattr(
                row,
                column,
                None,
            )

            target_position = (
                position + window
            )

            # Prisarkivet räcker ännu inte fram till
            # den observation som krävs för detta target.
            #
            # Detta är förväntat i ett löpande dataset och
            # ska därför inte ge FAIL.
            if (
                target_position
                >= len(group)
            ):
                counts[
                    "not_yet_verifiable"
                ] += 1

                if len(
                    examples[
                        "not_yet_verifiable"
                    ]
                ) < max_examples:
                    examples[
                        "not_yet_verifiable"
                    ].append(
                        {
                            "yahoo_symbol": symbol,
                            "price_date": (
                                price_date
                                .date()
                                .isoformat()
                            ),
                            "window": window,
                            "stored": (
                                None
                                if stored is None
                                or pd.isna(stored)
                                else float(stored)
                            ),
                            "current_last_date": (
                                group.iloc[-1][
                                    "date"
                                ]
                                .date()
                                .isoformat()
                            ),
                        }
                    )

                continue

            future_price = float(
                group.loc[
                    target_position,
                    "close",
                ]
            )

            if (
                not math.isfinite(
                    future_price
                )
                or future_price <= 0
            ):
                counts[
                    "value_mismatch"
                ] += 1

                if len(
                    examples[
                        "value_mismatch"
                    ]
                ) < max_examples:
                    examples[
                        "value_mismatch"
                    ].append(
                        {
                            "yahoo_symbol":
