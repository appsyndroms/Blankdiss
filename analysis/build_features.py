"""
Bygger den första kombinerade FI + pris-feature-dataseten för Blankdiss.

Input:
    data/processed/fi/aggregate/reconstructed.jsonl
    data/raw/prices/prices_*.jsonl

Output:
    data/processed/analysis/features.jsonl
    data/processed/analysis/features_metadata.json

Viktiga principer:
- FI:s rekonstruerade serie är synlig blankning från publicerade
  individuella positioner, inte FI:s officiella aggregat.
- Saknad FI-observation tolkas INTE som noll blankning.
- ISIN används som primär identitet när den finns.
- Issuer används som fallback när ISIN saknas.
- Pris kopplas via instrumentets Yahoo-symbol från instrument_map.json.
- Forward returns beräknas från handelsdagar, inte kalenderdagar.
- Vi använder framtida priser endast som target/utfall.
  De får inte läcka in i samtidiga features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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

MAPPING_PATH = (
    ROOT
    / "data"
    / "analysis"
    / "instrument_map.json"
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


FORWARD_WINDOWS = (
    1,
    5,
    20,
    60,
)

THRESHOLDS = (
    1.0,
    2.0,
    3.0,
    5.0,
)


def load_jsonl(
    path: Path,
) -> pd.DataFrame:
    """Läser en JSONL-fil till DataFrame."""

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


def find_price_file() -> Path:
    """Hittar senaste prisfil."""

    files = sorted(
        PRICE_DIR.glob(
            "prices_*.jsonl"
        )
    )

    if not files:
        raise FileNotFoundError(
            "Ingen prices_*.jsonl hittades "
            f"i {PRICE_DIR}"
        )

    return files[-1]


def load_mapping() -> dict[str, dict[str, Any]]:
    """Läser instrumentmappningen."""

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Saknar instrumentmappning: "
            f"{MAPPING_PATH}"
        )

    with MAPPING_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            "instrument_map.json måste "
            "innehålla ett JSON-objekt."
        )

    return data


def normalise_text(
    value: Any,
) -> str | None:
    """Normaliserar textfält."""

    if value is None:
        return None

    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def build_mapping_lookup(
    mapping: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, str],
    dict[str, str],
]:
    """
    Bygger två lookup-tabeller:

    ISIN -> Yahoo-symbol
    issuer -> Yahoo-symbol

    ISIN är primär identitet.

    Issuer används bara som fallback och endast
    när mappingen har exakt en Yahoo-symbol för
    issuer-namnet.
    """

    isin_symbols: dict[
        str,
        set[str],
    ] = {}

    issuer_symbols: dict[
        str,
        set[str],
    ] = {}

    for entry in mapping.values():
        isin = normalise_text(
            entry.get("isin")
        )

        issuer = normalise_text(
            entry.get("issuer")
        )

        symbol = normalise_text(
            entry.get("yahoo_symbol")
        )

        if not symbol:
            continue

        if isin:
            isin_symbols.setdefault(
                isin,
                set(),
            ).add(symbol)

        if issuer:
            issuer_symbols.setdefault(
                issuer.casefold(),
                set(),
            ).add(symbol)

    isin_lookup = {
        isin: next(iter(symbols))
        for isin, symbols
        in isin_symbols.items()
        if len(symbols) == 1
    }

    issuer_lookup = {
        issuer: next(iter(symbols))
        for issuer, symbols
        in issuer_symbols.items()
        if len(symbols) == 1
    }

    return (
        isin_lookup,
        issuer_lookup,
    )


def attach_yahoo_symbols(
    fi: pd.DataFrame,
    mapping: dict[str, dict[str, Any]],
) -> tuple[
    pd.DataFrame,
    dict[str, int],
]:
    """
    Kopplar Yahoo-symbol till FI-dataseten.

    Prioritet:
        1. ISIN
        2. exakt issuer-fallback

    Ingen gissning via ticker eller fuzzy matching.
    """

    (
        isin_lookup,
        issuer_lookup,
    ) = build_mapping_lookup(
        mapping
    )

    frame = fi.copy()

    symbols: list[str | None] = []
    sources: list[str | None] = []

    stats = {
        "matched_by_isin": 0,
        "matched_by_issuer": 0,
        "unmatched": 0,
    }

    for row in frame.itertuples(
        index=False
    ):
        isin = normalise_text(
            getattr(
                row,
                "isin",
                None,
            )
        )

        issuer = normalise_text(
            getattr(
                row,
                "issuer",
                None,
            )
        )

        symbol = None
        source = None

        if isin and isin in isin_lookup:
            symbol = isin_lookup[isin]
            source = "isin"
            stats["matched_by_isin"] += 1

        elif (
            issuer
            and issuer.casefold()
            in issuer_lookup
        ):
            symbol = issuer_lookup[
                issuer.casefold()
            ]
            source = "issuer"
            stats["matched_by_issuer"] += 1

        else:
            stats["unmatched"] += 1

        symbols.append(symbol)
        sources.append(source)

    frame[
        "yahoo_symbol"
    ] = symbols

    frame[
        "price_mapping_source"
    ] = sources

    return frame, stats


def prepare_fi(
    fi: pd.DataFrame,
) -> pd.DataFrame:
    """Förbereder FI-dataseten."""

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
        fi.columns
    )

    if missing:
        raise ValueError(
            "FI-dataset saknar kolumner: "
            + ", ".join(
                sorted(missing)
            )
        )

    frame = fi.copy()

    frame[
        "snapshot_date"
    ] = pd.to_datetime(
        frame[
            "snapshot_date"
        ],
        errors="coerce",
    )

    frame = frame.loc[
        frame[
            "snapshot_date"
        ].notna()
    ].copy()

    frame["issuer"] = (
        frame["issuer"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    frame["isin"] = (
        frame["isin"]
        .where(
            frame["isin"].notna(),
            None,
        )
    )

    frame["isin"] = frame[
        "isin"
    ].map(
        normalise_text
    )

    numeric_columns = [
        "short_interest_pct",
        "active_holders",
        "max_individual_position_pct",
        "max_position_share_pct",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.sort_values(
        [
            "issuer",
            "isin",
            "snapshot_date",
        ],
        kind="mergesort",
    )

    return frame


def add_identity_key(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Skapar en intern analysidentitet.

    ISIN är primär.

    Om ISIN saknas används issuer som fallback.

    Detta är en analysnyckel, inte ett påstående om
    juridisk/instrumentmässig identitet.
    """

    result = frame.copy()

    result["security_key"] = (
        result["isin"]
        .where(
            result["isin"].notna()
            & (
                result["isin"]
                .astype(str)
                .str.strip()
                != ""
            ),
            "ISSUER:"
            + result["issuer"]
            .astype(str)
            .str.strip(),
        )
    )

    return result


def add_fi_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Skapar samtidiga FI-features.

    Gruppningen sker per security_key.

    Viktigt:
    pct_change() kan bli missvisande kring FI:s
    synlighetsgräns. Därför sparas både absolut
    förändring och relativ förändring.
    """

    result = frame.copy()

    result = result.sort_values(
        [
            "security_key",
            "snapshot_date",
        ],
        kind="mergesort",
    )

    grouped = result.groupby(
        "security_key",
        sort=False,
    )

    result[
        "short_interest_delta_pp"
    ] = grouped[
        "short_interest_pct"
    ].diff()

    previous_short = grouped[
        "short_interest_pct"
    ].shift(1)

    result[
        "short_interest_relative_change"
    ] = (
        result[
            "short_interest_delta_pp"
        ]
        / previous_short
    )

    previous_delta = grouped[
        "short_interest_delta_pp"
    ].shift(1)

    result[
        "short_interest_acceleration_pp"
    ] = (
        result[
            "short_interest_delta_pp"
        ]
        - previous_delta
    )

    result[
        "holder_delta"
    ] = grouped[
        "active_holders"
    ].diff()

    result[
        "max_position_delta_pp"
    ] = grouped[
        "max_individual_position_pct"
    ].diff()

    result[
        "concentration_delta_pp"
    ] = grouped[
        "max_position_share_pct"
    ].diff()

    result[
        "days_since_previous_fi_observation"
    ] = (
        grouped[
            "snapshot_date"
        ].diff()
        .dt.days
    )

    result[
        "fi_observation_number"
    ] = grouped.cumcount() + 1

    for threshold in THRESHOLDS:
        suffix = str(
            threshold
        ).replace(
            ".",
            "_",
        )

        previous = grouped[
            "short_interest_pct"
        ].shift(1)

        current = result[
            "short_interest_pct"
        ]

        result[
            f"above_{suffix}pct"
        ] = (
            current >= threshold
        )

        result[
            f"entered_above_{suffix}pct"
        ] = (
            (current >= threshold)
            & (
                previous < threshold
            )
        )

        result[
            f"exited_below_{suffix}pct"
        ] = (
            (current < threshold)
            & (
                previous >= threshold
            )
        )

    return result


def load_prices(
    path: Path,
) -> pd.DataFrame:
    """Läser och validerar prisdata."""

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
            "Prisdata saknar kolumner: "
            + ", ".join(
                sorted(missing)
            )
        )

    prices = prices.copy()

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
        & (
            prices[
                "yahoo_symbol"
            ]
            != ""
        )
        & prices["close"].notna()
    ].copy()

    prices = prices.sort_values(
        [
            "yahoo_symbol",
            "date",
        ],
        kind="mergesort",
    )

    duplicate_mask = prices.duplicated(
        subset=[
            "yahoo_symbol",
            "date",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            "Prisdata innehåller "
            f"{int(duplicate_mask.sum())} "
            "rader med duplicerad "
            "(yahoo_symbol, date)."
        )

    return prices


def calculate_forward_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Skapar forward returns på handelsdagsnivå.

    return_1d:
        nästa tillgängliga handelsdag

    return_5d:
        femte framtida handelsdagen

    osv.

    Inga framtida värden används för FI-features.
    """

    result = prices.copy()

    result = result.sort_values(
        [
            "yahoo_symbol",
            "date",
        ],
        kind="mergesort",
    )

    grouped = result.groupby(
        "yahoo_symbol",
        sort=False,
    )

    for window in FORWARD_WINDOWS:
        future_close = grouped[
            "close"
        ].shift(
            -window
        )

        result[
            f"forward_return_{window}d"
        ] = (
            future_close
            / result["close"]
            - 1.0
        )

    return result


def merge_fi_and_prices(
    fi: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
]:
    """
    Kopplar FI-observation till första priset på
    eller efter FI-datumet.

    Joinen görs separat per Yahoo-symbol.

    Detta är medvetet gjort i stället för en enda
    merge_asof(..., by="yahoo_symbol"), eftersom
    pandas kräver strikt sorterade tidsnycklar och
    stora multi-instrument-dataset annars kan ge
    "left keys must be sorted".

    Vi använder inte framtida prisdata som feature.
    Forward returns är targets/utfall.
    """

    price_frame = calculate_forward_returns(
        prices
    )

    price_columns = [
        "yahoo_symbol",
        "date",
        "close",
    ]

    price_columns.extend(
        [
            f"forward_return_{window}d"
            for window in FORWARD_WINDOWS
        ]
    )

    price_frame = price_frame[
        price_columns
    ].copy()

    price_frame = price_frame.rename(
        columns={
            "date": "price_date",
            "close": "close_on_signal_date",
        }
    )

    fi_frame = fi.copy()

    fi_with_symbol = fi_frame.loc[
        fi_frame[
            "yahoo_symbol"
        ].notna()
    ].copy()

    fi_without_symbol = (
        len(fi_frame)
        - len(fi_with_symbol)
    )

    if fi_with_symbol.empty:
        return (
            pd.DataFrame(
                columns=list(
                    fi_frame.columns
                )
            ),
            {
                "fi_rows": int(
                    len(fi_frame)
                ),
                "fi_rows_without_yahoo_symbol": int(
                    fi_without_symbol
                ),
                "fi_rows_with_yahoo_symbol": 0,
                "merged_rows": 0,
                "merged_rows_with_price": 0,
                "merged_rows_without_price": 0,
            },
        )

    merged_parts: list[
        pd.DataFrame
    ] = []

    symbols = sorted(
        fi_with_symbol[
            "yahoo_symbol"
        ]
        .dropna()
        .unique()
    )

    for symbol in symbols:
        fi_group = fi_with_symbol.loc[
            fi_with_symbol[
                "yahoo_symbol"
            ]
            == symbol
        ].copy()

        price_group = price_frame.loc[
            price_frame[
                "yahoo_symbol"
            ]
            == symbol
        ].copy()

        fi_group = fi_group.sort_values(
            "snapshot_date",
            kind="mergesort",
        )

        price_group = price_group.sort_values(
            "price_date",
            kind="mergesort",
        )

        if price_group.empty:
            fi_group[
                "price_date"
            ] = pd.NaT

            fi_group[
                "close_on_signal_date"
            ] = None

            for window in FORWARD_WINDOWS:
                fi_group[
                    f"forward_return_{window}d"
                ] = None

            merged_parts.append(
                fi_group
            )

            continue

        merged_group = pd.merge_asof(
            fi_group,
            price_group,
            left_on="snapshot_date",
            right_on="price_date",
            direction="forward",
            allow_exact_matches=True,
        )

        merged_parts.append(
            merged_group
        )

    merged = pd.concat(
        merged_parts,
        ignore_index=True,
    )

    merged[
        "days_from_fi_to_price"
    ] = (
        merged["price_date"]
        - merged["snapshot_date"]
    ).dt.days

    merged[
        "price_match_available"
    ] = merged[
        "price_date"
    ].notna()

    stats = {
        "fi_rows": int(
            len(fi_frame)
        ),
        "fi_rows_without_yahoo_symbol": int(
            fi_without_symbol
        ),
        "fi_rows_with_yahoo_symbol": int(
            len(fi_with_symbol)
        ),
        "merged_rows": int(
            len(merged)
        ),
        "merged_rows_with_price": int(
            merged[
                "price_match_available"
            ].sum()
        ),
        "merged_rows_without_price": int(
            (
                ~merged[
                    "price_match_available"
                ]
            ).sum()
        ),
    }

    return (
        merged,
        stats,
    )


def clean_feature_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Städar och sorterar slutresultatet."""

    result = frame.copy()

    result = result.sort_values(
        [
            "security_key",
            "snapshot_date",
        ],
        kind="mergesort",
    )

    result["snapshot_date"] = (
        result[
            "snapshot_date"
        ].dt.strftime(
            "%Y-%m-%d"
        )
    )

    result["price_date"] = (
        result[
            "price_date"
        ].dt.strftime(
            "%Y-%m-%d"
        )
    )

    boolean_columns = [
        column
        for column in result.columns
        if column.startswith(
            "above_"
        )
        or column.startswith(
            "entered_"
        )
        or column.startswith(
            "exited_"
        )
        or column == "price_match_available"
    ]

    for column in boolean_columns:
        result[column] = (
            result[column]
            .fillna(False)
            .astype(bool)
        )

    return result


def write_jsonl(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """Skriver DataFrame som JSONL."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = frame.to_dict(
        orient="records"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            cleaned: dict[str, Any] = {}

            for key, value in record.items():
                if pd.isna(value):
                    cleaned[key] = None
                else:
                    cleaned[key] = value

            handle.write(
                json.dumps(
                    cleaned,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
            handle.write("\n")


def build_metadata(
    *,
    fi: pd.DataFrame,
    features: pd.DataFrame,
    price_file: Path,
    mapping_stats: dict[str, int],
    merge_stats: dict[str, int],
) -> dict[str, Any]:
    """Bygger metadata för feature-dataseten."""

    dates = pd.to_datetime(
        features[
            "snapshot_date"
        ],
        errors="coerce",
    )

    matched = features.loc[
        features[
            "price_match_available"
        ]
    ]

    metadata = {
        "dataset": "Blankdiss FI + price features",
        "version": 1,
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
            "mapping_file": str(
                MAPPING_PATH.relative_to(
                    ROOT
                )
            ),
        },
        "rows": {
            "fi_input": int(
                len(fi)
            ),
            "features": int(
                len(features)
            ),
            "with_price": int(
                len(matched)
            ),
        },
        "period": {
            "start": (
                dates.min()
                .date()
                .isoformat()
                if not dates.empty
                else None
            ),
            "end": (
                dates.max()
                .date()
                .isoformat()
                if not dates.empty
                else None
            ),
        },
        "identity": {
            "primary": "isin",
            "fallback": "issuer",
            "price_key": "yahoo_symbol",
        },
        "mapping": mapping_stats,
        "merge": merge_stats,
        "forward_return_windows": list(
            FORWARD_WINDOWS
        ),
        "short_interest_thresholds": list(
            THRESHOLDS
        ),
        "important_semantics": [
            (
                "Missing FI observation is not "
                "interpreted as zero short interest."
            ),
            (
                "FI short interest is reconstructed "
                "from visible individual positions."
            ),
            (
                "Price is matched to the first "
                "available trading day on or after "
                "the FI snapshot date."
            ),
            (
                "Forward returns are targets/outcomes "
                "and must not be used as contemporaneous "
                "features."
            ),
        ],
    }

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bygg Blankdiss FI + pris-"
            "feature-dataset."
        )
    )

    parser.add_argument(
        "--price-file",
        default=None,
        help=(
            "Valfri prisfil. "
            "Standard: senaste prices_*.jsonl."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(
        "Featurejobb: startar."
    )

    print(
        f"FI-fil: {FI_PATH}"
    )

    fi = load_jsonl(
        FI_PATH
    )

    print(
        "Featurejobb: "
        f"{len(fi)} FI-rader lästa."
    )

    fi = prepare_fi(
        fi
    )

    fi = add_identity_key(
        fi
    )

    mapping = load_mapping()

    (
        fi,
        mapping_stats,
    ) = attach_yahoo_symbols(
        fi,
        mapping,
    )

    print(
        "Mapping: "
        f"{mapping_stats['matched_by_isin']} "
        "FI-rader matchade via ISIN, "
        f"{mapping_stats['matched_by_issuer']} "
        "via issuer, "
        f"{mapping_stats['unmatched']} "
        "utan prisidentitet."
    )

    fi = add_fi_features(
        fi
    )

    if args.price_file:
        price_file = Path(
            args.price_file
        )

        if not price_file.is_absolute():
            price_file = (
                ROOT
                / price_file
            )
    else:
        price_file = find_price_file()

    print(
        "Prisfil: "
        f"{price_file}"
    )

    prices = load_prices(
        price_file
    )

    print(
        "Featurejobb: "
        f"{len(prices)} prisrader lästa."
    )

    (
        features,
        merge_stats,
    ) = merge_fi_and_prices(
        fi,
        prices,
    )

    print(
        "Prisjoin: "
        f"{merge_stats['merged_rows_with_price']} "
        "FI-rader fick prisdata, "
        f"{merge_stats['merged_rows_without_price']} "
        "saknar pris."
    )

    features = clean_feature_frame(
        features
    )

    write_jsonl(
        features,
        OUTPUT_PATH,
    )

    metadata = build_metadata(
        fi=fi,
        features=features,
        price_file=price_file,
        mapping_stats=mapping_stats,
        merge_stats=merge_stats,
    )

    OUTPUT_DIR.mkdir(
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

    print()
    print(
        "Featurejobb: klart."
    )
    print(
        "Features: "
        f"{OUTPUT_PATH}"
    )
    print(
        "Metadata: "
        f"{METADATA_PATH}"
    )
    print(
        "Feature-rader: "
        f"{len(features)}"
    )


if __name__ == "__main__":
    main()
