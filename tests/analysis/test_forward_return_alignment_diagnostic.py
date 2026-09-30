from __future__ import annotations

import math

import numpy as np
import pandas as pd

from analysis.feature_config import (
    FEATURE_GLOB,
    OUTPUT_DIR,
    PRICE_DIR,
)
from analysis.feature_prices import (
    find_price_files,
    load_prices,
)
from analysis.features_qc import (
    check_forward_return_alignment,
)


FORWARD_WINDOWS = (1, 5, 20, 60)
MAX_EXAMPLES = 30
TOLERANCE = 1e-9


def _load_features() -> pd.DataFrame:
    paths = sorted(
        OUTPUT_DIR.glob(FEATURE_GLOB)
    )

    if not paths:
        raise FileNotFoundError(
            f"Inga feature-chunks hittades i {OUTPUT_DIR}"
        )

    frames = [
        pd.read_json(
            path,
            lines=True,
        )
        for path in paths
    ]

    return pd.concat(
        frames,
        ignore_index=True,
    )


def _same(
    left,
    right,
) -> bool:
    left_missing = (
        left is None
        or pd.isna(left)
    )

    right_missing = (
        right is None
        or pd.isna(right)
    )

    if left_missing and right_missing:
        return True

    if left_missing != right_missing:
        return False

    return math.isclose(
        float(left),
        float(right),
        rel_tol=TOLERANCE,
        abs_tol=TOLERANCE,
    )


def _build_price_lookup(
    prices: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        symbol: group.sort_values(
            "date",
            kind="mergesort",
        ).reset_index(
            drop=True
        )
        for symbol, group in prices.groupby(
            "yahoo_symbol",
            sort=False,
        )
    }


def _mapping_source(
    row: pd.Series,
) -> str:
    value = row.get(
        "price_mapping_source"
    )

    if value is None or pd.isna(value):
        return "unknown"

    return str(value)


def test_forward_return_alignment_diagnostic():
    """
    Diagnostik för den faktiska forward_return_alignment-QC:n.

    Testet körs mot det aktuella feature- och prisarkivet och
    reproducerar QC-logiken samtidigt som det bryter ned eventuella
    fel per:

      - felkategori
      - forward-return-horisont
      - price_mapping_source
      - konkret feature-rad

    Testet ändrar ingen produktionskod.

    Avsikten är att ge tillräcklig information i GitHub Actions-loggen
    för att avgöra exakt varför forward_return_alignment eventuellt
    rapporterar FAIL.
    """

    features = _load_features()

    price_files = find_price_files(
        PRICE_DIR
    )

    prices = load_prices(
        price_files
    )

    qc_result = check_forward_return_alignment(
        features,
        prices,
    )

    matched = features.loc[
        features[
            "price_match_available"
        ].fillna(False)
        & features[
            "yahoo_symbol"
        ].notna()
        & features[
            "price_date"
        ].notna()
    ].copy()

    price_lookup = _build_price_lookup(
        prices
    )

    # ------------------------------------------------------------------
    # SECURITY KEY / YAHOO SYMBOL DIAGNOSTIC
    #
    # Produktionskoden grupperar prices på security_key.
    # Den oberoende diagnostiken ovan grupperar på yahoo_symbol.
    #
    # Om samma security_key förekommer för flera yahoo_symbol kan
    # produktionsserien därför skilja sig från den serie som används
    # för verifieringen ovan.
    # ------------------------------------------------------------------

    security_key_symbols = (
        prices.groupby(
            "security_key",
            sort=False,
        )["yahoo_symbol"]
        .nunique()
    )

    multi_symbol_keys = (
        security_key_symbols[
            security_key_symbols > 1
        ]
        .sort_values(
            ascending=False
        )
    )

    security_key_date_duplicates = (
        prices.groupby(
            ["security_key", "date"],
            sort=False,
        )
        .size()
    )

    duplicate_security_key_dates = (
        security_key_date_duplicates[
            security_key_date_duplicates > 1
        ]
        .sort_values(
            ascending=False
        )
    )

    print()
    print("=" * 72)
    print(
        "SECURITY KEY / YAHOO SYMBOL DIAGNOSTIC"
    )
    print("-" * 72)

    print(
        "Security keys with multiple Yahoo symbols: "
        f"{len(multi_symbol_keys):,}"
    )

    if not multi_symbol_keys.empty:
        print()
        print(
            "MULTI-SYMBOL SECURITY KEYS"
        )

        for key in multi_symbol_keys.index[:30]:
            symbols = sorted(
                prices.loc[
                    prices["security_key"] == key,
                    "yahoo_symbol",
                ]
                .dropna()
                .astype(str)
                .unique()
            )

            print(
                f"  {key}: {symbols}"
            )

    print()
    print(
        "Security key/date combinations with "
        f"multiple price rows: "
        f"{len(duplicate_security_key_dates):,}"
    )

    if not duplicate_security_key_dates.empty:
        print()
        print(
            "DUPLICATE SECURITY KEY / DATE"
        )

        for (
            key,
            date,
        ), count in duplicate_security_key_dates.head(30).items():
            rows = prices.loc[
                (
                    prices["security_key"] == key
                )
                & (
                    prices["date"] == date
                )
            ]

            print(
                f"  {key} | "
                f"{pd.Timestamp(date).date()} | "
                f"rows={count}"
            )

            for _, price_row in rows.iterrows():
                print(
                    "    "
                    f"symbol={price_row.get('yahoo_symbol')} "
                    f"close={price_row.get('close')}"
                )

    # ------------------------------------------------------------------
    # Reproduce production lookup exactly.
    # ------------------------------------------------------------------

    production_lookup = {
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

    missing_rows = matched.loc[
        matched[
            [
                "forward_return_1d",
                "forward_return_5d",
                "forward_return_20d",
                "forward_return_60d",
            ]
        ]
        .isna()
        .any(axis=1)
    ]

    print()
    print("=" * 72)
    print(
        "PRODUCTION LOOKUP DIAGNOSTIC"
    )
    print("-" * 72)

    print(
        f"Rows with at least one missing "
        f"forward return: {len(missing_rows):,}"
    )

    printed_missing = 0

    for index, row in missing_rows.iterrows():
        security = row.get(
            "security_key"
        )
        symbol = str(
            row.get("yahoo_symbol")
        )

        production_series = (
            production_lookup.get(
                security
            )
        )

        symbol_series = price_lookup.get(
            symbol
        )

        if (
            production_series is None
            or symbol_series is None
        ):
            continue

        price_date = pd.to_datetime(
            row.get("price_date"),
            errors="coerce",
        )

        if pd.isna(price_date):
            continue

        production_dates = (
            production_series["date"]
            .to_numpy(
                dtype="datetime64[ns]"
            )
        )

        symbol_dates = (
            symbol_series["date"]
            .to_numpy(
                dtype="datetime64[ns]"
            )
        )

        price_date_np = np.datetime64(
            price_date.to_datetime64(),
            "ns",
        )

        production_entry = int(
            np.searchsorted(
                production_dates,
                price_date_np,
                side="left",
            )
        )

        symbol_positions = (
            symbol_series.index[
                symbol_series["date"]
                == price_date
            ].tolist()
        )

        print()
        print(
            f"[MISSING ROW #{printed_missing + 1}]"
        )
        print(
            f"  feature_index: {index}"
        )
        print(
            f"  security_key: {security}"
        )
        print(
            f"  yahoo_symbol: {symbol}"
        )
        print(
            f"  mapping_source: "
            f"{row.get('price_mapping_source')}"
        )
        print(
            f"  price_date: "
            f"{price_date.date()}"
        )

        print()
        print(
            "  PRODUCTION SERIES"
        )
        print(
            f"    rows: "
            f"{len(production_series)}"
        )
        print(
            f"    symbols: "
            f"{sorted(production_series['yahoo_symbol'].dropna().astype(str).unique())}"
        )
        print(
            f"    entry_position: "
            f"{production_entry}"
        )

        if (
            0
            <= production_entry
            < len(production_series)
        ):
            start = max(
                0,
                production_entry - 2,
            )
            end = min(
                len(production_series),
                production_entry + 8,
            )

            for pos in range(
                start,
                end,
            ):
                price_row = (
                    production_series.iloc[pos]
                )

                print(
                    "      "
                    f"[{pos}] "
                    f"{pd.Timestamp(price_row['date']).date()} "
                    f"symbol={price_row['yahoo_symbol']} "
                    f"close={price_row['close']}"
                )

        print()
        print(
            "  DIAGNOSTIC YAHOO SERIES"
        )
        print(
            f"    rows: "
            f"{len(symbol_series)}"
        )
        print(
            f"    entry_positions: "
            f"{symbol_positions}"
        )

        if symbol_positions:
            symbol_entry = symbol_positions[0]

            start = max(
                0,
                symbol_entry - 2,
            )
            end = min(
                len(symbol_series),
                symbol_entry + 8,
            )

            for pos in range(
                start,
                end,
            ):
                price_row = (
                    symbol_series.iloc[pos]
                )

                print(
                    "      "
                    f"[{pos}] "
                    f"{pd.Timestamp(price_row['date']).date()} "
                    f"symbol={price_row['yahoo_symbol']} "
                    f"close={price_row['close']}"
                )

        print()
        print(
            "  STORED RETURNS"
        )

        for horizon in FORWARD_WINDOWS:
            print(
                f"    {horizon}d: "
                f"{row.get(f'forward_return_{horizon}d')}"
            )

        printed_missing += 1

        if printed_missing >= MAX_EXAMPLES:
            break

    print()
    print("=" * 72)
    print(
        "SECURITY KEY DIAGNOSTIC COMPLETE"
    )
    print("=" * 72)

    categories = {
        "verified_match": 0,
        "not_yet_verifiable": 0,
        "value_mismatch": 0,
        "price_date_not_found": 0,
        "missing_forward_return": 0,
    }

    by_horizon = {
        horizon: {
            category: 0
            for category in categories
        }
        for horizon in FORWARD_WINDOWS
    }

    by_mapping_source = {
        "isin": {
            category: 0
            for category in categories
        },
        "issuer": {
            category: 0
            for category in categories
        },
        "unknown": {
            category: 0
            for category in categories
        },
        "other": {
            category: 0
            for category in categories
        },
    }

    failure_examples = {
        "value_mismatch": [],
        "price_date_not_found": [],
        "missing_forward_return": [],
        "not_yet_verifiable": [],
    }

    def add_example(
        category: str,
        example: dict,
    ) -> None:
        if len(
            failure_examples[category]
        ) < MAX_EXAMPLES:
            failure_examples[
                category
            ].append(example)

    for _, row in matched.iterrows():
        symbol = str(
            row["yahoo_symbol"]
        )

        price_date = pd.to_datetime(
            row["price_date"],
            errors="coerce",
        )

        source = _mapping_source(
            row
        )

        if source not in by_mapping_source:
            source = "other"

        group = price_lookup.get(
            symbol
        )

        if group is None:
            categories[
                "price_date_not_found"
            ] += 1

            by_mapping_source[
                source
            ]["price_date_not_found"] += 1

            add_example(
                "price_date_not_found",
                {
                    "category": (
                        "price_date_not_found"
                    ),
                    "reason": (
                        "symbol_not_found_in_price_data"
                    ),
                    "price_mapping_source": (
                        _mapping_source(row)
                    ),
                    "security_key": row.get(
                        "security_key"
                    ),
                    "isin": row.get("isin"),
                    "issuer": row.get(
                        "issuer"
                    ),
                    "yahoo_symbol": symbol,
                    "price_date": str(
                        price_date.date()
                    ),
                },
            )

            continue

        positions = group.index[
            group["date"] == price_date
        ].tolist()

        if not positions:
            categories[
                "price_date_not_found"
            ] += 1

            by_mapping_source[
                source
            ]["price_date_not_found"] += 1

            add_example(
                "price_date_not_found",
                {
                    "category": (
                        "price_date_not_found"
                    ),
                    "reason": (
                        "date_not_found_for_symbol"
                    ),
                    "price_mapping_source": (
                        _mapping_source(row)
                    ),
                    "security_key": row.get(
                        "security_key"
                    ),
                    "isin": row.get("isin"),
                    "issuer": row.get(
                        "issuer"
                    ),
                    "yahoo_symbol": symbol,
                    "price_date": str(
                        price_date.date()
                    ),
                },
            )

            continue

        position = positions[0]

        base_price = float(
            group.loc[
                position,
                "close",
            ]
        )

        for window in FORWARD_WINDOWS:
            column = (
                f"forward_return_{window}d"
            )

            stored = row.get(
                column
            )

            target_position = (
                position + window
            )

            common = {
                "price_mapping_source": (
                    _mapping_source(row)
                ),
                "security_key": row.get(
                    "security_key"
                ),
                "isin": row.get("isin"),
                "issuer": row.get(
                    "issuer"
                ),
                "yahoo_symbol": symbol,
                "price_date": str(
                    price_date.date()
                ),
                "window": window,
                "stored": (
                    None
                    if stored is None
                    or pd.isna(stored)
                    else float(stored)
                ),
            }

            if target_position >= len(
                group
            ):
                category = (
                    "not_yet_verifiable"
                )

                categories[
                    category
                ] += 1

                by_horizon[
                    window
                ][category] += 1

                by_mapping_source[
                    source
                ][category] += 1

                add_example(
                    category,
                    {
                        **common,
                        "current_last_date": str(
                            pd.Timestamp(
                                group.iloc[-1][
                                    "date"
                                ]
                            ).date()
                        ),
                        "current_observations": len(
                            group
                        ),
                        "target_position": (
                            target_position
                        ),
                    },
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
                    base_price
                )
                or base_price <= 0
                or not math.isfinite(
                    future_price
                )
                or future_price <= 0
            ):
                category = (
                    "value_mismatch"
                )

                categories[
                    category
                ] += 1

                by_horizon[
                    window
                ][category] += 1

                by_mapping_source[
                    source
                ][category] += 1

                add_example(
                    category,
                    {
                        **common,
                        "reason": (
                            "invalid_price"
                        ),
                        "base_price": (
                            base_price
                        ),
                        "future_price": (
                            future_price
                        ),
                    },
                )

                continue

            expected = (
                future_price
                / base_price
                - 1.0
            )

            stored_missing = (
                stored is None
                or pd.isna(stored)
            )

            if stored_missing:
                category = (
                    "missing_forward_return"
                )

                categories[
                    category
                ] += 1

                by_horizon[
                    window
                ][category] += 1

                by_mapping_source[
                    source
                ][category] += 1

                add_example(
                    category,
                    {
                        **common,
                        "expected": (
                            expected
                        ),
                        "target_date": str(
                            pd.Timestamp(
                                group.iloc[
                                    target_position
                                ]["date"]
                            ).date()
                        ),
                    },
                )

                continue

            if not _same(
                stored,
                expected,
            ):
                category = (
                    "value_mismatch"
                )

                categories[
                    category
                ] += 1

                by_horizon[
                    window
                ][category] += 1

                by_mapping_source[
                    source
                ][category] += 1

                add_example(
                    category,
                    {
                        **common,
                        "expected": (
                            expected
                        ),
                        "difference": (
                            float(stored)
                            - expected
                        ),
                        "target_date": str(
                            pd.Timestamp(
                                group.iloc[
                                    target_position
                                ]["date"]
                            ).date()
                        ),
                    },
                )

                continue

            category = (
                "verified_match"
            )

            categories[
                category
            ] += 1

            by_horizon[
                window
            ][category] += 1

            by_mapping_source[
                source
            ][category] += 1

    print()
    print("=" * 72)
    print(
        "FORWARD RETURN ALIGNMENT - DETAILED DIAGNOSTIC"
    )
    print("=" * 72)

    print(
        f"Feature rows: {len(features):,}"
    )

    print(
        f"Matched rows: {len(matched):,}"
    )

    print(
        f"Price rows: {len(prices):,}"
    )

    print(
        f"QC status: {qc_result['status']}"
    )

    print()
    print("QC RESULT FROM PRODUCTION FUNCTION")
    print("-" * 72)

    for key in (
        "verified_match",
        "not_yet_verifiable",
        "value_mismatch",
        "price_date_not_found",
        "missing_forward_return",
    ):
        print(
            f"{key}: "
            f"{qc_result[key]:,}"
        )

    print()
    print("INDEPENDENT DETAILED CLASSIFICATION")
    print("-" * 72)

    for key, value in categories.items():
        print(
            f"{key}: {value:,}"
        )

    print()
    print("BY HORIZON")
    print("-" * 72)

    for window in FORWARD_WINDOWS:
        print(
            f"{window}d:"
        )

        for category, value in (
            by_horizon[
                window
            ].items()
        ):
            print(
                f"  {category}: {value:,}"
            )

    print()
    print("BY PRICE MAPPING SOURCE")
    print("-" * 72)

    for source, counts in (
        by_mapping_source.items()
    ):
        print(
            f"{source}:"
        )

        for category, value in (
            counts.items()
        ):
            print(
                f"  {category}: {value:,}"
            )

    print()
    print("FAILURE EXAMPLES")
    print("-" * 72)

    for category in (
        "value_mismatch",
        "price_date_not_found",
        "missing_forward_return",
    ):
        print()
        print(
            f"[{category}]"
        )

        examples = failure_examples[
            category
        ]

        if not examples:
            print("  NONE")
            continue

        for example in examples:
            print(
                f"  {example}"
            )

    print()
    print("NOT-YET-VERIFIABLE EXAMPLES")
    print("-" * 72)

    examples = failure_examples[
        "not_yet_verifiable"
    ]

    if not examples:
        print("  NONE")
    else:
        for example in examples:
            print(
                f"  {example}"
            )

    print()
    print("=" * 72)
    print(
        "DIAGNOSTIC COMPLETE"
    )
    print("=" * 72)

    # Detta är ett diagnostiskt test.
    #
    # Det ska inte själv göra workflowet rött. Om QC säger FAIL
    # är det just det vi vill undersöka i loggen.
    #
    # De befintliga unit-testerna ansvarar för att själva QC-regeln
    # beter sig korrekt.
    assert (
        categories["value_mismatch"]
        == qc_result["value_mismatch"]
    )

    assert (
        categories["price_date_not_found"]
        == qc_result[
            "price_date_not_found"
        ]
    )

    assert (
        categories["missing_forward_return"]
        == qc_result[
            "missing_forward_return"
        ]
    )

    assert (
        categories["not_yet_verifiable"]
        == qc_result[
            "not_yet_verifiable"
        ]
    )
