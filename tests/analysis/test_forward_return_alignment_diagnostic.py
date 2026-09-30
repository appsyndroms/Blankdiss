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
MAX_TRACE_ROWS = 12
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


def _build_production_lookup(
    prices: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Exakt samma lookup-struktur som add_forward_returns().
    """
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


def _mapping_source(
    row: pd.Series,
) -> str:
    value = row.get(
        "price_mapping_source"
    )

    if value is None or pd.isna(value):
        return "unknown"

    return str(value)


def _trace_production_row(
    row: pd.Series,
    production_lookup: dict[str, pd.DataFrame],
) -> None:
    """
    Följ exakt den lookup- och indexlogik som add_forward_returns()
    använder för en konkret feature-rad.

    Funktionen ändrar ingenting.
    """

    security_key = row.get(
        "security_key"
    )

    yahoo_symbol = row.get(
        "yahoo_symbol"
    )

    price_date = pd.to_datetime(
        row.get("price_date"),
        errors="coerce",
    )

    series = production_lookup.get(
        security_key
    )

    print()
    print(
        "  PRODUCTION TRACE"
    )
    print(
        "  " + "-" * 66
    )

    print(
        f"  security_key: {security_key}"
    )

    print(
        f"  feature yahoo_symbol: {yahoo_symbol}"
    )

    print(
        f"  feature price_date: "
        f"{None if pd.isna(price_date) else price_date.date()}"
    )

    if series is None:
        print(
            "  RESULT: production lookup returned None"
        )
        print(
            "  -> add_forward_returns() would skip this row."
        )
        return

    print(
        f"  production series rows: {len(series)}"
    )

    symbols = sorted(
        series[
            "yahoo_symbol"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    print(
        f"  production series symbols: {symbols}"
    )

    duplicate_dates = (
        series.groupby(
            "date",
            sort=False,
        )
        .size()
    )

    duplicate_dates = duplicate_dates[
        duplicate_dates > 1
    ]

    print(
        "  production duplicate dates: "
        f"{len(duplicate_dates):,}"
    )

    if pd.isna(price_date):
        print(
            "  RESULT: invalid price_date"
        )
        print(
            "  -> add_forward_returns() would skip this row."
        )
        return

    dates = series[
        "date"
    ].to_numpy(
        dtype="datetime64[ns]"
    )

    price_date_np = np.datetime64(
        price_date.to_datetime64(),
        "ns",
    )

    entry_idx = int(
        np.searchsorted(
            dates,
            price_date_np,
            side="left",
        )
    )

    print(
        f"  production entry_idx: {entry_idx}"
    )

    if entry_idx >= len(series):
        print(
            "  RESULT: entry_idx outside production series"
        )
        print(
            "  -> add_forward_returns() would skip this row."
        )
        return

    entry_row = series.iloc[
        entry_idx
    ]

    entry_date = pd.Timestamp(
        entry_row["date"]
    )

    entry_symbol = entry_row.get(
        "yahoo_symbol"
    )

    entry_price = float(
        entry_row["close"]
    )

    print(
        f"  production entry_date: "
        f"{entry_date.date()}"
    )

    print(
        f"  production entry_symbol: "
        f"{entry_symbol}"
    )

    print(
        f"  production entry_close: "
        f"{entry_price}"
    )

    if (
        not np.isfinite(entry_price)
        or entry_price <= 0
    ):
        print(
            "  RESULT: invalid entry price"
        )
        print(
            "  -> add_forward_returns() would skip all horizons."
        )
        return

    print()
    print(
        "  PRODUCTION SERIES AROUND ENTRY"
    )

    start = max(
        0,
        entry_idx - 2,
    )

    end = min(
        len(series),
        entry_idx + MAX_TRACE_ROWS,
    )

    for position in range(
        start,
        end,
    ):
        price_row = series.iloc[
            position
        ]

        marker = (
            " <-- ENTRY"
            if position == entry_idx
            else ""
        )

        print(
            "    "
            f"[{position}] "
            f"{pd.Timestamp(price_row['date']).date()} "
            f"symbol={price_row.get('yahoo_symbol')} "
            f"close={price_row.get('close')}"
            f"{marker}"
        )

    print()
    print(
        "  HORIZON TRACE"
    )

    for horizon in FORWARD_WINDOWS:
        target_idx = (
            entry_idx + horizon
        )

        stored = row.get(
            f"forward_return_{horizon}d"
        )

        print()
        print(
            f"    {horizon}d:"
        )

        print(
            f"      target_idx: {target_idx}"
        )

        print(
            f"      stored: {stored}"
        )

        if target_idx >= len(series):
            print(
                "      RESULT: target_idx outside series"
            )
            print(
                "      -> add_forward_returns() would leave NaN."
            )
            continue

        target_row = series.iloc[
            target_idx
        ]

        target_date = pd.Timestamp(
            target_row["date"]
        )

        target_symbol = target_row.get(
            "yahoo_symbol"
        )

        target_price = float(
            target_row["close"]
        )

        print(
            f"      target_date: {target_date.date()}"
        )

        print(
            f"      target_symbol: {target_symbol}"
        )

        print(
            f"      target_close: {target_price}"
        )

        if (
            not np.isfinite(target_price)
            or target_price <= 0
        ):
            print(
                "      RESULT: invalid target price"
            )
            print(
                "      -> add_forward_returns() would leave NaN."
            )
            continue

        expected = (
            target_price
            / entry_price
            - 1.0
        )

        print(
            f"      calculated: {expected}"
        )

        if (
            stored is None
            or pd.isna(stored)
        ):
            print(
                "      RESULT: CALCULATION SUCCEEDS "
                "BUT STORED VALUE IS NaN"
            )
        elif _same(
            stored,
            expected,
        ):
            print(
                "      RESULT: STORED VALUE MATCHES"
            )
        else:
            print(
                "      RESULT: STORED VALUE MISMATCH"
            )


def test_forward_return_alignment_diagnostic():
    """
    Diagnostik för forward_return_alignment-QC.

    Testet:
      1. kör produktions-QC,
      2. reproducerar QC oberoende,
      3. klassificerar alla rader,
      4. samlar konkreta missing_forward_return-fall,
      5. följer dessa fall genom exakt samma lookup/indexlogik
         som add_forward_returns() använder.

    Testet ändrar ingen produktionskod.
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

    production_lookup = (
        _build_production_lookup(
            prices
        )
    )

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

    for index, row in matched.iterrows():
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
            category = (
                "price_date_not_found"
            )

            categories[
                category
            ] += 1

            by_mapping_source[
                source
            ][category] += 1

            add_example(
                category,
                {
                    "feature_index": index,
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
                    "reason": (
                        "symbol_not_found_in_price_data"
                    ),
                },
            )

            continue

        positions = group.index[
            group["date"] == price_date
        ].tolist()

        if not positions:
            category = (
                "price_date_not_found"
            )

            categories[
                category
            ] += 1

            by_mapping_source[
                source
            ][category] += 1

            add_example(
                category,
                {
                    "feature_index": index,
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
                    "reason": (
                        "date_not_found_for_symbol"
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
                "feature_index": index,
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

    # ------------------------------------------------------------------
    # SAMMANFATTNING
    # ------------------------------------------------------------------

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
    print(
        "QC RESULT FROM PRODUCTION FUNCTION"
    )
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
    print(
        "INDEPENDENT DETAILED CLASSIFICATION"
    )
    print("-" * 72)

    for key, value in categories.items():
        print(
            f"{key}: {value:,}"
        )

    print()
    print(
        "BY HORIZON"
    )
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
    print(
        "BY PRICE MAPPING SOURCE"
    )
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
    print(
        "FAILURE EXAMPLES"
    )
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
    print(
        "NOT-YET-VERIFIABLE EXAMPLES"
    )
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

    # ------------------------------------------------------------------
    # SECURITY KEY DIAGNOSTIC
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

    security_key_date_counts = (
        prices.groupby(
            ["security_key", "date"],
            sort=False,
        )
        .size()
    )

    duplicate_security_key_dates = (
        security_key_date_counts[
            security_key_date_counts > 1
        ]
        .sort_values(
            ascending=False
        )
    )

    print()
    print("=" * 72)
    print(
        "SECURITY KEY DIAGNOSTIC"
    )
    print("=" * 72)

    print(
        "Security keys with multiple Yahoo symbols: "
        f"{len(multi_symbol_keys):,}"
    )

    if not multi_symbol_keys.empty:
        for key in multi_symbol_keys.head(
            MAX_EXAMPLES
        ).index:
            symbols = sorted(
                prices.loc[
                    prices[
                        "security_key"
                    ] == key,
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
        "multiple price rows: "
        f"{len(duplicate_security_key_dates):,}"
    )

    if not duplicate_security_key_dates.empty:
        for (
            key,
            date,
        ), count in (
            duplicate_security_key_dates
            .head(MAX_EXAMPLES)
            .items()
        ):
            print(
                f"  {key} | "
                f"{pd.Timestamp(date).date()} | "
                f"rows={count}"
            )

            rows = prices.loc[
                (
                    prices[
                        "security_key"
                    ] == key
                )
                & (
                    prices[
                        "date"
                    ] == date
                )
            ]

            for _, price_row in rows.iterrows():
                print(
                    "    "
                    f"symbol={price_row.get('yahoo_symbol')} "
                    f"close={price_row.get('close')}"
                )

    # ------------------------------------------------------------------
    # EXAKT PRODUKTIONS-SPÅRNING AV MISSING-FALLEN
    # ------------------------------------------------------------------

    missing_examples = failure_examples[
        "missing_forward_return"
    ]

    print()
    print("=" * 72)
    print(
        "PRODUCTION TRACE FOR MISSING FORWARD RETURNS"
    )
    print("=" * 72)

    print(
        "Missing forward-return cases: "
        f"{len(missing_examples):,}"
    )

    for number, example in enumerate(
        missing_examples,
        start=1,
    ):
        print()
        print(
            "=" * 72
        )
        print(
            f"MISSING CASE #{number}"
        )
        print(
            "=" * 72
        )

        print(
            f"feature_index: "
            f"{example.get('feature_index')}"
        )

        print(
            f"security_key: "
            f"{example.get('security_key')}"
        )

        print(
            f"issuer: "
            f"{example.get('issuer')}"
        )

        print(
            f"isin: "
            f"{example.get('isin')}"
        )

        print(
            f"yahoo_symbol: "
            f"{example.get('yahoo_symbol')}"
        )

        print(
            f"mapping_source: "
            f"{example.get('price_mapping_source')}"
        )

        print(
            f"price_date: "
            f"{example.get('price_date')}"
        )

        print(
            f"window: "
            f"{example.get('window')}"
        )

        print(
            f"expected: "
            f"{example.get('expected')}"
        )

        _trace_production_row(
            features.loc[
                example["feature_index"]
            ],
            production_lookup,
        )

    print()
    print("=" * 72)
    print(
        "DIAGNOSTIC COMPLETE"
    )
    print("=" * 72)

    # Diagnostiskt test: produktions-QC och oberoende klassificering
    # ska fortfarande överensstämma.
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
