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
from analysis.feature_returns import (
    add_forward_returns,
)


FORWARD_WINDOWS = (1, 5, 20, 60)

MAX_EXAMPLES = 30

TOLERANCE = 1e-9


def _load_features() -> pd.DataFrame:
    """Läser befintliga feature-chunks utan att bygga om datasetet."""

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
    """Jämför två numeriska värden med liten tolerans."""

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


def _build_symbol_lookup(
    prices: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Samma prisserie-identitet som add_forward_returns()
    använder: yahoo_symbol.
    """

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


def _expected_return(
    series: pd.DataFrame,
    price_date: pd.Timestamp,
    horizon: int,
) -> tuple[str, float | None, str | None]:
    """
    Beräknar vad forward-return ska vara för en konkret
    yahoo-symbolserie.

    Returnerar:
      status,
      expected_value,
      target_date

    status:
      "verified"
      "not_yet_verifiable"
      "invalid"
    """

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

    if entry_idx >= len(series):
        return (
            "not_yet_verifiable",
            None,
            None,
        )

    target_idx = (
        entry_idx + horizon
    )

    if target_idx >= len(series):
        return (
            "not_yet_verifiable",
            None,
            None,
        )

    entry_price = float(
        series.iloc[
            entry_idx
        ]["close"]
    )

    target_price = float(
        series.iloc[
            target_idx
        ]["close"]
    )

    if (
        not np.isfinite(entry_price)
        or entry_price <= 0
        or not np.isfinite(target_price)
        or target_price <= 0
    ):
        return (
            "invalid",
            None,
            None,
        )

    expected = (
        target_price
        / entry_price
        - 1.0
    )

    target_date = str(
        pd.Timestamp(
            series.iloc[
                target_idx
            ]["date"]
        ).date()
    )

    return (
        "verified",
        expected,
        target_date,
    )


def test_forward_return_alignment_against_existing_dataset():
    """
    Fristående diagnostik för forward returns.

    Testet bygger INTE om feature-datasetet.

    Det:
      1. läser befintliga feature-chunks,
      2. läser befintliga prisfiler,
      3. kör endast add_forward_returns(),
      4. verifierar resultatet mot prisserierna,
      5. skiljer verifierbara targets från framtida targets,
      6. visar konkreta fel.

    Testet är därför möjligt att köra direkt efter en kodändring
    utan att först köra Force/build_features.
    """

    features = _load_features()

    price_files = find_price_files(
        PRICE_DIR
    )

    prices = load_prices(
        price_files
    )

    print()
    print("=" * 72)
    print(
        "FORWARD RETURN ALIGNMENT - STANDALONE TEST"
    )
    print("=" * 72)

    print(
        f"Feature rows: {len(features):,}"
    )

    print(
        f"Price rows: {len(prices):,}"
    )

    print(
        f"Price files: {len(price_files):,}"
    )

    required_columns = {
        "yahoo_symbol",
        "price_date",
        "price_match_available",
    }

    missing_columns = sorted(
        required_columns.difference(
            features.columns
        )
    )

    assert not missing_columns, (
        "Feature-dataset saknar kolumner: "
        + ", ".join(missing_columns)
    )

    # Endast rader som faktiskt har en prisidentitet
    # och en matchad prisdag kan verifieras.
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

    print(
        f"Matched rows: {len(matched):,}"
    )

    # --------------------------------------------------------------
    # KÖR ENDAST DEN PRODUKTIONSFUNKTION VI VILL TESTA
    # --------------------------------------------------------------

    recomputed = add_forward_returns(
        matched,
        prices,
    )

    # --------------------------------------------------------------
    # PRISLOOKUP FÖR OBEROENDE VERIFIERING
    # --------------------------------------------------------------

    price_lookup = _build_symbol_lookup(
        prices
    )

    categories = {
        "verified_match": 0,
        "not_yet_verifiable": 0,
        "missing_forward_return": 0,
        "value_mismatch": 0,
        "price_series_not_found": 0,
        "invalid_price": 0,
    }

    by_horizon = {
        horizon: {
            category: 0
            for category in categories
        }
        for horizon in FORWARD_WINDOWS
    }

    failures: list[dict] = []

    def add_failure(
        example: dict,
    ) -> None:
        if len(failures) < MAX_EXAMPLES:
            failures.append(example)

    # --------------------------------------------------------------
    # VERIFIERA VARJE MATCHAD RAD
    # --------------------------------------------------------------

    for index, row in matched.iterrows():
        symbol = str(
            row["yahoo_symbol"]
        )

        price_date = pd.to_datetime(
            row["price_date"],
            errors="coerce",
        )

        series = price_lookup.get(
            symbol
        )

        if series is None:
            categories[
                "price_series_not_found"
            ] += 1

            add_failure(
                {
                    "feature_index": index,
                    "security_key": row.get(
                        "security_key"
                    ),
                    "issuer": row.get(
                        "issuer"
                    ),
                    "isin": row.get(
                        "isin"
                    ),
                    "yahoo_symbol": symbol,
                    "price_date": str(
                        price_date.date()
                    ),
                    "reason": (
                        "yahoo_symbol_not_found"
                    ),
                }
            )

            continue

        for horizon in FORWARD_WINDOWS:
            column = (
                f"forward_return_{horizon}d"
            )

            stored = row.get(
                column
            )

            recomputed_value = (
                recomputed.loc[
                    index,
                    column,
                ]
            )

            status, expected, target_date = (
                _expected_return(
                    series,
                    price_date,
                    horizon,
                )
            )

            # ------------------------------------------------------
            # Framtida target finns ännu inte.
            # Detta är normalt och ska inte vara FAIL.
            # ------------------------------------------------------

            if status == (
                "not_yet_verifiable"
            ):
                categories[
                    "not_yet_verifiable"
                ] += 1

                by_horizon[
                    horizon
                ]["not_yet_verifiable"] += 1

                continue

            # ------------------------------------------------------
            # Ogiltigt pris.
            # Detta är ett faktiskt dataproblem.
            # ------------------------------------------------------

            if status == "invalid":
                categories[
                    "invalid_price"
                ] += 1

                by_horizon[
                    horizon
                ]["invalid_price"] += 1

                add_failure(
                    {
                        "feature_index": index,
                        "security_key": row.get(
                            "security_key"
                        ),
                        "issuer": row.get(
                            "issuer"
                        ),
                        "yahoo_symbol": symbol,
                        "price_date": str(
                            price_date.date()
                        ),
                        "window": horizon,
                        "stored": (
                            None
                            if pd.isna(stored)
                            else float(stored)
                        ),
                        "recomputed": (
                            None
                            if pd.isna(
                                recomputed_value
                            )
                            else float(
                                recomputed_value
                            )
                        ),
                        "reason": (
                            "invalid_price"
                        ),
                    }
                )

                continue

            # ------------------------------------------------------
            # Här finns targetpriset.
            # Nu måste production function ha producerat
            # ett värde.
            # ------------------------------------------------------

            if (
                recomputed_value is None
                or pd.isna(
                    recomputed_value
                )
            ):
                categories[
                    "missing_forward_return"
                ] += 1

                by_horizon[
                    horizon
                ]["missing_forward_return"] += 1

                add_failure(
                    {
                        "feature_index": index,
                        "security_key": row.get(
                            "security_key"
                        ),
                        "issuer": row.get(
                            "issuer"
                        ),
                        "isin": row.get(
                            "isin"
                        ),
                        "yahoo_symbol": symbol,
                        "price_date": str(
                            price_date.date()
                        ),
                        "window": horizon,
                        "stored": (
                            None
                            if pd.isna(stored)
                            else float(stored)
                        ),
                        "recomputed": None,
                        "expected": expected,
                        "target_date": target_date,
                        "reason": (
                            "production_function_returned_nan"
                        ),
                    }
                )

                continue

            # ------------------------------------------------------
            # Kontrollera den omräknade production-funktionen
            # mot den oberoende prisberäkningen.
            # ------------------------------------------------------

            if not _same(
                recomputed_value,
                expected,
            ):
                categories[
                    "value_mismatch"
                ] += 1

                by_horizon[
                    horizon
                ]["value_mismatch"] += 1

                add_failure(
                    {
                        "feature_index": index,
                        "security_key": row.get(
                            "security_key"
                        ),
                        "issuer": row.get(
                            "issuer"
                        ),
                        "isin": row.get(
                            "isin"
                        ),
                        "yahoo_symbol": symbol,
                        "price_date": str(
                            price_date.date()
                        ),
                        "window": horizon,
                        "stored": (
                            None
                            if pd.isna(stored)
                            else float(stored)
                        ),
                        "recomputed": float(
                            recomputed_value
                        ),
                        "expected": expected,
                        "target_date": target_date,
                        "difference": (
                            float(
                                recomputed_value
                            )
                            - expected
                        ),
                        "reason": (
                            "production_result_mismatch"
                        ),
                    }
                )

                continue

            categories[
                "verified_match"
            ] += 1

            by_horizon[
                horizon
            ]["verified_match"] += 1

    # --------------------------------------------------------------
    # RESULTAT
    # --------------------------------------------------------------

    print()
    print(
        "CLASSIFICATION"
    )
    print("-" * 72)

    for category, value in (
        categories.items()
    ):
        print(
            f"{category}: {value:,}"
        )

    print()
    print(
        "BY HORIZON"
    )
    print("-" * 72)

    for horizon in FORWARD_WINDOWS:
        print(
            f"{horizon}d:"
        )

        for category, value in (
            by_horizon[
                horizon
            ].items()
        ):
            print(
                f"  {category}: {value:,}"
            )

    print()
    print(
        "FAILURE EXAMPLES"
    )
    print("-" * 72)

    if not failures:
        print(
            "NONE"
        )
    else:
        for failure in failures:
            print(
                f"  {failure}"
            )

    print()
    print(
        "EXPECTED INTERPRETATION"
    )
    print("-" * 72)

    print(
        "verified_match:"
        " target finns och production-resultatet "
        "matchar prisdata."
    )

    print(
        "not_yet_verifiable:"
        " targetdagen ligger efter tillgänglig "
        "prisdata och är därför normalt."
    )

    print(
        "missing_forward_return:"
        " target finns men add_forward_returns() "
        "producerar NaN."
    )

    print(
        "value_mismatch:"
        " add_forward_returns() producerar fel värde."
    )

    print()
    print("=" * 72)
    print(
        "STANDALONE TEST RESULT"
    )
    print("=" * 72)

    actual_failures = (
        categories[
            "missing_forward_return"
        ]
        + categories[
            "value_mismatch"
        ]
        + categories[
            "price_series_not_found"
        ]
        + categories[
            "invalid_price"
        ]
    )

    print(
        f"Actual failures: {actual_failures:,}"
    )

    assert actual_failures == 0, (
        "Forward-return alignment har "
        f"{actual_failures:,} faktiska fel. "
        "Se FAILURE EXAMPLES ovan."
    )

    print(
        "PASS: alla verifierbara "
        "forward returns är korrekta."
    )
