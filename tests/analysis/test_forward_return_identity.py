from __future__ import annotations
import math
import pandas as pd
from analysis.feature_config import (
    OUTPUT_DIR,
    PRICE_DIR,
    RETURN_HORIZONS,
    FEATURE_GLOB,
)
from analysis.feature_prices import (
    find_price_files,
    load_prices,
)
from analysis.feature_returns import (
    add_forward_returns,
)
TOLERANCE = 1e-9
MAX_EXAMPLES = 20
def _load_features() -> pd.DataFrame:
    paths = sorted(
        OUTPUT_DIR.glob(
            FEATURE_GLOB
        )
    )
    if not paths:
        raise FileNotFoundError(
            "Inga feature-chunks hittades i "
            f"{OUTPUT_DIR}"
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
def _calculate_yahoo_forward_returns(
    frame: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Oberoende beräkning av forward returns där
    yahoo_symbol används som identitet.
    Detta ska INTE använda feature_returns.py:s
    implementation, eftersom det är just identiteten
    vi vill jämföra.
    """
    result = frame.copy()
    price_groups = {
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
    for horizon in RETURN_HORIZONS:
        result[
            f"_yahoo_forward_return_{horizon}d"
        ] = float("nan")
    for index, row in result.iterrows():
        symbol = row.get(
            "yahoo_symbol"
        )
        if pd.isna(symbol):
            continue
        symbol = str(
            symbol
        ).strip()
        if not symbol:
            continue
        series = price_groups.get(
            symbol
        )
        if series is None or series.empty:
            continue
        price_date = pd.to_datetime(
            row.get(
                "price_date"
            ),
            errors="coerce",
        )
        if pd.isna(price_date):
            continue
        dates = series[
            "date"
        ].to_numpy(
            dtype="datetime64[ns]"
        )
        price_date_np = (
            price_date.to_datetime64()
        )
        entry_idx = int(
            dates.searchsorted(
                price_date_np,
                side="left",
            )
        )
        if entry_idx >= len(series):
            continue
        entry_price = float(
            series.iloc[
                entry_idx
            ]["close"]
        )
        if (
            not math.isfinite(
                entry_price
            )
            or entry_price <= 0
        ):
            continue
        for horizon in RETURN_HORIZONS:
            target_idx = (
                entry_idx + horizon
            )
            if target_idx >= len(series):
                continue
            target_price = float(
                series.iloc[
                    target_idx
                ]["close"]
            )
            if (
                not math.isfinite(
                    target_price
                )
                or target_price <= 0
            ):
                continue
            result.at[
                index,
                f"_yahoo_forward_return_{horizon}d",
            ] = (
                target_price
                / entry_price
                - 1.0
            )
    return result
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
def test_forward_return_identity_diagnostic():
    """
    Diagnostiskt test för att avgöra om forward returns
    är konsekventa mellan security_key och yahoo_symbol.
    Produktionslogiken använder security_key.
    QC använder i nuläget yahoo_symbol.
    Testet ändrar ingen produktionskod och committar inget.
    """
    features = _load_features()
    price_files = find_price_files(
        PRICE_DIR
    )
    prices = load_prices(
        price_files
    )
    matched = features.loc[
        features[
            "price_match_available"
        ].fillna(False)
        & features[
            "security_key"
        ].notna()
        & features[
            "yahoo_symbol"
        ].notna()
        & features[
            "price_date"
        ].notna()
    ].copy()
    if matched.empty:
        raise AssertionError(
            "Inga matchade feature-rader "
            "kunde analyseras."
        )
    # ---------------------------------------------------------
    # 1. SECURITY_KEY
    #
    # Använd exakt samma produktionsfunktion som bygger
    # forward returns.
    # ---------------------------------------------------------
    security_key_result = add_forward_returns(
        matched[
            [
                "security_key",
                "price_date",
            ]
        ].copy(),
        prices,
    )
    # ---------------------------------------------------------
    # 2. YAHOO_SYMBOL
    #
    # Oberoende implementation med yahoo_symbol som identitet.
    # ---------------------------------------------------------
    yahoo_result = (
        _calculate_yahoo_forward_returns(
            matched,
            prices,
        )
    )
    classification_counts = {
        "matches_both": 0,
        "matches_security_key_only": 0,
        "matches_yahoo_only": 0,
        "matches_neither": 0,
        "not_comparable": 0,
    }
    examples = []
    yahoo_mismatch_rows = 0
    security_mismatch_rows = 0
    for position, (_, row) in enumerate(
        matched.iterrows()
    ):
        security_row = (
            security_key_result.iloc[
                position
            ]
        )
        yahoo_row = (
            yahoo_result.iloc[
                position
            ]
        )
        security_matches = True
        yahoo_matches = True
        horizon_details = {}
        for horizon in RETURN_HORIZONS:
            column = (
                f"forward_return_{horizon}d"
            )
            stored = row.get(
                column
            )
            security_value = (
                security_row.get(
                    column
                )
            )
            yahoo_value = (
                yahoo_row.get(
                    f"_yahoo_forward_return_{horizon}d"
                )
            )
            if not _same(
                stored,
                security_value,
            ):
                security_matches = False
            if not _same(
                stored,
                yahoo_value,
            ):
                yahoo_matches = False
            horizon_details[horizon] = {
                "stored": (
                    None
                    if pd.isna(stored)
                    else float(stored)
                ),
                "security_key": (
                    None
                    if pd.isna(
                        security_value
                    )
                    else float(
                        security_value
                    )
                ),
                "yahoo_symbol": (
                    None
                    if pd.isna(
                        yahoo_value
                    )
                    else float(
                        yahoo_value
                    )
                ),
            }
        if (
            security_matches
            and yahoo_matches
        ):
            classification = (
                "matches_both"
            )
        elif security_matches:
            classification = (
                "matches_security_key_only"
            )
            yahoo_mismatch_rows += 1
        elif yahoo_matches:
            classification = (
                "matches_yahoo_only"
            )
            security_mismatch_rows += 1
        elif (
            any(
                value["security_key"]
                is not None
                for value in horizon_details.values()
            )
            or any(
                value["yahoo_symbol"]
                is not None
                for value in horizon_details.values()
            )
        ):
            classification = (
                "matches_neither"
            )
            security_mismatch_rows += 1
            yahoo_mismatch_rows += 1
        else:
            classification = (
                "not_comparable"
            )
        classification_counts[
            classification
        ] += 1
        if (
            classification
            != "matches_both"
            and len(examples)
            < MAX_EXAMPLES
        ):
            examples.append(
                {
                    "security_key": (
                        row.get(
                            "security_key"
                        )
                    ),
                    "yahoo_symbol": (
                        row.get(
                            "yahoo_symbol"
                        )
                    ),
                    "isin": row.get(
                        "isin"
                    ),
                    "issuer": row.get(
                        "issuer"
                    ),
                    "price_date": (
                        str(
                            row.get(
                                "price_date"
                            )
                        )
                    ),
                    "price_mapping_source": (
                        row.get(
                            "price_mapping_source"
                        )
                    ),
                    "classification": (
                        classification
                    ),
                    "horizons": (
                        horizon_details
                    ),
                }
            )
    # ---------------------------------------------------------
    # 3. Kontrollera identitetskollisioner i prisdata
    #
    # Detta är den centrala kontrollen:
    # samma Yahoo-symbol + datum ska kunna tillhöra
    # flera security_keys om Yahoo-symbolen inte är
    # en entydig instrumentidentitet.
    # ---------------------------------------------------------
    collision_counts = (
        prices
        .groupby(
            [
                "yahoo_symbol",
                "date",
            ]
        )["security_key"]
        .nunique()
    )
    collisions = collision_counts.loc[
        collision_counts > 1
    ]
    collision_examples = []
    for (
        symbol,
        date,
    ), count in collisions.head(
        MAX_EXAMPLES
    ).items():
        keys = sorted(
            prices.loc[
                (
                    prices[
                        "yahoo_symbol"
                    ]
                    == symbol
                )
                & (
                    prices["date"]
                    == date
                ),
                "security_key",
            ]
            .dropna()
            .unique()
            .tolist()
        )
        collision_examples.append(
            {
                "yahoo_symbol": symbol,
                "date": str(
                    pd.Timestamp(
                        date
                    ).date()
                ),
                "security_key_count": int(
                    count
                ),
                "security_keys": keys,
            }
        )
    # ---------------------------------------------------------
    # 4. Kontrollera symboler som förekommer på flera
    #    security_keys över tid.
    # ---------------------------------------------------------
    symbol_identity_counts = (
        prices
        .groupby(
            "yahoo_symbol"
        )["security_key"]
        .nunique()
    )
    ambiguous_symbols = (
        symbol_identity_counts.loc[
            symbol_identity_counts > 1
        ]
    )
    ambiguous_examples = []
    for symbol in ambiguous_symbols.head(
        MAX_EXAMPLES
    ).index:
        keys = sorted(
            prices.loc[
                prices[
                    "yahoo_symbol"
                ]
                == symbol,
                "security_key",
            ]
            .dropna()
            .unique()
            .tolist()
        )
        ambiguous_examples.append(
            {
                "yahoo_symbol": symbol,
                "security_keys": keys,
            }
        )
    print()
    print(
        "=========================================="
    )
    print(
        "FORWARD RETURN IDENTITY DIAGNOSTIC"
    )
    print(
        "=========================================="
    )
    print(
        f"Feature rows: "
        f"{len(features):,}"
    )
    print(
        f"Matched rows: "
        f"{len(matched):,}"
    )
    print(
        f"Price rows: "
        f"{len(prices):,}"
    )
    print()
    print(
        "Classification:"
    )
    for name, count in (
        classification_counts.items()
    ):
        print(
            f"  {name}: "
            f"{count:,}"
        )
    print()
    print(
        "Rows matching security_key "
        "but not yahoo_symbol: "
        f"{yahoo_mismatch_rows:,}"
    )
    print(
        "Rows matching yahoo_symbol "
        "but not security_key: "
        f"{security_mismatch_rows:,}"
    )
    print()
    print(
        "Yahoo symbols with >1 security_key: "
        f"{len(ambiguous_symbols):,}"
    )
    print(
        "Yahoo symbol/date collision groups: "
        f"{len(collisions):,}"
    )
    if ambiguous_examples:
        print()
        print(
            "Ambiguous Yahoo symbols:"
        )
        for example in ambiguous_examples:
            print(
                f"  {example}"
            )
    if collision_examples:
        print()
        print(
            "Yahoo symbol/date collisions:"
        )
        for example in collision_examples:
            print(
                f"  {example}"
            )
    if examples:
        print()
        print(
            "Forward-return examples:"
        )
        for example in examples:
            print(
                f"  {example}"
            )
    print(
        "=========================================="
    )
    # ---------------------------------------------------------
    # 5. Själva diagnostiska assertionen.
    #
    # Vi vill INTE kräva att yahoo_symbol fungerar.
    # Vi vill få ett tydligt FAIL om det finns rader där
    # produktionsresultatet följer security_key men QC:s
    # yahoo-baserade serie ger ett annat resultat.
    # ---------------------------------------------------------
    assert (
        classification_counts[
            "matches_security_key_only"
        ]
        == 0
    ), (
        "Forward returns matchar "
        "security_key men inte yahoo_symbol. "
        "Det visar att yahoo_symbol och security_key "
        "inte representerar samma prisserie för dessa "
        "observationer. Se diagnostiken ovan."
    )
