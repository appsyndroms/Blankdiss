from __future__ import annotations
import math
import numpy as np
import pandas as pd
from analysis.feature_config import (
    FEATURE_GLOB,
    OUTPUT_DIR,
    PRICE_DIR,
    RETURN_HORIZONS,
)
from analysis.feature_prices import (
    find_price_files,
    load_prices,
)
from analysis.features_qc import (
    check_forward_return_alignment,
)
TOLERANCE = 1e-9
MAX_EXAMPLES = 50
def _load_features() -> pd.DataFrame:
    paths = sorted(
        OUTPUT_DIR.glob(FEATURE_GLOB)
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
def test_forward_return_availability_diagnostic():
    """
    Diagnostik för forward-return targets.
    Testet undersöker varför vissa redan lagrade
    forward_return_* targets inte längre kan räknas fram
    från det aktuella prisarkivet.
    Ingen produktionskod ändras.
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
            "price_date"
        ].notna()
    ].copy()
    if matched.empty:
        raise AssertionError(
            "Inga matchade feature-rader "
            "kunde analyseras."
        )
    lookup = _build_price_lookup(
        prices
    )
    classification_counts = {
        "exact_match": 0,
        "stored_only": 0,
        "computed_only": 0,
        "value_mismatch": 0,
        "both_missing": 0,
    }
    mismatch_examples = []
    stored_only_examples = []
    horizon_counts = {
        horizon: {
            "exact_match": 0,
            "stored_only": 0,
            "computed_only": 0,
            "value_mismatch": 0,
            "both_missing": 0,
        }
        for horizon in RETURN_HORIZONS
    }
    current_max_dates = []
    for _, row in matched.iterrows():
        security_key = row[
            "security_key"
        ]
        series = lookup.get(
            security_key
        )
        if (
            series is None
            or series.empty
        ):
            continue
        dates = series[
            "date"
        ].to_numpy(
            dtype="datetime64[ns]"
        )
        price_date = pd.to_datetime(
            row["price_date"],
            errors="coerce",
        )
        if pd.isna(price_date):
            continue
        entry_idx = int(
            np.searchsorted(
                dates,
                price_date.to_datetime64(),
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
        max_date = pd.Timestamp(
            series.iloc[-1]["date"]
        )
        current_max_dates.append(
            max_date
        )
        row_problem_horizons = []
        for horizon in RETURN_HORIZONS:
            column = (
                f"forward_return_{horizon}d"
            )
            stored = row.get(
                column
            )
            target_idx = (
                entry_idx + horizon
            )
            if target_idx >= len(series):
                computed = np.nan
                required_date = None
            else:
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
                    computed = np.nan
                else:
                    computed = (
                        target_price
                        / entry_price
                        - 1.0
                    )
                required_date = pd.Timestamp(
                    series.iloc[
                        target_idx
                    ]["date"]
                )
            stored_missing = (
                stored is None
                or pd.isna(stored)
            )
            computed_missing = (
                computed is None
                or pd.isna(computed)
            )
            if (
                not stored_missing
                and not computed_missing
            ):
                if _same(
                    stored,
                    computed,
                ):
                    classification = (
                        "exact_match"
                    )
                else:
                    classification = (
                        "value_mismatch"
                    )
            elif (
                not stored_missing
                and computed_missing
            ):
                classification = (
                    "stored_only"
                )
                row_problem_horizons.append(
                    {
                        "horizon": horizon,
                        "stored": float(stored),
                        "entry_index": entry_idx,
                        "target_index": target_idx,
                        "current_rows": len(series),
                        "current_last_date": str(
                            max_date.date()
                        ),
                        "required_date": (
                            None
                            if required_date is None
                            else str(
                                required_date.date()
                            )
                        ),
                    }
                )
            elif (
                stored_missing
                and not computed_missing
            ):
                classification = (
                    "computed_only"
                )
            else:
                classification = (
                    "both_missing"
                )
            classification_counts[
                classification
            ] += 1
            horizon_counts[
                horizon
            ][
                classification
            ] += 1
            if classification == "value_mismatch":
                if (
                    len(mismatch_examples)
                    < MAX_EXAMPLES
                ):
                    mismatch_examples.append(
                        {
                            "security_key": security_key,
                            "yahoo_symbol": row.get(
                                "yahoo_symbol"
                            ),
                            "isin": row.get(
                                "isin"
                            ),
                            "issuer": row.get(
                                "issuer"
                            ),
                            "price_date": str(
                                pd.Timestamp(
                                    price_date
                                ).date()
                            ),
                            "horizon": horizon,
                            "stored": float(stored),
                            "computed": float(computed),
                            "difference": float(
                                computed - stored
                            ),
                            "current_last_date": str(
                                max_date.date()
                            ),
                        }
                    )
        if row_problem_horizons:
            if (
                len(stored_only_examples)
                < MAX_EXAMPLES
            ):
                stored_only_examples.append(
                    {
                        "security_key": security_key,
                        "yahoo_symbol": row.get(
                            "yahoo_symbol"
                        ),
                        "isin": row.get(
                            "isin"
                        ),
                        "issuer": row.get(
                            "issuer"
                        ),
                        "price_date": str(
                            pd.Timestamp(
                                price_date
                            ).date()
                        ),
                        "current_last_date": str(
                            max_date.date()
                        ),
                        "current_observations": len(
                            series
                        ),
                        "missing_targets": (
                            row_problem_horizons
                        ),
                    }
                )
    print()
    print(
        "============================================================"
    )
    print(
        "FORWARD RETURN AVAILABILITY DIAGNOSTIC"
    )
    print(
        "============================================================"
    )
    print(
        f"Feature rows: {len(features):,}"
    )
    print(
        f"Matched rows: {len(matched):,}"
    )
    print(
        f"Price rows: {len(prices):,}"
    )
    if current_max_dates:
        print(
            "Current maximum price date: "
            f"{max(current_max_dates).date()}"
        )
    print()
    print(
        "OVERALL CLASSIFICATION"
    )
    for name, count in (
        classification_counts.items()
    ):
        print(
            f"  {name}: {count:,}"
        )
    print()
    print(
        "PER HORIZON"
    )
    for horizon in RETURN_HORIZONS:
        print(
            f"  {horizon}d:"
        )
        for name, count in (
            horizon_counts[
                horizon
            ].items()
        ):
            print(
                f"    {name}: {count:,}"
            )
    print()
    if stored_only_examples:
        print(
            "STORED TARGETS THAT CURRENT PRICE DATA "
            "CANNOT RECONSTRUCT"
        )
        for example in stored_only_examples:
            print(
                f"  {example}"
            )
    print()
    if mismatch_examples:
        print(
            "ACTUAL NUMERICAL MISMATCHES"
        )
        for example in mismatch_examples:
            print(
                f"  {example}"
            )
    else:
        print(
            "ACTUAL NUMERICAL MISMATCHES: 0"
        )
    print(
        "============================================================"
    )
    # Detta är den viktiga kontrollen.
    #
    # Om båda värdena finns måste de vara numeriskt identiska.
    # stored_only är däremot inte ett fel i forward-return
    # beräkningen; det betyder att dagens prisarkiv inte längre
    # innehåller tillräckligt många framtida observationer.
    assert (
        classification_counts[
            "value_mismatch"
        ]
        == 0
    ), (
        "Det finns faktiska numeriska skillnader mellan "
        "lagrade forward returns och aktuell prisdata. "
        "Se diagnostiken ovan."
    )
def test_forward_return_alignment_qc_distinguishes_unverifiable_targets():
    """
    Kontrollerar QC-reglerna för forward-return alignment.
    Tre fall testas:
    1. Target kan verifieras och stämmer:
       -> verified_match
    2. Target ligger efter aktuell prisseries slut:
       -> not_yet_verifiable
       -> ska INTE ge FAIL
    3. Target kan verifieras men saknas:
       -> missing_forward_return
       -> ska ge FAIL
    """
    prices = pd.DataFrame(
        [
            {
                "yahoo_symbol": "TEST.ST",
                "date": pd.Timestamp(
                    "2026-09-14"
                ),
                "close": 100.0,
            },
            {
                "yahoo_symbol": "TEST.ST",
                "date": pd.Timestamp(
                    "2026-09-15"
                ),
                "close": 101.0,
            },
            {
                "yahoo_symbol": "TEST.ST",
                "date": pd.Timestamp(
                    "2026-09-16"
                ),
                "close": 102.0,
            },
        ]
    )
    verified_frame = pd.DataFrame(
        [
            {
                "price_match_available": True,
                "yahoo_symbol": "TEST.ST",
                "price_date": pd.Timestamp(
                    "2026-09-14"
                ),
                "forward_return_1d": 0.01,
                "forward_return_5d": np.nan,
                "forward_return_20d": np.nan,
                "forward_return_60d": np.nan,
            }
        ]
    )
    result = check_forward_return_alignment(
        verified_frame,
        prices,
    )
    assert result["status"] == "PASS"
    assert result["verified_match"] == 1
    assert result["not_yet_verifiable"] == 3
    assert result["value_mismatch"] == 0
    assert result["price_date_not_found"] == 0
    assert result["missing_forward_return"] == 0
    stored_only_frame = pd.DataFrame(
        [
            {
                "price_match_available": True,
                "yahoo_symbol": "TEST.ST",
                "price_date": pd.Timestamp(
                    "2026-09-14"
                ),
                "forward_return_1d": 0.01,
                "forward_return_5d": 0.10,
                "forward_return_20d": np.nan,
                "forward_return_60d": np.nan,
            }
        ]
    )
    result = check_forward_return_alignment(
        stored_only_frame,
        prices,
    )
    assert result["status"] == "PASS"
    assert result["verified_match"] == 1
    assert result["not_yet_verifiable"] == 3
    assert result["value_mismatch"] == 0
    assert result["price_date_not_found"] == 0
    assert result["missing_forward_return"] == 0
    missing_target_frame = pd.DataFrame(
        [
            {
                "price_match_available": True,
                "yahoo_symbol": "TEST.ST",
                "price_date": pd.Timestamp(
                    "2026-09-14"
                ),
                "forward_return_1d": np.nan,
                "forward_return_5d": np.nan,
                "forward_return_20d": np.nan,
                "forward_return_60d": np.nan,
            }
        ]
    )
    result = check_forward_return_alignment(
        missing_target_frame,
        prices,
    )
    assert result["status"] == "FAIL"
    assert result["verified_match"] == 0
    assert result["not_yet_verifiable"] == 3
    assert result["value_mismatch"] == 0
    assert result["price_date_not_found"] == 0
    assert result["missing_forward_return"] == 1
