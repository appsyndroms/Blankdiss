from __future__ import annotations
import json
import math
from collections import Counter
from pathlib import Path
import pandas as pd
import pytest
ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = (
    ROOT
    / "data"
    / "processed"
    / "analysis"
)
PRICES_DIR = (
    ROOT
    / "data"
    / "raw"
    / "prices"
)
HORIZONS = (
    1,
    5,
    20,
    60,
)
TOLERANCE = 1e-9
MAX_EXAMPLES = 20
def normalise_text(
    value,
) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    value = str(value).strip()
    if not value:
        return None
    return value
def load_jsonl_files(
    directory: Path,
    pattern: str,
) -> pd.DataFrame:
    paths = sorted(
        directory.glob(pattern)
    )
    if not paths:
        pytest.skip(
            f"Inga filer hittades: "
            f"{directory}/{pattern}"
        )
    frames = []
    for path in paths:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            rows = [
                json.loads(line)
                for line in handle
                if line.strip()
            ]
        if rows:
            frames.append(
                pd.DataFrame(rows)
            )
    if not frames:
        pytest.skip(
            f"Alla filer är tomma: "
            f"{directory}/{pattern}"
        )
    return pd.concat(
        frames,
        ignore_index=True,
    )
def load_features() -> pd.DataFrame:
    frame = load_jsonl_files(
        FEATURES_DIR,
        "features_*.jsonl",
    )
    required = {
        "security_key",
        "yahoo_symbol",
        "price_date",
    }
    required.update(
        f"forward_return_{horizon}d"
        for horizon in HORIZONS
    )
    missing = sorted(
        required - set(frame.columns)
    )
    assert not missing, (
        "Feature-data saknar kolumner: "
        + ", ".join(missing)
    )
    frame["price_date"] = pd.to_datetime(
        frame["price_date"],
        errors="coerce",
    )
    frame["security_key"] = (
        frame["security_key"]
        .map(normalise_text)
    )
    frame["yahoo_symbol"] = (
        frame["yahoo_symbol"]
        .map(normalise_text)
    )
    return frame
def load_prices() -> pd.DataFrame:
    frame = load_jsonl_files(
        PRICES_DIR,
        "prices_*.jsonl",
    )
    required = {
        "date",
        "yahoo_symbol",
        "close",
    }
    missing = sorted(
        required - set(frame.columns)
    )
    assert not missing, (
        "Pris-data saknar kolumner: "
        + ", ".join(missing)
    )
    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )
    frame["close"] = pd.to_numeric(
        frame["close"],
        errors="coerce",
    )
    frame["yahoo_symbol"] = (
        frame["yahoo_symbol"]
        .map(normalise_text)
    )
    return frame.loc[
        frame["date"].notna()
        & frame["yahoo_symbol"].notna()
        & frame["close"].notna()
    ].copy()
def calculate_forward_returns(
    prices: pd.DataFrame,
    identity_column: str,
) -> dict[tuple[str, pd.Timestamp], dict[int, float | None]]:
    """
    Beräknar forward returns för en prisserie grupperad på
    den angivna identiteten.
    identity_column är antingen:
        yahoo_symbol
    eller:
        security_key
    """
    if identity_column not in prices.columns:
        return {}
    result = {}
    working = prices.loc[
        prices[identity_column].notna()
    ].copy()
    working = (
        working
        .sort_values(
            [
                identity_column,
                "date",
            ],
            kind="mergesort",
        )
        .drop_duplicates(
            subset=[
                identity_column,
                "date",
            ],
            keep="last",
        )
    )
    for identity, group in working.groupby(
        identity_column,
        sort=False,
    ):
        group = (
            group
            .sort_values(
                "date",
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )
        for position in range(
            len(group)
        ):
            date = group.iloc[
                position
            ]["date"]
            base_price = float(
                group.iloc[
                    position
                ]["close"]
            )
            values = {}
            for horizon in HORIZONS:
                target = (
                    position + horizon
                )
                if target >= len(group):
                    values[horizon] = None
                    continue
                future_price = float(
                    group.iloc[
                        target
                    ]["close"]
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
                    values[horizon] = None
                    continue
                values[horizon] = (
                    future_price
                    / base_price
                    - 1.0
                )
            result[
                (
                    str(identity),
                    pd.Timestamp(date),
                )
            ] = values
    return result
def compare(
    stored,
    calculated,
) -> str:
    stored_missing = (
        stored is None
        or pd.isna(stored)
    )
    calculated_missing = (
        calculated is None
        or pd.isna(calculated)
    )
    if (
        stored_missing
        and calculated_missing
    ):
        return "both_missing"
    if stored_missing:
        return "stored_missing"
    if calculated_missing:
        return "calculated_missing"
    if math.isclose(
        float(stored),
        float(calculated),
        rel_tol=TOLERANCE,
        abs_tol=TOLERANCE,
    ):
        return "match"
    return "mismatch"
def test_forward_return_identity_alignment():
    """
    Diagnostiskt test för skillnaden mellan Yahoo-symbol och
    security_key som identitet för forward returns.
    Testet ändrar ingen produktionskod och skriver inte om
    feature- eller prisdata.
    Ett eventuellt assertion-fel innehåller sammanfattningen som
    behövs för att avgöra om QC använder fel identitet.
    """
    features = load_features()
    prices = load_prices()
    yahoo_returns = calculate_forward_returns(
        prices,
        "yahoo_symbol",
    )
    has_security_key = (
        "security_key" in prices.columns
        and prices["security_key"].notna().any()
    )
    security_returns = {}
    if has_security_key:
        security_returns = calculate_forward_returns(
            prices,
            "security_key",
        )
    matched = features.loc[
        features["price_date"].notna()
        & features["yahoo_symbol"].notna()
        & features["security_key"].notna()
    ].copy()
    if matched.empty:
        pytest.skip(
            "Inga feature-rader med "
            "security_key, yahoo_symbol och price_date."
        )
    counts = Counter()
    examples = []
    yahoo_mismatch_rows = 0
    security_mismatch_rows = 0
    security_fixes_yahoo = 0
    for _, row in matched.iterrows():
        yahoo_key = (
            str(row["yahoo_symbol"]),
            pd.Timestamp(
                row["price_date"]
            ),
        )
        security_key = (
            str(row["security_key"]),
            pd.Timestamp(
                row["price_date"]
            ),
        )
        yahoo_values = yahoo_returns.get(
            yahoo_key
        )
        security_values = (
            security_returns.get(
                security_key
            )
            if has_security_key
            else None
        )
        yahoo_mismatches = 0
        security_mismatches = 0
        horizon_details = {}
        for horizon in HORIZONS:
            column = (
                f"forward_return_{horizon}d"
            )
            stored = row[column]
            yahoo_calculated = (
                yahoo_values[horizon]
                if yahoo_values is not None
                else None
            )
            security_calculated = (
                security_values[horizon]
                if security_values is not None
                else None
            )
            yahoo_status = compare(
                stored,
                yahoo_calculated,
            )
            security_status = compare(
                stored,
                security_calculated,
            )
            if yahoo_status == "mismatch":
                yahoo_mismatches += 1
            if security_status == "mismatch":
                security_mismatches += 1
            horizon_details[
                horizon
            ] = {
                "stored": (
                    float(stored)
                    if not pd.isna(stored)
                    else None
                ),
                "yahoo": yahoo_calculated,
                "security_key": security_calculated,
                "yahoo_status": yahoo_status,
                "security_status": security_status,
            }
        if yahoo_mismatches:
            yahoo_mismatch_rows += 1
        if security_mismatches:
            security_mismatch_rows += 1
        if (
            has_security_key
            and yahoo_mismatches
            and not security_mismatches
        ):
            security_fixes_yahoo += 1
            classification = (
                "MATCHES_SECURITY_KEY_ONLY"
            )
        elif (
            not yahoo_mismatches
            and not security_mismatches
        ):
            classification = (
                "MATCHES_BOTH"
            )
        elif (
            yahoo_mismatches
            and not security_mismatches
        ):
            classification = (
                "MATCHES_SECURITY_KEY_ONLY"
            )
        elif (
            not yahoo_mismatches
            and security_mismatches
        ):
            classification = (
                "MATCHES_YAHOO_ONLY"
            )
        elif (
            yahoo_mismatches
            and security_mismatches
        ):
            classification = (
                "MATCHES_NEITHER"
            )
        else:
            classification = (
                "NO_COMPARABLE_RETURN"
            )
        counts[classification] += 1
        if (
            classification
            != "MATCHES_BOTH"
            and len(examples)
            < MAX_EXAMPLES
        ):
            examples.append(
                {
                    "security_key": (
                        row["security_key"]
                    ),
                    "yahoo_symbol": (
                        row["yahoo_symbol"]
                    ),
                    "price_date": (
                        pd.Timestamp(
                            row["price_date"]
                        )
                        .date()
                        .isoformat()
                    ),
                    "price_mapping_source": (
                        row.get(
                            "price_mapping_source"
                        )
                    ),
                    "isin": row.get(
                        "isin"
                    ),
                    "issuer": row.get(
                        "issuer"
                    ),
                    "classification": (
                        classification
                    ),
                    "horizons": (
                        horizon_details
                    ),
                }
            )
    ambiguous_symbols = (
        features.loc[
            features["yahoo_symbol"].notna()
            & features["security_key"].notna()
        ]
        .groupby(
            "yahoo_symbol"
        )["security_key"]
        .nunique()
    )
    ambiguous_symbols = ambiguous_symbols.loc[
        ambiguous_symbols > 1
    ]
    collision_groups = 0
    collision_examples = []
    if "security_key" in prices.columns:
        collisions = (
            prices.loc[
                prices["security_key"].notna()
            ]
            .groupby(
                [
                    "yahoo_symbol",
                    "date",
                ]
            )["security_key"]
            .nunique()
        )
        collisions = collisions.loc[
            collisions > 1
        ]
        collision_groups = len(
            collisions
        )
        for (
            symbol,
            date,
        ), count in collisions.head(
            MAX_EXAMPLES
        ).items():
            keys = (
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
                    "date": (
                        pd.Timestamp(
                            date
                        )
                        .date()
                        .isoformat()
                    ),
                    "security_key_count": int(
                        count
                    ),
                    "security_keys": keys,
                }
            )
    diagnostic = {
        "feature_rows": len(features),
        "matched_rows": len(matched),
        "price_rows": len(prices),
        "price_contains_security_key": (
            has_security_key
        ),
        "yahoo_mismatch_rows": (
            yahoo_mismatch_rows
        ),
        "security_key_mismatch_rows": (
            security_mismatch_rows
        ),
        "security_key_fixes_yahoo_mismatch": (
            security_fixes_yahoo
        ),
        "classification": dict(
            counts
        ),
        "symbols_with_multiple_security_keys": (
            int(
                len(
                    ambiguous_symbols
                )
            )
        ),
        "raw_price_collision_groups": (
            collision_groups
        ),
        "collision_examples": (
            collision_examples
        ),
        "examples": examples,
    }
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
        json.dumps(
            diagnostic,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    print(
        "=========================================="
    )
    if not has_security_key:
        pytest.fail(
            "Prisdata saknar security_key. "
            "Diagnostiken kan därför inte direkt "
            "jämföra forward returns mot security_key. "
            "Se diagnostiken ovan."
        )
    assert security_fixes_yahoo == 0, (
        "Forward-return identity mismatch hittades: "
        f"{security_fixes_yahoo:,} rader matchar "
        "security_key men inte yahoo_symbol. "
        "Detta är exakt den situation vi behöver "
        "undersöka innan QC ändras. "
        "Se diagnostiken ovan för exempel."
    )
def test_yahoo_symbol_does_not_mix_security_keys():
    """
    Separat diagnostiskt test.
    Om samma yahoo_symbol används för flera security_keys
    är Yahoo-symbolen inte en entydig instrumentidentitet.
    """
    features = load_features()
    working = features.loc[
        features["yahoo_symbol"].notna()
        & features["security_key"].notna()
    ]
    if working.empty:
        pytest.skip(
            "Ingen Yahoo/security_key-mappning."
        )
    ambiguous = (
        working
        .groupby(
            "yahoo_symbol"
        )["security_key"]
        .nunique()
    )
    ambiguous = ambiguous.loc[
        ambiguous > 1
    ]
    examples = []
    for symbol in ambiguous.head(
        MAX_EXAMPLES
    ).index:
        keys = sorted(
            working.loc[
                working[
                    "yahoo_symbol"
                ]
                == symbol,
                "security_key",
            ]
            .unique()
            .tolist()
        )
        examples.append(
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
        "YAHOO SYMBOL / SECURITY KEY AMBIGUITY"
    )
    print(
        "=========================================="
    )
    print(
        f"Symbols med flera security_keys: "
        f"{len(ambiguous):,}"
    )
    print(
        json.dumps(
            examples,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    print(
        "=========================================="
    )
    assert not ambiguous.any(), (
        "Samma yahoo_symbol förekommer med flera "
        "security_keys. "
        f"Antal symboler: {len(ambiguous):,}. "
        "Se diagnostiken ovan."
    )
