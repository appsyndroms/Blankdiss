"""
Diagnostik för FI → feature → price → forward return → web.
Filen ändrar ingen data.
Den kontrollerar den kanoniska kedjan:
    FI aggregate
        ↓
    feature_fi.py
        ↓
    features_*.jsonl
        ↓
    feature_prices.py
        ↓
    feature_returns.py
        ↓
    data_loader.py
        ↓
    events.py
        ↓
    Köpläge
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.feature_config import (
    FEATURE_GLOB,
    PRICE_DIR,
    RETURN_HORIZONS,
)
from analysis.feature_prices import (
    find_price_files,
    load_prices,
)
from analysis.feature_returns import (
    add_forward_returns,
)
from web.build.data_loader import (
    _event_identity_keys,
    _feature_identity_keys,
    _read_feature_returns,
)
ROOT = Path(
    __file__
).resolve().parents[1]
FEATURE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "analysis"
)
EVENT_DIR = (
    ROOT
    / "data"
    / "events"
)
FI_AGGREGATE_DIR = (
    ROOT
    / "data"
    / "raw"
    / "fi"
    / "aggregate"
    / "snapshots"
)
MAX_EXAMPLES = 30
TOLERANCE = 1e-9
def _load_features() -> pd.DataFrame:
    paths = sorted(
        FEATURE_DIR.glob(
            FEATURE_GLOB
        )
    )
    if not paths:
        raise FileNotFoundError(
            "Inga feature-chunks hittades i "
            f"{FEATURE_DIR}"
        )
    frames = []
    for path in paths:
        frames.append(
            pd.read_json(
                path,
                lines=True,
            )
        )
    return pd.concat(
        frames,
        ignore_index=True,
    )
def _load_events() -> list[dict]:
    records: list[dict] = []
    for path in sorted(
        EVENT_DIR.glob(
            "short_events_*.jsonl"
        )
    ):
        with path.open(
            encoding="utf-8"
        ) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(
                        json.loads(line)
                    )
    return records
def _load_raw_aggregate() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(
        FI_AGGREGATE_DIR.glob(
            "*.jsonl"
        )
    ):
        try:
            frame = pd.read_json(
                path,
                lines=True,
            )
        except (
            OSError,
            ValueError,
            TypeError,
        ):
            continue
        if frame.empty:
            continue
        required = {
            "position_date",
            "issuer",
        }
        if not required.issubset(
            frame.columns
        ):
            continue
        frames.append(
            frame
        )
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(
        frames,
        ignore_index=True,
    )
    frame["position_date"] = pd.to_datetime(
        frame["position_date"],
        errors="coerce",
    )
    frame["issuer"] = (
        frame["issuer"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    if "lei" not in frame.columns:
        frame["lei"] = None
    return frame.loc[
        frame["position_date"].notna()
    ].copy()
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
def _feature_key(
    date_value,
    field: str,
    value: str,
) -> tuple[str, str, str]:
    return (
        str(date_value)[:10],
        field,
        str(value),
    )
def _build_feature_index(
    features: pd.DataFrame,
) -> dict[
    tuple[str, str, str],
    int,
]:
    index: dict[
        tuple[str, str, str],
        int,
    ] = {}
    for row_index, row in features.iterrows():
        feature_date = str(
            row.get(
                "snapshot_date"
            )
            or ""
        )[:10]
        if not feature_date:
            continue
        feature = row.to_dict()
        for field, value in _feature_identity_keys(
            feature
        ):
            index[
                _feature_key(
                    feature_date,
                    field,
                    value,
                )
            ] = row_index
    return index
def _check_raw_fi_to_features(
    raw_fi: pd.DataFrame,
    features: pd.DataFrame,
) -> dict:
    """
    Kontrollerar:
        raw position_date
            ==
        feature snapshot_date
    """
    result = {
        "raw_rows": int(
            len(raw_fi)
        ),
        "matched": 0,
        "missing": 0,
        "examples": [],
    }
    feature_index = _build_feature_index(
        features
    )
    for _, row in raw_fi.iterrows():
        feature_date = (
            row["position_date"]
            .strftime("%Y-%m-%d")
        )
        identities = []
        if pd.notna(
            row.get("lei")
        ):
            identities.append(
                (
                    "lei",
                    str(
                        row["lei"]
                    ).strip(),
                )
            )
        identities.append(
            (
                "issuer",
                str(
                    row["issuer"]
                ).strip(),
            )
        )
        found = False
        for field, value in identities:
            if not value:
                continue
            key = _feature_key(
                feature_date,
                field,
                value,
            )
            if key in feature_index:
                found = True
                break
        if found:
            result["matched"] += 1
        else:
            result["missing"] += 1
            if len(
                result["examples"]
            ) < MAX_EXAMPLES:
                result["examples"].append(
                    {
                        "position_date": (
                            feature_date
                        ),
                        "lei": row.get(
                            "lei"
                        ),
                        "issuer": row.get(
                            "issuer"
                        ),
                    }
                )
    return result
def _check_feature_price_alignment(
    features: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict:
    """
    Kontrollerar att price_date är den första
    tillgängliga handelsdagen på eller efter
    feature snapshot_date.
    """
    lookup = {
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
    result = {
        "matched_rows": 0,
        "verified": 0,
        "missing": 0,
        "mismatch": 0,
        "examples": [],
    }
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
    ]
    result["matched_rows"] = int(
        len(matched)
    )
    for index, row in matched.iterrows():
        symbol = str(
            row["yahoo_symbol"]
        ).strip()
        series = lookup.get(
            symbol
        )
        if series is None:
            result["missing"] += 1
            continue
        feature_date = pd.to_datetime(
            row["snapshot_date"],
            errors="coerce",
        )
        stored_price_date = pd.to_datetime(
            row["price_date"],
            errors="coerce",
        )
        if (
            pd.isna(feature_date)
            or pd.isna(stored_price_date)
        ):
            result["missing"] += 1
            continue
        dates = series[
            "date"
        ].to_numpy(
            dtype="datetime64[ns]"
        )
        entry_idx = int(
            np.searchsorted(
                dates,
                feature_date.to_datetime64(),
                side="left",
            )
        )
        if entry_idx >= len(
            series
        ):
            result["missing"] += 1
            continue
        expected_date = pd.Timestamp(
            series.iloc[
                entry_idx
            ]["date"]
        )
        if expected_date.normalize() == (
            stored_price_date.normalize()
        ):
            result["verified"] += 1
        else:
            result["mismatch"] += 1
            if len(
                result["examples"]
            ) < MAX_EXAMPLES:
                result["examples"].append(
                    {
                        "feature_index": index,
                        "issuer": row.get(
                            "issuer"
                        ),
                        "snapshot_date": str(
                            feature_date.date()
                        ),
                        "stored_price_date": str(
                            stored_price_date.date()
                        ),
                        "expected_price_date": str(
                            expected_date.date()
                        ),
                        "yahoo_symbol": symbol,
                    }
                )
    return result
def _check_forward_returns(
    features: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict:
    """
    Kör production-funktionen add_forward_returns()
    och verifierar den mot samma konkreta prisserier.
    """
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
    recomputed = add_forward_returns(
        matched,
        prices,
    )
    lookup = {
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
    result = {
        "verified": 0,
        "not_yet_verifiable": 0,
        "missing": 0,
        "mismatch": 0,
        "examples": [],
    }
    for index, row in matched.iterrows():
        symbol = str(
            row["yahoo_symbol"]
        ).strip()
        series = lookup.get(
            symbol
        )
        if series is None:
            result["missing"] += 1
            continue
        price_date = pd.to_datetime(
            row["price_date"],
            errors="coerce",
        )
        if pd.isna(price_date):
            result["missing"] += 1
            continue
        dates = series[
            "date"
        ].to_numpy(
            dtype="datetime64[ns]"
        )
        entry_idx = int(
            np.searchsorted(
                dates,
                price_date.to_datetime64(),
                side="left",
            )
        )
        if entry_idx >= len(
            series
        ):
            result["missing"] += 1
            continue
        entry_price = float(
            series.iloc[
                entry_idx
            ]["close"]
        )
        if (
            not np.isfinite(entry_price)
            or entry_price <= 0
        ):
            result["missing"] += 1
            continue
        for horizon in RETURN_HORIZONS:
            target_idx = (
                entry_idx + horizon
            )
            column = (
                f"forward_return_{horizon}d"
            )
            stored = row.get(
                column
            )
            computed = recomputed.loc[
                index,
                column,
            ]
            if target_idx >= len(
                series
            ):
                result[
                    "not_yet_verifiable"
                ] += 1
                continue
            target_price = float(
                series.iloc[
                    target_idx
                ]["close"]
            )
            if (
                not np.isfinite(
                    target_price
                )
                or target_price <= 0
            ):
                result["missing"] += 1
                continue
            expected = (
                target_price
                / entry_price
                - 1.0
            )
            if (
                pd.isna(computed)
                or computed is None
            ):
                result["missing"] += 1
                if len(
                    result["examples"]
                ) < MAX_EXAMPLES:
                    result["examples"].append(
                        {
                            "feature_index": index,
                            "issuer": row.get(
                                "issuer"
                            ),
                            "yahoo_symbol": symbol,
                            "price_date": str(
                                price_date.date()
                            ),
                            "horizon": horizon,
                            "stored": stored,
                            "expected": expected,
                            "reason": (
                                "production_return_missing"
                            ),
                        }
                    )
                continue
            if not _same(
                computed,
                expected,
            ):
                result["mismatch"] += 1
                if len(
                    result["examples"]
                ) < MAX_EXAMPLES:
                    result["examples"].append(
                        {
                            "feature_index": index,
                            "issuer": row.get(
                                "issuer"
                            ),
                            "yahoo_symbol": symbol,
                            "price_date": str(
                                price_date.date()
                            ),
                            "horizon": horizon,
                            "stored": stored,
                            "computed": computed,
                            "expected": expected,
                        }
                    )
                continue
            result["verified"] += 1
    return result
def _check_events_to_features(
    events: list[dict],
    features: pd.DataFrame,
) -> dict:
    """
    Kontrollerar:
        event_date
            ==
        feature snapshot_date
    samt att eventets identity faktiskt kan hitta
    feature-raden.
    """
    feature_index = _build_feature_index(
        features
    )
    result = {
        "events": int(
            len(events)
        ),
        "matched": 0,
        "missing": 0,
        "examples": [],
    }
    for event in events:
        event_date = str(
            event.get(
                "event_date"
            )
            or ""
        )[:10]
        found = False
        for field, value in _event_identity_keys(
            event
        ):
            key = _feature_key(
                event_date,
                field,
                value,
            )
            if key in feature_index:
                found = True
                break
        if found:
            result["matched"] += 1
        else:
            result["missing"] += 1
            if len(
                result["examples"]
            ) < MAX_EXAMPLES:
                result["examples"].append(
                    {
                        "event_date": event_date,
                        "isin": event.get(
                            "isin"
                        ),
                        "lei": event.get(
                            "lei"
                        ),
                        "issuer": event.get(
                            "issuer"
                        ),
                        "yahoo_symbol": event.get(
                            "yahoo_symbol"
                        ),
                    }
                )
    return result
def _check_web_enrichment(
    events: list[dict],
) -> dict:
    """
    Kör samma canonical enrichment som webbens
    data_loader använder.
    """
    enriched = []
    feature_returns = _read_feature_returns(
        events
    )
    for event in events:
        event_date = str(
            event.get(
                "event_date"
            )
            or ""
        )[:10]
        matched = None
        for field, value in _event_identity_keys(
            event
        ):
            matched = feature_returns.get(
                (
                    event_date,
                    field,
                    value,
                )
            )
            if matched is not None:
                break
        enriched.append(
            (
                event,
                matched,
            )
        )
    result = {
        "events": int(
            len(events)
        ),
        "matched": 0,
        "missing": 0,
        "return_fields_available": 0,
        "examples": [],
    }
    for event, matched in enriched:
        if matched is None:
            result["missing"] += 1
            if len(
                result["examples"]
            ) < MAX_EXAMPLES:
                result["examples"].append(
                    {
                        "event_date": event.get(
                            "event_date"
                        ),
                        "issuer": event.get(
                            "issuer"
                        ),
                        "isin": event.get(
                            "isin"
                        ),
                        "lei": event.get(
                            "lei"
                        ),
                    }
                )
            continue
        result["matched"] += 1
        available = sum(
            1
            for field in (
                "forward_return_1d",
                "forward_return_5d",
                "forward_return_20d",
                "forward_return_60d",
            )
            if matched.get(
                field
            ) is not None
        )
        result[
            "return_fields_available"
        ] += available
    return result
def main() -> None:
    print()
    print("=" * 78)
    print(
        "BLANKDISS FI → FEATURE → PRICE → RETURN → WEB DIAGNOSTIC"
    )
    print("=" * 78)
    features = _load_features()
    events = _load_events()
    raw_fi = _load_raw_aggregate()
    price_files = find_price_files(
        PRICE_DIR
    )
    prices = load_prices(
        price_files
    )
    print()
    print(
        "DATASET"
    )
    print("-" * 78)
    print(
        f"Feature rows:       {len(features):,}"
    )
    print(
        f"Event rows:         {len(events):,}"
    )
    print(
        f"Raw FI rows:        {len(raw_fi):,}"
    )
    print(
        f"Price rows:         {len(prices):,}"
    )
    print(
        f"Price files:        {len(price_files):,}"
    )
    print()
    print(
        "FEATURE DATE"
    )
    print("-" * 78)
    if "snapshot_date" in features.columns:
        dates = pd.to_datetime(
            features["snapshot_date"],
            errors="coerce",
        ).dropna()
        if not dates.empty:
            print(
                "Feature date range: "
                f"{dates.min().date()} → "
                f"{dates.max().date()}"
            )
    raw_result = _check_raw_fi_to_features(
        raw_fi,
        features,
    )
    print()
    print(
        "1. RAW FI → FEATURE"
    )
    print("-" * 78)
    print(
        f"Raw FI rows: {raw_result['raw_rows']:,}"
    )
    print(
        f"Matched:     {raw_result['matched']:,}"
    )
    print(
        f"Missing:     {raw_result['missing']:,}"
    )
    if raw_result["examples"]:
        print(
            "Examples:"
        )
        for example in raw_result[
            "examples"
        ]:
            print(
                f"  {example}"
            )
    price_result = _check_feature_price_alignment(
        features,
        prices,
    )
    print()
    print(
        "2. FEATURE → PRICE"
    )
    print("-" * 78)
    print(
        f"Matched rows: {price_result['matched_rows']:,}"
    )
    print(
        f"Verified:     {price_result['verified']:,}"
    )
    print(
        f"Missing:      {price_result['missing']:,}"
    )
    print(
        f"Mismatch:     {price_result['mismatch']:,}"
    )
    if price_result["examples"]:
        print(
            "Examples:"
        )
        for example in price_result[
            "examples"
        ]:
            print(
                f"  {example}"
            )
    return_result = _check_forward_returns(
        features,
        prices,
    )
    print()
    print(
        "3. PRICE → FORWARD RETURN"
    )
    print("-" * 78)
    print(
        f"Verified:             {return_result['verified']:,}"
    )
    print(
        f"Not yet verifiable:   {return_result['not_yet_verifiable']:,}"
    )
    print(
        f"Missing:              {return_result['missing']:,}"
    )
    print(
        f"Mismatch:             {return_result['mismatch']:,}"
    )
    if return_result["examples"]:
        print(
            "Examples:"
        )
        for example in return_result[
            "examples"
        ]:
            print(
                f"  {example}"
            )
    event_result = _check_events_to_features(
        events,
        features,
    )
    print()
    print(
        "4. EVENT → FEATURE"
    )
    print("-" * 78)
    print(
        f"Events:       {event_result['events']:,}"
    )
    print(
        f"Matched:      {event_result['matched']:,}"
    )
    print(
        f"Missing:      {event_result['missing']:,}"
    )
    if event_result["examples"]:
        print(
            "Examples:"
        )
        for example in event_result[
            "examples"
        ]:
            print(
                f"  {example}"
            )
    web_result = _check_web_enrichment(
        events
    )
    print()
    print(
        "5. FEATURE → WEB / KÖPLÄGE"
    )
    print("-" * 78)
    print(
        f"Events:                    {web_result['events']:,}"
    )
    print(
        f"Feature matched:           {web_result['matched']:,}"
    )
    print(
        f"Feature missing:           {web_result['missing']:,}"
    )
    print(
        "Return fields available:   "
        f"{web_result['return_fields_available']:,}"
    )
    if web_result["examples"]:
        print(
            "Examples:"
        )
        for example in web_result[
            "examples"
        ]:
            print(
                f"  {example}"
            )
    actual_failures = (
        raw_result["missing"]
        + price_result["missing"]
        + price_result["mismatch"]
        + return_result["missing"]
        + return_result["mismatch"]
        + event_result["missing"]
        + web_result["missing"]
    )
    print()
    print("=" * 78)
    print(
        "RESULT"
    )
    print("=" * 78)
    print(
        f"Actual failures: {actual_failures:,}"
    )
    if actual_failures:
        print(
            "FAIL"
        )
        raise SystemExit(1)
    print(
        "PASS"
    )
if __name__ == "__main__":
    main()
