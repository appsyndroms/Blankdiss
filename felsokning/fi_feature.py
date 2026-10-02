"""Diagnostik för FI -> feature -> price-kedjan."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

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

FEATURE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "analysis"
)

PRICE_DIR = (
    ROOT
    / "data"
    / "raw"
    / "prices"
)

EVENT_DIR = (
    ROOT
    / "data"
    / "events"
)


TARGETS = {
    "Fingerprint": {
        "issuer": "Fingerprint Cards AB",
        "event_date": "2023-11-24",
        "isin": "SE0008374250",
        "lei": "5493004YF5D7Z612Z822",
        "yahoo_symbol": "FING-B.ST",
    },
    "Viaplay": {
        "issuer": "Viaplay Group AB (publ)",
        "event_date": "2026-08-28",
        "isin": "SE0012116390",
        "lei": "5493006E0IJD0DHJSR89",
        "yahoo_symbol": "VPLAY-B.ST",
    },
}


def normalize(value) -> str:
    if value is None:
        return ""

    return str(value).strip().casefold()


def security_key(isin, issuer) -> str:
    normalized_isin = normalize(isin)

    if normalized_isin:
        return "ISIN:" + normalized_isin

    return "ISSUER:" + normalize(issuer)


def identity_keys(row) -> list[tuple[str, str]]:
    keys = []

    for field in (
        "isin",
        "lei",
        "yahoo_symbol",
        "issuer",
    ):
        value = normalize(row.get(field))

        if value:
            keys.append(
                (
                    field,
                    value,
                )
            )

    return keys


def matches_identity(row, target) -> bool:
    for field in (
        "isin",
        "yahoo_symbol",
        "issuer",
    ):
        target_value = normalize(
            target.get(field)
        )

        row_value = normalize(
            row.get(field)
        )

        if (
            target_value
            and row_value
            and target_value == row_value
        ):
            return True

    return False


def separator(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def load_fi() -> pd.DataFrame:
    if not FI_PATH.exists():
        raise SystemExit(
            f"FI-fil saknas: {FI_PATH}"
        )

    frame = pd.read_json(
        FI_PATH,
        lines=True,
    )

    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    frame["isin"] = frame["isin"].where(
        frame["isin"].notna(),
        None,
    )

    frame["security_key"] = [
        security_key(
            isin,
            issuer,
        )
        for isin, issuer in zip(
            frame["isin"],
            frame["issuer"],
        )
    ]

    return frame


def load_features() -> pd.DataFrame:
    files = sorted(
        FEATURE_DIR.glob(
            "features_*.jsonl"
        )
    )

    if not files:
        raise SystemExit(
            "Inga feature-chunks hittades."
        )

    rows = []

    for path in files:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            for line_number, line in enumerate(
                handle,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                row["_source_file"] = str(
                    path.relative_to(ROOT)
                )

                row["_source_line"] = line_number

                rows.append(row)

    frame = pd.DataFrame(rows)

    if frame.empty:
        raise SystemExit(
            "Feature-datasetet är tomt."
        )

    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    return frame


def load_prices() -> pd.DataFrame:
    files = sorted(
        PRICE_DIR.glob(
            "prices_*.jsonl"
        )
    )

    if not files:
        raise SystemExit(
            "Inga price-filer hittades."
        )

    rows = []

    for path in files:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            for line_number, line in enumerate(
                handle,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                row["_source_file"] = str(
                    path.relative_to(ROOT)
                )

                row["_source_line"] = line_number

                rows.append(row)

    frame = pd.DataFrame(rows)

    if frame.empty:
        return frame

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    frame["security_key"] = [
        security_key(
            isin,
            issuer,
        )
        for isin, issuer in zip(
            frame["isin"],
            frame["issuer"],
        )
    ]

    return frame


def load_events() -> dict[str, list[dict]]:
    events = {
        name: []
        for name in TARGETS
    }

    files = sorted(
        EVENT_DIR.glob(
            "short_events_*.jsonl"
        )
    )

    for path in files:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            for line_number, line in enumerate(
                handle,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                for name, target in TARGETS.items():
                    if matches_identity(
                        row,
                        target,
                    ):
                        events[name].append(
                            {
                                "path": path,
                                "line": line_number,
                                "row": row,
                            }
                        )

    return events


def print_event_diagnosis(events) -> None:
    separator("1. EVENTS")

    for name, items in events.items():
        print()
        print(f"### {name}")

        if not items:
            print("EVENT: MISSING")
            continue

        seen = set()

        for item in items:
            row = item["row"]

            identity = (
                row.get("event_date"),
                row.get("isin"),
                row.get("yahoo_symbol"),
            )

            if identity in seen:
                continue

            seen.add(identity)

            print(
                f"{item['path'].relative_to(ROOT)}:"
                f"{item['line']}"
            )

            print(
                f"  event_date   : {row.get('event_date')}"
            )

            print(
                f"  issuer       : {row.get('issuer')}"
            )

            print(
                f"  isin         : {row.get('isin')}"
            )

            print(
                f"  lei          : {row.get('lei')}"
            )

            print(
                f"  yahoo_symbol : {row.get('yahoo_symbol')}"
            )


def print_fi_diagnosis(fi: pd.DataFrame) -> dict:
    separator("2. FI -> SECURITY KEY")

    target_fi = {}

    for name, target in TARGETS.items():
        print()
        print(f"### {name}")

        target_date = pd.Timestamp(
            target["event_date"]
        )

        rows = fi.loc[
            fi["snapshot_date"]
            == target_date
        ].copy()

        rows = rows.loc[
            rows.apply(
                lambda row: matches_identity(
                    row,
                    target,
                ),
                axis=1,
            )
        ]

        target_fi[name] = rows

        if rows.empty:
            print("FI MATCH: MISSING")
            continue

        print(
            f"FI MATCH: {len(rows)} row(s)"
        )

        for index, row in rows.iterrows():
            print()
            print(
                f"  row index      : {index}"
            )

            print(
                f"  snapshot_date  : "
                f"{row.get('snapshot_date')}"
            )

            print(
                f"  issuer         : "
                f"{row.get('issuer')}"
            )

            print(
                f"  isin           : "
                f"{row.get('isin')}"
            )

            print(
                f"  security_key   : "
                f"{row.get('security_key')}"
            )

            print(
                f"  short_interest : "
                f"{row.get('short_interest_pct')}"
            )

        if len(rows) > 1:
            print()
            print(
                "WARNING: flera FI-rader matchar "
                "samma target."
            )

    return target_fi


def print_feature_exact_diagnosis(
    features: pd.DataFrame,
) -> dict:
    separator(
        "3. EXACT FI DATE -> FEATURE"
    )

    target_features = {}

    for name, target in TARGETS.items():
        print()
        print(f"### {name}")

        target_date = pd.Timestamp(
            target["event_date"]
        )

        rows = features.loc[
            features["snapshot_date"]
            == target_date
        ].copy()

        rows = rows.loc[
            rows.apply(
                lambda row: matches_identity(
                    row,
                    target,
                ),
                axis=1,
            )
        ]

        target_features[name] = rows

        if rows.empty:
            print(
                "FEATURE MATCH: MISSING"
            )
            continue

        print(
            f"FEATURE MATCH: {len(rows)} row(s)"
        )

        for _, row in rows.iterrows():
            print()
            print(
                f"  source       : "
                f"{row.get('_source_file')}:"
                f"{row.get('_source_line')}"
            )

            print(
                f"  snapshot     : "
                f"{row.get('snapshot_date')}"
            )

            print(
                f"  issuer       : "
                f"{row.get('issuer')}"
            )

            print(
                f"  isin         : "
                f"{row.get('isin')}"
            )

            print(
                f"  security_key : "
                f"{row.get('security_key')}"
            )

            print(
                f"  price_date   : "
                f"{row.get('price_date')}"
            )

            print(
                f"  price_match  : "
                f"{row.get('price_match_available')}"
            )

            print(
                f"  yahoo_symbol : "
                f"{row.get('yahoo_symbol')}"
            )

            print(
                f"  mapping      : "
                f"{row.get('price_mapping_source')}"
            )

            print(
                f"  close        : "
                f"{row.get('close')}"
            )

            for horizon in (
                1,
                5,
                20,
                60,
            ):
                print(
                    f"  return {horizon:>2}d : "
                    f"{row.get(f'forward_return_{horizon}d')}"
                )

    return target_features


def print_same_identity_features(
    features: pd.DataFrame,
) -> None:
    separator(
        "4. SAME IDENTITY THROUGH FEATURE DATASET"
    )

    for name, target in TARGETS.items():
        print()
        print(f"### {name}")

        rows = features.loc[
            features.apply(
                lambda row: matches_identity(
                    row,
                    target,
                ),
                axis=1,
            )
        ].copy()

        if rows.empty:
            print(
                "NO FEATURE ROWS FOR IDENTITY"
            )
            continue

        rows = rows.sort_values(
            "snapshot_date"
        )

        print(
            f"Feature rows: {len(rows)}"
        )

        for _, row in rows.tail(25).iterrows():
            print(
                f"  "
                f"{row['snapshot_date'].date()} "
                f"| security={row.get('security_key')} "
                f"| price={row.get('price_date')} "
                f"| symbol={row.get('yahoo_symbol')} "
                f"| mapping={row.get('price_mapping_source')}"
            )


def print_security_key_diagnosis(
    target_fi: dict,
    features: pd.DataFrame,
) -> None:
    separator(
        "5. SECURITY KEY -> FEATURE CONTINUITY"
    )

    for name, target in TARGETS.items():
        print()
        print(f"### {name}")

        fi_rows = target_fi[name]

        if fi_rows.empty:
            print(
                "SKIP: ingen FI-rad."
            )
            continue

        keys = sorted(
            fi_rows[
                "security_key"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        for key in keys:
            rows = features.loc[
                features["security_key"]
                == key
            ].sort_values(
                "snapshot_date"
            )

            print()
            print(
                f"SECURITY KEY: {key}"
            )

            if rows.empty:
                print(
                    "  NO FEATURE ROWS"
                )
                continue

            target_date = pd.Timestamp(
                target["event_date"]
            )

            before = rows.loc[
                rows["snapshot_date"]
                < target_date
            ]

            exact = rows.loc[
                rows["snapshot_date"]
                == target_date
            ]

            after = rows.loc[
                rows["snapshot_date"]
                > target_date
            ]

            print(
                "  before: "
                + (
                    str(
                        before[
                            "snapshot_date"
                        ].max().date()
                    )
                    if not before.empty
                    else "NONE"
                )
            )

            print(
                f"  exact : {len(exact)}"
            )

            print(
                "  after : "
                + (
                    str(
                        after[
                            "snapshot_date"
                        ].min().date()
                    )
                    if not after.empty
                    else "NONE"
                )
            )


def print_price_diagnosis(
    prices: pd.DataFrame,
) -> None:
    separator(
        "6. RAW PRICE DATA"
    )

    for name, target in TARGETS.items():
        print()
        print(f"### {name}")

        rows = prices.loc[
            (
                prices["isin"]
                .fillna("")
                .map(normalize)
                == normalize(target["isin"])
            )
            |
            (
                prices["yahoo_symbol"]
                .fillna("")
                .map(normalize)
                == normalize(
                    target["yahoo_symbol"]
                )
            )
            |
            (
                prices["issuer"]
                .fillna("")
                .map(normalize)
                == normalize(
                    target["issuer"]
                )
            )
        ].copy()

        if rows.empty:
            print(
                "PRICE DATA: MISSING"
            )
            continue

        print(
            f"price rows: {len(rows)}"
        )

        print(
            f"min date: "
            f"{rows['date'].min().date()}"
        )

        print(
            f"max date: "
            f"{rows['date'].max().date()}"
        )

        print(
            "symbols: "
            + str(
                sorted(
                    rows[
                        "yahoo_symbol"
                    ]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )
        )

        print()
        print(
            "rows around event:"
        )

        event_date = pd.Timestamp(
            target["event_date"]
        )

        around = rows.loc[
            rows["date"].between(
                event_date
                - pd.Timedelta(days=10),
                event_date
                + pd.Timedelta(days=70),
            )
        ].sort_values(
            "date"
        )

        if around.empty:
            print(
                "  NONE"
            )
        else:
            for _, row in around.iterrows():
                print(
                    f"  "
                    f"{row['date'].date()} "
                    f"| "
                    f"{row['yahoo_symbol']} "
                    f"| "
                    f"close={row['close']} "
                    f"| "
                    f"{row['_source_file']}"
                )


def print_feature_price_relation(
    target_features: dict,
) -> None:
    separator(
        "7. FEATURE -> PRICE RELATION"
    )

    for name, target in TARGETS.items():
        print()
        print(f"### {name}")

        rows = target_features[name]

        if rows.empty:
            print(
                "SKIP: feature saknas."
            )
            continue

        for _, row in rows.iterrows():
            snapshot_value = row.get(
                "snapshot_date"
            )

            if pd.isna(snapshot_value):
                print(
                    "  snapshot_date: MISSING"
                )
                continue

            snapshot = pd.Timestamp(
                snapshot_value
            ).date()

            price_value = row.get(
                "price_date"
            )

            if pd.isna(price_value):
                print(
                    f"  snapshot : {snapshot}"
                )
                print(
                    "  price    : MISSING"
                )
                print(
                    "  delta    : MISSING"
                )
                print(
                    f"  symbol   : "
                    f"{row.get('yahoo_symbol')}"
                )
                print(
                    f"  mapping  : "
                    f"{row.get('price_mapping_source')}"
                )
                continue

            price_date = pd.Timestamp(
                price_value
            ).date()

            days = (
                price_date - snapshot
            ).days

            print(
                f"  snapshot : {snapshot}"
            )

            print(
                f"  price    : {price_date}"
            )

            print(
                f"  delta    : "
                f"{days} calendar days"
            )

            print(
                f"  symbol   : "
                f"{row.get('yahoo_symbol')}"
            )

            print(
                f"  mapping  : "
                f"{row.get('price_mapping_source')}"
            )

            if days > 5:
                print(
                    "  WARNING: price_date ligger "
                    "mer än 5 kalenderdagar efter "
                    "snapshot_date."
                )


def print_forward_return_diagnosis(
    target_features: dict,
) -> None:
    separator(
        "8. FORWARD RETURN MATURITY"
    )

    today = date.today()

    for name, target in TARGETS.items():
        print()
        print(f"### {name}")

        rows = target_features[name]

        if rows.empty:
            print(
                "SKIP: feature saknas."
            )
            continue

        event_date = pd.Timestamp(
            target["event_date"]
        ).date()

        age = (
            today - event_date
        ).days

        print(
            f"event age: {age} calendar days"
        )

        expected = []

        if age >= 5:
            expected.extend(
                [1, 5]
            )

        if age >= 30:
            expected.append(20)

        if age >= 90:
            expected.append(60)

        for _, row in rows.iterrows():
            for horizon in expected:
                value = row.get(
                    f"forward_return_{horizon}d"
                )

                print(
                    f"  {horizon}d: "
                    + (
                        f"OK ({value})"
                        if value is not None
                        else "MISSING"
                    )
                )


def print_event_date_check(events) -> None:
    separator(
        "9. EVENT DATE"
    )

    today = date.today()

    for name, items in events.items():
        print()
        print(f"### {name}")

        if not items:
            print(
                "EVENT: MISSING"
            )
            continue

        for item in items:
            event_date = item["row"].get(
                "event_date"
            )

            if not event_date:
                continue

            value = pd.Timestamp(
                event_date
            ).date()

            status = (
                "FUTURE"
                if value > today
                else "OK"
            )

            print(
                f"  {value}: {status}"
            )


def main() -> None:
    separator(
        "FI -> FEATURE DIAGNOSTIK"
    )

    print(
        f"Repository: {ROOT}"
    )

    print(
        f"FI:        {FI_PATH}"
    )

    print(
        f"Features:  {FEATURE_DIR}"
    )

    print(
        f"Prices:    {PRICE_DIR}"
    )

    print(
        f"Events:    {EVENT_DIR}"
    )

    fi = load_fi()
    features = load_features()
    prices = load_prices()
    events = load_events()

    print()
    print(
        f"FI rows:       {len(fi)}"
    )

    print(
        f"Feature rows:  {len(features)}"
    )

    print(
        f"Price rows:    {len(prices)}"
    )

    print(
        f"Event files:   "
        f"{len(list(EVENT_DIR.glob('short_events_*.jsonl')))}"
    )

    print_event_diagnosis(
        events
    )

    target_fi = print_fi_diagnosis(
        fi
    )

    target_features = (
        print_feature_exact_diagnosis(
            features
        )
    )

    print_same_identity_features(
        features
    )

    print_security_key_diagnosis(
        target_fi,
        features,
    )

    print_price_diagnosis(
        prices
    )

    print_feature_price_relation(
        target_features
    )

    print_forward_return_diagnosis(
        target_features
    )

    print_event_date_check(
        events
    )

    separator(
        "DIAGNOSTIK KLAR"
    )

    print(
        "Inga filer ändrades av diagnostiken."
    )


if __name__ == "__main__":
    main()
