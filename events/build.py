"""
Bygger event-data från blankningssnapshots.

Event-data innehåller endast själva FI-händelsen.
Prisdata och forward returns kommer från det kanoniska
feature-datasetet och beräknas/enrichas inte här.

Identity-regel:
    issuer identity
        ↓
    entity identity
        ↓
    instrument identity
        ↓
    price identity

ISIN:et från själva eventet bevaras alltid.
Identitetslagret används för att avgöra vilken långsiktig
entity eventet tillhör och vilket instrument som gäller
för eventets datum.

Ett identifierat instrument får därefter fortfarande sin
prisidentitet från instrument_map.json.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from shared.instrument_identity import (
    load_identity,
)


ROOT = Path(__file__).resolve().parents[1]

FI_RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "fi"
    / "aggregate"
    / "snapshots"
)

EVENT_DIR = (
    ROOT
    / "data"
    / "events"
)

ANALYSIS_DIR = (
    ROOT
    / "data"
    / "analysis"
)

INSTRUMENT_MAP = (
    ANALYSIS_DIR
    / "instrument_map.json"
)


def read_jsonl(
    path: Path,
) -> list[dict]:
    if not path.exists():
        return []

    records: list[dict] = []

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


def read_all_jsonl(
    directory: Path,
    pattern: str,
) -> list[dict]:
    records: list[dict] = []

    for path in sorted(
        directory.glob(pattern)
    ):
        records.extend(
            read_jsonl(path)
        )

    return records


def load_instrument_map() -> dict:
    if not INSTRUMENT_MAP.exists():
        return {}

    return json.loads(
        INSTRUMENT_MAP.read_text(
            encoding="utf-8"
        )
    )


def normalize(
    value,
) -> str:
    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
    )


def normalize_isin(
    value,
) -> str:
    return (
        normalize(value)
        .replace(" ", "")
    )


def resolve_instrument(
    row: dict,
    mapping: dict,
) -> tuple[dict | None, str | None]:
    """
    Resolve event -> instrument deterministiskt.

    Prioritet:
        1. exakt ISIN
        2. exakt LEI om entydig
        3. exakt issuer om entydig

    En issuer får aldrig välja första bästa
    instrument när flera mappingar finns.
    """

    isin = normalize_isin(
        row.get("isin")
    )

    lei = normalize(
        row.get("lei")
    )

    issuer = normalize(
        row.get("issuer")
    )

    # ---------------------------------------------------------
    # 1. ISIN
    # ---------------------------------------------------------

    if isin:
        candidates = []

        for key, item in mapping.items():
            if not isinstance(
                item,
                dict,
            ):
                continue

            mapped_isin = normalize_isin(
                item.get("isin")
            )

            if (
                normalize(key) == isin
                or mapped_isin == isin
            ):
                candidates.append(
                    item
                )

        if len(candidates) == 1:
            return (
                candidates[0],
                "isin",
            )

        if len(candidates) > 1:
            # Samma ISIN får inte ge flera
            # konkurrerande instrument.
            unique_symbols = {
                normalize(
                    item.get(
                        "yahoo_symbol"
                    )
                )
                for item in candidates
            }

            unique_symbols.discard("")

            if len(unique_symbols) == 1:
                return (
                    candidates[0],
                    "isin",
                )

            return (
                None,
                "ambiguous_isin",
            )

    # ---------------------------------------------------------
    # 2. LEI
    # ---------------------------------------------------------

    if lei:
        candidates = [
            item
            for item in mapping.values()
            if isinstance(
                item,
                dict,
            )
            and normalize(
                item.get("lei")
            ) == lei
        ]

        if len(candidates) == 1:
            return (
                candidates[0],
                "lei",
            )

        if len(candidates) > 1:
            return (
                None,
                "ambiguous_lei",
            )

    # ---------------------------------------------------------
    # 3. Issuer
    # ---------------------------------------------------------

    if issuer:
        candidates = [
            item
            for item in mapping.values()
            if isinstance(
                item,
                dict,
            )
            and normalize(
                item.get("issuer")
            ) == issuer
        ]

        if len(candidates) == 1:
            return (
                candidates[0],
                "issuer",
            )

        if len(candidates) > 1:
            return (
                None,
                "ambiguous_issuer",
            )

    return (
        None,
        "unresolved",
    )


def load_fi_data() -> pd.DataFrame:
    records = read_all_jsonl(
        FI_RAW_DIR,
        "fi_aggregate_*.jsonl",
    )

    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame(
        records
    )

    required = {
        "position_date",
        "lei",
        "issuer",
        "short_interest_pct",
    }

    missing = (
        required
        - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            "FI-data saknar kolumner: "
            + ", ".join(
                sorted(missing)
            )
        )

    frame["position_date"] = (
        pd.to_datetime(
            frame["position_date"],
            errors="coerce",
        )
    )

    frame["short_interest_pct"] = (
        pd.to_numeric(
            frame["short_interest_pct"],
            errors="coerce",
        )
    )

    frame = frame.dropna(
        subset=[
            "position_date",
            "short_interest_pct",
        ]
    )

    return frame


def prepare_events(
    fi: pd.DataFrame,
    mapping: dict,
) -> list[dict]:
    if fi.empty:
        return []

    identity = load_identity()

    frame = fi.copy()

    frame = frame.sort_values(
        [
            "lei",
            "position_date",
        ]
    )

    # ---------------------------------------------------------
    # Behåll separata security-identiteter.
    #
    # LEI är issuer-identitet och får därför inte ensamt
    # användas som security-nyckel när samma issuer kan ha
    # flera värdepapper.
    # ---------------------------------------------------------

    identity_columns = [
        column
        for column in (
            "isin",
            "lei",
            "issuer",
        )
        if column in frame.columns
    ]

    frame = frame.drop_duplicates(
        subset=[
            *identity_columns,
            "position_date",
        ],
        keep="last",
    )

    # ---------------------------------------------------------
    # Föregående observation beräknas inom samma security
    # när ISIN finns.
    #
    # Om ISIN saknas används LEI som fallback.
    # ---------------------------------------------------------

    if "isin" in frame.columns:
        frame["_security_group"] = (
            frame["isin"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        frame.loc[
            frame["_security_group"] == "",
            "_security_group",
        ] = (
            "LEI:"
            + frame["lei"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    else:
        frame["_security_group"] = (
            "LEI:"
            + frame["lei"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    frame[
        "previous_short_interest_pct"
    ] = (
        frame.groupby(
            "_security_group"
        )["short_interest_pct"]
        .shift(1)
    )

    frame["change_pp"] = (
        frame["short_interest_pct"]
        - frame[
            "previous_short_interest_pct"
        ]
    )

    frame = frame[
        frame[
            "previous_short_interest_pct"
        ].notna()
    ]

    frame = frame[
        frame["change_pp"] != 0
    ]

    events: list[dict] = []

    for _, row in frame.iterrows():
        source_row = row.to_dict()

        event_date = row[
            "position_date"
        ]

        event_date_value = (
            event_date.date()
            if hasattr(
                event_date,
                "date",
            )
            else event_date
        )

        # -----------------------------------------------------
        # 1. Långsiktig identity
        #
        # Detta svarar på:
        #
        # "Vilken entity tillhör denna observation?"
        #
        # Matchningen är datumstyrd.
        # -----------------------------------------------------

        identity_match = identity.resolve(
            isin=source_row.get(
                "isin"
            ),
            lei=source_row.get(
                "lei"
            ),
            issuer=source_row.get(
                "issuer"
            ),
            target_date=event_date_value,
        )

        entity_id = (
            identity_match.entity_id
            if identity_match is not None
            else None
        )

        identity_resolution = (
            identity_match.resolution
            if identity_match is not None
            else None
        )

        # -----------------------------------------------------
        # 2. Befintlig instrument -> price mapping
        #
        # Detta ändrar inte eventets ISIN.
        # -----------------------------------------------------

        instrument, resolution = (
            resolve_instrument(
                source_row,
                mapping,
            )
        )

        if instrument is None:
            instrument = {}

        isin = (
            source_row.get("isin")
            or instrument.get("isin")
        )

        lei = (
            instrument.get("lei")
            or row.get("lei")
        )

        issuer = (
            instrument.get("issuer")
            or row.get("issuer")
        )

        ticker = instrument.get(
            "ticker"
        )

        yahoo_symbol = (
            instrument.get(
                "yahoo_symbol"
            )
        )

        event = {
            "event_date": (
                event_date.strftime(
                    "%Y-%m-%d"
                )
            ),
            "entity_id": entity_id,
            "isin": isin,
            "lei": lei,
            "issuer": issuer,
            "ticker": ticker,
            "yahoo_symbol": yahoo_symbol,
            "identity_resolution": (
                identity_resolution
            ),
            "instrument_resolution": (
                resolution
            ),
            "short_interest_pct": float(
                row[
                    "short_interest_pct"
                ]
            ),
            "previous_short_interest_pct": float(
                row[
                    "previous_short_interest_pct"
                ]
            ),
            "change_pp": float(
                row["change_pp"]
            ),
        }

        # -----------------------------------------------------
        # Inga returns beräknas här.
        #
        # Canonical pipeline:
        #
        # FI position_date
        #       ↓
        # entity identity
        #       ↓
        # instrument identity
        #       ↓
        # feature snapshot_date
        #       ↓
        # price identity
        #       ↓
        # price_date
        #       ↓
        # forward returns
        # -----------------------------------------------------

        events.append(
            event
        )

    return events


def write_events(
    events: list[dict],
) -> Path:
    EVENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        EVENT_DIR
        / (
            "short_events_"
            f"{date.today().isoformat()}.jsonl"
        )
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for event in events:
            handle.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                )
                + "\n"
            )

    return path


def main() -> None:
    fi = load_fi_data()

    if fi.empty:
        path = write_events([])

        print(
            f"Events: 0 → {path}"
        )

        return

    mapping = load_instrument_map()

    events = prepare_events(
        fi=fi,
        mapping=mapping,
    )

    path = write_events(
        events
    )

    mapped = sum(
        1
        for event in events
        if event.get(
            "yahoo_symbol"
        )
    )

    unresolved = sum(
        1
        for event in events
        if event.get(
            "identity_resolution"
        ) is None
        or event.get(
            "instrument_resolution"
        ) in {
            "unresolved",
            "ambiguous_isin",
            "ambiguous_lei",
            "ambiguous_issuer",
        }
    )

    print(
        f"Events: {len(events)} "
        f"({mapped} med prisinstrument, "
        f"{unresolved} olösta/ambigua) "
        f"→ {path}"
    )


if __name__ == "__main__":
    main()
