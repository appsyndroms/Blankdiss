"""
Bygger event-data från blankningssnapshots.
Event-data innehåller endast själva FI-händelsen.
Prisdata och forward returns kommer från det kanoniska
feature-datasetet och beräknas/enrichas inte här.
"""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path
import pandas as pd
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
def resolve_instrument(
    row: dict,
    mapping: dict,
) -> dict | None:
    isin = normalize(
        row.get("isin")
    )
    lei = normalize(
        row.get("lei")
    )
    issuer = normalize(
        row.get("issuer")
    )
    if isin:
        for key, item in mapping.items():
            if normalize(key) == isin:
                return item
            if normalize(
                item.get("isin")
            ) == isin:
                return item
    if lei:
        for key, item in mapping.items():
            if normalize(key) == lei:
                return item
            if normalize(
                item.get("lei")
            ) == lei:
                return item
    if issuer:
        for item in mapping.values():
            mapped_issuer = normalize(
                item.get("issuer")
            )
            if (
                mapped_issuer
                and mapped_issuer == issuer
            ):
                return item
    return None
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
    missing = required - set(
        frame.columns
    )
    if missing:
        raise RuntimeError(
            "FI-data saknar kolumner: "
            + ", ".join(
                sorted(missing)
            )
        )
    frame["position_date"] = pd.to_datetime(
        frame["position_date"],
        errors="coerce",
    )
    frame["short_interest_pct"] = pd.to_numeric(
        frame["short_interest_pct"],
        errors="coerce",
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
    frame = fi.copy()
    frame = frame.sort_values(
        [
            "lei",
            "position_date",
        ]
    )
    frame = frame.drop_duplicates(
        subset=[
            "lei",
            "position_date",
        ],
        keep="last",
    )
    frame[
        "previous_short_interest_pct"
    ] = (
        frame.groupby("lei")[
            "short_interest_pct"
        ].shift(1)
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
        instrument = resolve_instrument(
            source_row,
            mapping,
        )
        if instrument is None:
            instrument = {}
        isin = instrument.get(
            "isin"
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
        yahoo_symbol = instrument.get(
            "yahoo_symbol"
        )
        event_date = row[
            "position_date"
        ]
        # Event-datumet kommer direkt från
        # FI position_date.
        #
        # Detta måste vara samma datum som
        # feature-datasetets snapshot_date.
        event = {
            "event_date": event_date.strftime(
                "%Y-%m-%d"
            ),
            "isin": isin,
            "lei": lei,
            "issuer": issuer,
            "ticker": ticker,
            "yahoo_symbol": yahoo_symbol,
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
        # Inga returns beräknas här.
        #
        # Canonical pipeline:
        #
        # FI position_date
        #       ↓
        # feature snapshot_date
        #       ↓
        # feature price_date
        #       ↓
        # feature forward_return_*
        #       ↓
        # web data_loader
        #
        # Event-filerna ska alltså inte ha en
        # konkurrerande return-beräkning.
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
        encoding="utf-8"
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
        if event.get("ticker")
    )
    print(
        f"Events: {len(events)} "
        f"({mapped} med prisinstrument) "
        f"→ {path}"
    )
if __name__ == "__main__":
    main()
