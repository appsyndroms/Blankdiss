"""
Bygger event-data från blankningssnapshots och prisdata.

Datakedja:

FI LEI / emittent
        ↓
instrument_map.json
        ↓
ISIN / ticker / Yahoo-symbol
        ↓
prisdata
        ↓
framtida avkastning

Event-filerna är härledda data och kan därför byggas om
från rådata när som helst.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
EVENT_DIR = ROOT / "data" / "events"
ANALYSIS_DIR = ROOT / "data" / "analysis"

INSTRUMENT_MAP = (
    ANALYSIS_DIR
    / "instrument_map.json"
)


def read_jsonl(path: Path) -> list[dict]:
    """Läser en JSONL-fil."""

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
    pattern: str,
) -> list[dict]:
    """Läser samtliga filer som matchar ett mönster."""

    records: list[dict] = []

    for path in sorted(
        RAW_DIR.glob(pattern)
    ):
        records.extend(
            read_jsonl(path)
        )

    return records


def load_instrument_map() -> dict:
    """
    Läser ISIN/LEI/ticker-mappningen.

    Format:

    {
        "SE0012345678": {
            "isin": "SE0012345678",
            "lei": "...",
            "issuer": "Example AB",
            "ticker": "EXAMPLE",
            "yahoo_symbol": "EXAMPLE.ST"
        }
    }

    Nyckeln kan vara ISIN, LEI eller valfri identifierare.
    """

    if not INSTRUMENT_MAP.exists():
        return {}

    return json.loads(
        INSTRUMENT_MAP.read_text(
            encoding="utf-8"
        )
    )


def normalize(value) -> str:
    """Normaliserar text för jämförelse."""

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
    """
    Försöker hitta rätt instrument.

    Prioritetsordning:

    1. ISIN
    2. LEI
    3. exakt emittentnamn
    """

    isin = normalize(
        row.get("isin")
    )

    lei = normalize(
        row.get("lei")
    )

    issuer = normalize(
        row.get("issuer")
    )

    # Direkt träff på ISIN.
    if isin:
        for key, item in mapping.items():

            if normalize(key) == isin:
                return item

            if normalize(
                item.get("isin")
            ) == isin:
                return item

    # Träff på LEI.
    if lei:
        for key, item in mapping.items():

            if normalize(key) == lei:
                return item

            if normalize(
                item.get("lei")
            ) == lei:
                return item

    # Träff på emittentnamn.
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
    """Läser alla FI-råfiler."""

    records = read_all_jsonl(
        "fi_aggregate_*.jsonl"
    )

    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame(records)

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
            + ", ".join(sorted(missing))
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


def load_price_data() -> pd.DataFrame:
    """Läser all sparad prisdata."""

    records = read_all_jsonl(
        "prices_*.jsonl"
    )

    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame(records)

    if "date" not in frame.columns:
        return pd.DataFrame()

    if "ticker" not in frame.columns:
        return pd.DataFrame()

    if "close" not in frame.columns:
        return pd.DataFrame()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    frame["close"] = pd.to_numeric(
        frame["close"],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            "date",
            "close",
        ]
    )

    return frame.sort_values(
        [
            "ticker",
            "date",
        ]
    )


def calculate_return(
    prices: pd.DataFrame,
    ticker: str,
    event_date: pd.Timestamp,
    days: int,
) -> float | None:
    """
    Beräknar avkastningen från sista tillgängliga
    stängningskurs på/innan eventdatum till N:e
    handelsdagen därefter.
    """

    if not ticker:
        return None

    series = prices[
        prices["ticker"] == ticker
    ].copy()

    if series.empty:
        return None

    series = series.sort_values(
        "date"
    )

    before_or_on = series[
        series["date"] <= event_date
    ]

    if before_or_on.empty:
        return None

    base_index = (
        before_or_on.index[-1]
    )

    positions = list(
        series.index
    )

    try:
        base_position = positions.index(
            base_index
        )
    except ValueError:
        return None

    target_position = (
        base_position + days
    )

    if target_position >= len(
        positions
    ):
        return None

    target_index = positions[
        target_position
    ]

    base_close = float(
        series.loc[
            base_index,
            "close",
        ]
    )

    target_close = float(
        series.loc[
            target_index,
            "close",
        ]
    )

    if base_close == 0:
        return None

    return (
        target_close
        / base_close
        - 1.0
    )


def prepare_events(
    fi: pd.DataFrame,
    prices: pd.DataFrame,
    mapping: dict,
) -> list[dict]:
    """
    Skapar eventposter.

    För varje emittent jämförs aktuell blankning
    med föregående observerade nivå.
    """

    if fi.empty:
        return []

    frame = fi.copy()

    frame = frame.sort_values(
        [
            "lei",
            "position_date",
        ]
    )

    # Samma emittent + samma positionsdatum
    # ska bara förekomma en gång.
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

        for days in (
            1,
            5,
            20,
            60,
        ):

            event[
                f"return_{days}d"
            ] = calculate_return(
                prices=prices,
                ticker=ticker or "",
                event_date=event_date,
                days=days,
            )

        events.append(event)

    return events


def write_events(
    events: list[dict],
) -> Path:
    """Skriver en ny daterad event-fil."""

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

    prices = load_price_data()

    mapping = load_instrument_map()

    events = prepare_events(
        fi=fi,
        prices=prices,
        mapping=mapping,
    )

    path = write_events(events)

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
