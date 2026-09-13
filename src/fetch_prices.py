"""
Hämtar historiska dagskurser från Yahoo Finance.

Instrumenten hämtas från:

    data/analysis/instrument_map.json

Ingen ticker är hårdkodad i denna fil.

Mappningen fungerar som brygga mellan:

    FI
      ↓
    LEI / emittent
      ↓
    ISIN
      ↓
    ticker
      ↓
    Yahoo-symbol
      ↓
    prisdata

Prisdata sparas som daterade JSONL-filer.
Tidigare körningar skrivs aldrig över.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"

INSTRUMENT_MAP = (
    ROOT
    / "data"
    / "analysis"
    / "instrument_map.json"
)


def load_instrument_map() -> dict:
    """
    Läser instrumentmappningen.

    Exempel:

    {
        "SE0000000001": {
            "isin": "SE0000000001",
            "lei": "549300...",
            "issuer": "Example AB",
            "ticker": "EXAMPLE",
            "yahoo_symbol": "EXAMPLE.ST"
        }
    }
    """

    if not INSTRUMENT_MAP.exists():
        raise FileNotFoundError(
            "Instrumentmappningen saknas: "
            f"{INSTRUMENT_MAP}"
        )

    content = INSTRUMENT_MAP.read_text(
        encoding="utf-8"
    ).strip()

    if not content:
        return {}

    mapping = json.loads(content)

    if not isinstance(mapping, dict):
        raise ValueError(
            "instrument_map.json måste innehålla ett JSON-objekt."
        )

    return mapping


def get_yahoo_symbols(
    mapping: dict,
) -> list[dict]:
    """
    Hämtar alla instrument som har en Yahoo-symbol.

    Returnerar en lista eftersom samma Yahoo-symbol inte
    ska hämtas flera gånger.
    """

    instruments: list[dict] = []
    seen_symbols: set[str] = set()

    for key, item in mapping.items():

        if not isinstance(item, dict):
            continue

        yahoo_symbol = (
            item.get("yahoo_symbol")
        )

        if not yahoo_symbol:
            continue

        yahoo_symbol = str(
            yahoo_symbol
        ).strip()

        if not yahoo_symbol:
            continue

        if yahoo_symbol in seen_symbols:
            continue

        seen_symbols.add(
            yahoo_symbol
        )

        instruments.append(
            {
                "map_key": key,
                "isin": item.get("isin"),
                "lei": item.get("lei"),
                "issuer": item.get("issuer"),
                "ticker": item.get("ticker"),
                "yahoo_symbol": yahoo_symbol,
            }
        )

    return instruments


def extract_close_series(
    data,
    symbol: str,
):
    """
    Hämtar Close från yfinance och hanterar både
    vanliga och MultiIndex-kolumner.
    """

    if data.empty:
        return None

    if "Close" not in data.columns:
        return None

    close = data["Close"]

    # Nyare yfinance kan returnera MultiIndex även
    # när endast en ticker hämtats.
    if hasattr(close, "columns"):

        if symbol in close.columns:
            close = close[symbol]

        elif len(close.columns) == 1:
            close = close.iloc[:, 0]

        else:
            return None

    return close


def fetch_prices(
    instruments: list[dict],
    start: str,
    end: str | None = None,
) -> list[dict]:
    """
    Hämtar dagliga priser för samtliga mappade instrument.
    """

    records: list[dict] = []

    for instrument in instruments:

        symbol = instrument[
            "yahoo_symbol"
        ]

        data = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            actions=False,
        )

        close = extract_close_series(
            data,
            symbol,
        )

        if close is None:
            continue

        for timestamp, value in close.items():

            if value is None:
                continue

            try:
                price = float(value)
            except (
                TypeError,
                ValueError,
            ):
                continue

            if price <= 0:
                continue

            records.append(
                {
                    "date": timestamp.strftime(
                        "%Y-%m-%d"
                    ),

                    "isin": instrument.get(
                        "isin"
                    ),

                    "lei": instrument.get(
                        "lei"
                    ),

                    "issuer": instrument.get(
                        "issuer"
                    ),

                    "ticker": instrument.get(
                        "ticker"
                    ),

                    "yahoo_symbol": symbol,

                    "close": price,
                }
            )

    return records


def write_jsonl(
    records: list[dict],
) -> Path:
    """Skriver en ny daterad prisfil."""

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        RAW_DIR
        / (
            "prices_"
            f"{date.today().isoformat()}.jsonl"
        )
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for record in records:

            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    return path


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Hämta historiska priser för "
            "mappade Blankdiss-instrument."
        )
    )

    parser.add_argument(
        "--start",
        default="2020-01-01",
        help=(
            "Första datum för prisdata. "
            "Standard: 2020-01-01"
        ),
    )

    parser.add_argument(
        "--end",
        default=None,
        help=(
            "Sista datum för prisdata. "
            "Yahoo använder slutdatum exklusivt."
        ),
    )

    args = parser.parse_args()

    mapping = load_instrument_map()

    instruments = get_yahoo_symbols(
        mapping
    )

    if not instruments:
        path = write_jsonl([])

        print(
            "Priser: 0 instrument mappade "
            f"→ {path}"
        )

        return

    records = fetch_prices(
        instruments=instruments,
        start=args.start,
        end=args.end,
    )

    path = write_jsonl(
        records
    )

    symbols = {
        record["yahoo_symbol"]
        for record in records
    }

    print(
        f"Priser: {len(symbols)} instrument, "
        f"{len(records)} observationer "
        f"→ {path}"
    )


if __name__ == "__main__":
    main()
