"""
Hämtar historiska dagskurser från Yahoo Finance.

Instrumenten kan komma från:

    data/analysis/instrument_map.json

Om instrumentmappningen saknas eller är ofullständig försöker
programmet automatiskt hitta Yahoo-symboler via bolagsnamn från
FI:s sparade blankningsdata.

Datakedja:

    FI
      ↓
    LEI / emittent
      ↓
    Yahoo Search
      ↓
    Yahoo-symbol
      ↓
    prisdata

Mappningen sparas i:

    data/analysis/instrument_map.json

Prisdata sparas som daterade JSONL-filer.
Tidigare körningar skrivs aldrig över.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date, timedelta
from difflib import SequenceMatcher
from pathlib import Path

import requests
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
ANALYSIS_DIR = ROOT / "data" / "analysis"

INSTRUMENT_MAP = (
    ANALYSIS_DIR
    / "instrument_map.json"
)

YAHOO_SEARCH_URL = (
    "https://query1.finance.yahoo.com/"
    "v1/finance/search"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(compatible; Blankdiss/1.0; "
        "+https://github.com/appsyndroms/Blankdiss)"
    )
}


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

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )
            except json.JSONDecodeError:
                continue

    return records


def read_all_fi_data() -> list[dict]:
    """Läser alla sparade FI-snapshots."""

    records: list[dict] = []

    for path in sorted(
        RAW_DIR.glob(
            "fi_aggregate_*.jsonl"
        )
    ):
        records.extend(
            read_jsonl(path)
        )

    return records


def normalize_name(
    value: str | None,
) -> str:
    """
    Normaliserar bolagsnamn inför jämförelse.

    Juridiska suffix och vanlig interpunktion tas bort.
    """

    if not value:
        return ""

    text = str(value).lower()

    text = (
        text
        .replace("ä", "a")
        .replace("å", "a")
        .replace("ö", "o")
    )

    text = re.sub(
        r"\b(aktiebolag|ab|publ|publ\.)\b",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


def name_similarity(
    left: str,
    right: str,
) -> float:
    """Beräknar enkel textlikhet mellan två namn."""

    left_normalized = normalize_name(
        left
    )

    right_normalized = normalize_name(
        right
    )

    if not left_normalized:
        return 0.0

    if not right_normalized:
        return 0.0

    return SequenceMatcher(
        None,
        left_normalized,
        right_normalized,
    ).ratio()


def yahoo_search(
    query: str,
) -> list[dict]:
    """
    Söker efter finansiella instrument via Yahoo Finance.

    Endast sökresultat från Yahoo returneras.
    """

    if not query.strip():
        return []

    params = {
        "q": query,
        "quotesCount": 10,
        "newsCount": 0,
        "listsCount": 0,
        "enableFuzzyQuery": "true",
    }

    try:

        response = requests.get(
            YAHOO_SEARCH_URL,
            params=params,
            headers=HEADERS,
            timeout=15,
        )

        response.raise_for_status()

        payload = response.json()

    except (
        requests.RequestException,
        ValueError,
    ):
        return []

    quotes = payload.get(
        "quotes",
        [],
    )

    if not isinstance(
        quotes,
        list,
    ):
        return []

    return [
        quote
        for quote in quotes
        if isinstance(
            quote,
            dict,
        )
    ]


def choose_yahoo_symbol(
    issuer: str,
) -> str | None:
    """
    Försöker hitta bästa Yahoo-symbol för en svensk emittent.

    Prioriterar:

    1. Svenska aktier (.ST)
    2. Equity
    3. Hög namnlikhet

    Returnerar None om ingen rimlig kandidat hittas.
    """

    quotes = yahoo_search(
        issuer
    )

    candidates: list[tuple] = []

    for quote in quotes:

        symbol = str(
            quote.get(
                "symbol",
                ""
            )
        ).strip()

        if not symbol:
            continue

        quote_type = str(
            quote.get(
                "quoteType",
                ""
            )
        ).upper()

        if quote_type != "EQUITY":
            continue

        if not symbol.upper().endswith(
            ".ST"
        ):
            continue

        name = (
            quote.get("longname")
            or quote.get("shortname")
            or ""
        )

        similarity = name_similarity(
            issuer,
            str(name),
        )

        candidates.append(
            (
                similarity,
                symbol,
                str(name),
            )
        )

    if not candidates:
        return None

    candidates.sort(
        reverse=True
    )

    best_similarity, symbol, _ = (
        candidates[0]
    )

    # Undvik tveksamma automatiska träffar.
    if best_similarity < 0.35:
        return None

    return symbol


def load_instrument_map() -> dict:
    """
    Läser befintlig instrumentmappning.

    En tom eller saknad fil ger ett tomt objekt.
    """

    if not INSTRUMENT_MAP.exists():
        return {}

    try:

        content = (
            INSTRUMENT_MAP
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )

        if not content:
            return {}

        mapping = json.loads(
            content
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}

    if not isinstance(
        mapping,
        dict,
    ):
        return {}

    return mapping


def build_instrument_map(
    existing: dict,
    fi_records: list[dict],
) -> tuple[dict, int, int]:
    """
    Kompletterar instrumentmappningen automatiskt.

    Befintliga manuella/validerade mappingar behålls.

    Returnerar:

        mapping
        antal nya mappingar
        antal olösta emittenter
    """

    mapping = dict(
        existing
    )

    latest_by_lei: dict[str, dict] = {}

    for record in fi_records:

        lei = str(
            record.get(
                "lei",
                ""
            )
        ).strip()

        if not lei:
            continue

        snapshot_date = str(
            record.get(
                "snapshot_date",
                ""
            )
        )

        previous = latest_by_lei.get(
            lei
        )

        if (
            previous is None
            or snapshot_date
            >= str(
                previous.get(
                    "snapshot_date",
                    "",
                )
            )
        ):
            latest_by_lei[lei] = record

    new_mappings = 0
    unresolved = 0

    for lei, record in sorted(
        latest_by_lei.items()
    ):

        issuer = str(
            record.get(
                "issuer",
                ""
            )
        ).strip()

        if not issuer:
            continue

        existing_item = mapping.get(
            lei
        )

        if (
            isinstance(
                existing_item,
                dict,
            )
            and existing_item.get(
                "yahoo_symbol"
            )
        ):
            continue

        symbol = choose_yahoo_symbol(
            issuer
        )

        if symbol is None:

            unresolved += 1

            print(
                "Mappning: ej hittad - "
                f"{issuer}"
            )

            continue

        mapping[lei] = {
            "isin": (
                existing_item.get(
                    "isin"
                )
                if isinstance(
                    existing_item,
                    dict,
                )
                else None
            ),

            "lei": lei,

            "issuer": issuer,

            "ticker": symbol.removesuffix(
                ".ST"
            ),

            "yahoo_symbol": symbol,
        }

        new_mappings += 1

        print(
            "Mappning: "
            f"{issuer} → {symbol}"
        )

        # Lite luft mellan Yahoo-anropen.
        time.sleep(
            0.15
        )

    return (
        mapping,
        new_mappings,
        unresolved,
    )


def save_instrument_map(
    mapping: dict,
) -> Path:
    """Sparar instrumentmappningen."""

    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    INSTRUMENT_MAP.write_text(
        json.dumps(
            mapping,
            ensure_ascii=False,
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )

    return INSTRUMENT_MAP


def get_yahoo_symbols(
    mapping: dict,
) -> list[dict]:
    """
    Hämtar alla instrument som har en Yahoo-symbol.

    Samma Yahoo-symbol hämtas bara en gång.
    """

    instruments: list[dict] = []

    seen_symbols: set[str] = set()

    for key, item in mapping.items():

        if not isinstance(
            item,
            dict,
        ):
            continue

        yahoo_symbol = str(
            item.get(
                "yahoo_symbol",
                ""
            )
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

                "isin": item.get(
                    "isin"
                ),

                "lei": item.get(
                    "lei"
                ),

                "issuer": item.get(
                    "issuer"
                ),

                "ticker": item.get(
                    "ticker"
                ),

                "yahoo_symbol": (
                    yahoo_symbol
                ),
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

    if hasattr(
        close,
        "columns",
    ):

        if symbol in close.columns:
            close = close[symbol]

        elif len(
            close.columns
        ) == 1:
            close = close.iloc[
                :, 0
            ]

        else:
            return None

    return close


def fetch_prices(
    instruments: list[dict],
    start: str,
    end: str | None = None,
) -> list[dict]:
    """Hämtar dagliga priser för samtliga instrument."""

    records: list[dict] = []

    for instrument in instruments:

        symbol = instrument[
            "yahoo_symbol"
        ]

        try:

            data = yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                actions=False,
            )

        except Exception as exc:

            print(
                "Pris: FEL - "
                f"{symbol} "
                f"({type(exc).__name__})"
            )

            continue

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
                price = float(
                    value
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if price <= 0:
                continue

            records.append(
                {
                    "date": (
                        timestamp.strftime(
                            "%Y-%m-%d"
                        )
                    ),

                    "isin": (
                        instrument.get(
                            "isin"
                        )
                    ),

                    "lei": (
                        instrument.get(
                            "lei"
                        )
                    ),

                    "issuer": (
                        instrument.get(
                            "issuer"
                        )
                    ),

                    "ticker": (
                        instrument.get(
                            "ticker"
                        )
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
        default=None,
        help=(
            "Första datum för prisdata. "
            "Standard: 180 dagar bakåt."
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

    # Vi behöver inte hämta sex års prisdata
    # vid varje körning. 180 dagar räcker gott
    # för 60 handelsdagars framtida avkastning
    # och ger mycket mindre rådata.
    if args.start is None:

        start_date = (
            date.today()
            - timedelta(
                days=180
            )
        )

        start = (
            start_date.isoformat()
        )

    else:

        start = args.start

    fi_records = read_all_fi_data()

    existing_mapping = (
        load_instrument_map()
    )

    (
        mapping,
        new_mappings,
        unresolved,
    ) = build_instrument_map(
        existing=existing_mapping,
        fi_records=fi_records,
    )

    save_instrument_map(
        mapping
    )

    instruments = get_yahoo_symbols(
        mapping
    )

    if not instruments:

        path = write_jsonl([])

        print(
            "Priser: 0 instrument "
            "mappade → "
            f"{path}"
        )

        print(
            "Mappning: "
            f"{new_mappings} nya, "
            f"{unresolved} olösta"
        )

        return

    records = fetch_prices(
        instruments=instruments,
        start=start,
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

    print(
        "Mappning: "
        f"{new_mappings} nya, "
        f"{unresolved} olösta, "
        f"{len(instruments)} totalt"
    )


if __name__ == "__main__":
    main()
