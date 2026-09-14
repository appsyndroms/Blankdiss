"""
Hanterar mappningen mellan Finansinspektionens instrument
och Yahoo Finance-symboler.
Primär instrumentidentitet är ISIN.
LEI används som emittentidentitet.
Detta är viktigt eftersom samma emittent kan ha flera
instrument, exempelvis A- och B-aktier.
"""
from __future__ import annotations
import json
import re
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from difflib import SequenceMatcher
from pathlib import Path
import requests
ROOT = Path(__file__).resolve().parents[1]
FI_RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "fi"
    / "aggregate"
    / "snapshots"
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
MAX_SEARCH_WORKERS = 6
SEARCH_TIMEOUT_SECONDS = 8
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
    records: list[dict] = []
    for path in sorted(
        FI_RAW_DIR.glob(
            "fi_aggregate_*.jsonl"
        )
    ):
        records.extend(
            read_jsonl(path)
        )
    return records
def clean_value(
    value,
) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text
def get_isin(
    record: dict,
) -> str | None:
    """
    Hämtar ISIN från FI-record.
    Vi accepterar några vanliga alternativa
    fältnamn för att göra mappningen robust.
    """
    for field in (
        "isin",
        "ISIN",
        "instrument_isin",
        "security_isin",
    ):
        value = clean_value(
            record.get(field)
        )
        if value:
            return value.upper()
    return None
def get_lei(
    record: dict,
) -> str | None:
    for field in (
        "lei",
        "LEI",
        "issuer_lei",
    ):
        value = clean_value(
            record.get(field)
        )
        if value:
            return value
    return None
def get_issuer(
    record: dict,
) -> str:
    for field in (
        "issuer",
        "issuer_name",
        "emittent",
        "emittent_namn",
    ):
        value = clean_value(
            record.get(field)
        )
        if value:
            return value
    return ""
def get_ticker(
    record: dict,
) -> str | None:
    for field in (
        "ticker",
        "instrument",
        "instrument_ticker",
        "symbol",
    ):
        value = clean_value(
            record.get(field)
        )
        if value:
            return value
    return None
def normalize_name(
    value: str | None,
) -> str:
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
def normalize_ticker(
    value: str | None,
) -> str:
    if not value:
        return ""
    return (
        str(value)
        .strip()
        .upper()
        .replace(" ", "")
    )
def name_similarity(
    left: str,
    right: str,
) -> float:
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
            timeout=SEARCH_TIMEOUT_SECONDS,
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
    ticker: str | None,
) -> tuple[
    str | None,
    str,
]:
    """
    Försöker först hitta Yahoo-symbol via ticker.
    Faller därefter tillbaka till issuer-namn.
    Returnerar:
        (yahoo_symbol, source)
    """
    normalized_ticker = normalize_ticker(
        ticker
    )
    quotes: list[dict] = []
    # ---------------------------------------------------------
    # 1. Försök med FI:s ticker.
    # ---------------------------------------------------------
    if normalized_ticker:
        quotes = yahoo_search(
            normalized_ticker
        )
        ticker_candidates: list[
            tuple[
                int,
                float,
                str,
            ]
        ] = []
        for quote in quotes:
            symbol = str(
                quote.get(
                    "symbol",
                    "",
                )
            ).strip()
            if not symbol:
                continue
            quote_type = str(
                quote.get(
                    "quoteType",
                    "",
                )
            ).upper()
            if quote_type != "EQUITY":
                continue
            if not symbol.upper().endswith(
                ".ST"
            ):
                continue
            yahoo_ticker = normalize_ticker(
                symbol.removesuffix(".ST")
            )
            ticker_match = int(
                yahoo_ticker
                == normalized_ticker
            )
            name = (
                quote.get("longname")
                or quote.get("shortname")
                or ""
            )
            similarity = name_similarity(
                issuer,
                str(name),
            )
            ticker_candidates.append(
                (
                    ticker_match,
                    similarity,
                    symbol,
                )
            )
        if ticker_candidates:
            ticker_candidates.sort(
                reverse=True
            )
            (
                ticker_match,
                _similarity,
                symbol,
            ) = ticker_candidates[0]
            if ticker_match == 1:
                return (
                    symbol,
                    "fi_ticker",
                )
    # ---------------------------------------------------------
    # 2. Fallback: sök på issuer.
    # ---------------------------------------------------------
    quotes = yahoo_search(
        issuer
    )
    candidates: list[
        tuple[
            float,
            str,
        ]
    ] = []
    for quote in quotes:
        symbol = str(
            quote.get(
                "symbol",
                "",
            )
        ).strip()
        if not symbol:
            continue
        quote_type = str(
            quote.get(
                "quoteType",
                "",
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
            )
        )
    if not candidates:
        return (
            None,
            "unresolved",
        )
    candidates.sort(
        reverse=True
    )
    best_similarity, symbol = (
        candidates[0]
    )
    if best_similarity < 0.35:
        return (
            None,
            "unresolved",
        )
    return (
        symbol,
        "yahoo_name",
    )
def load_instrument_map() -> dict:
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
def instrument_key(
    record: dict,
) -> str | None:
    """
    Primärnyckel:
        ISIN
    Om ISIN saknas använder vi en
    deterministisk fallback baserad på:
        LEI + ticker
    Vi använder alltså inte längre LEI
    ensam som instrumentnyckel.
    """
    isin = get_isin(
        record
    )
    if isin:
        return isin
    lei = get_lei(
        record
    )
    ticker = get_ticker(
        record
    )
    if lei and ticker:
        return (
            f"{lei}:"
            f"{normalize_ticker(ticker)}"
        )
    if lei:
        return lei
    return None
def get_latest_fi_instruments(
    fi_records: list[dict],
) -> dict[str, dict]:
    """
    Bygger senaste FI-record per instrument.
    Till skillnad från den gamla implementationen
    tappar vi inte flera instrument för samma LEI.
    """
    latest: dict[
        str,
        dict,
    ] = {}
    for record in fi_records:
        key = instrument_key(
            record
        )
        if not key:
            continue
        snapshot_date = str(
            record.get(
                "snapshot_date",
                record.get(
                    "date",
                    "",
                ),
            )
        )
        previous = latest.get(
            key
        )
        if (
            previous is None
            or snapshot_date
            >= str(
                previous.get(
                    "snapshot_date",
                    previous.get(
                        "date",
                        "",
                    ),
                )
            )
        ):
            latest[key] = record
    return latest
def build_instrument_map(
    existing: dict,
    fi_records: list[dict],
) -> tuple[
    dict,
    int,
    int,
]:
    latest_by_instrument = (
        get_latest_fi_instruments(
            fi_records
        )
    )
    mapping: dict = {}
    # ---------------------------------------------------------
    # Bygg en normaliserad mapping från FI:s aktuella instrument.
    #
    # Det gör att gamla LEI-baserade entries kan migreras
    # till ISIN-baserade entries.
    # ---------------------------------------------------------
    for key, record in sorted(
        latest_by_instrument.items()
    ):
        isin = get_isin(
            record
        )
        lei = get_lei(
            record
        )
        issuer = get_issuer(
            record
        )
        ticker = get_ticker(
            record
        )
        existing_item = None
        # Försök först hitta exakt ISIN i gammal mapping.
        if isin:
            candidate = existing.get(
                isin
            )
            if isinstance(
                candidate,
                dict,
            ):
                existing_item = candidate
        # Fallback för gamla LEI-baserade mappings.
        if (
            existing_item is None
            and lei
        ):
            candidate = existing.get(
                lei
            )
            if isinstance(
                candidate,
                dict,
            ):
                existing_item = candidate
        mapping[key] = {
            "isin": isin,
            "lei": lei,
            "issuer": issuer,
            "ticker": ticker,
            "yahoo_symbol": (
                existing_item.get(
                    "yahoo_symbol"
                )
                if isinstance(
                    existing_item,
                    dict,
                )
                else None
            ),
            "mapping_source": (
                existing_item.get(
                    "mapping_source"
                )
                if isinstance(
                    existing_item,
                    dict,
                )
                else None
            ),
        }
    pending: list[
        tuple[
            str,
            dict,
        ]
    ] = []
    for key, record in sorted(
        latest_by_instrument.items()
    ):
        item = mapping.get(
            key
        )
        if not isinstance(
            item,
            dict,
        ):
            continue
        if item.get(
            "yahoo_symbol"
        ):
            continue
        pending.append(
            (
                key,
                record,
            )
        )
    total = len(
        pending
    )
    print(
        "Mappning: "
        f"{len(latest_by_instrument)} FI-instrument "
        "identifierade."
    )
    if total == 0:
        print(
            "Mappning: inga nya "
            "instrument att söka."
        )
        return (
            mapping,
            0,
            0,
        )
    print(
        "Mappning: "
        f"{total} instrument behöver "
        "Yahoo-sökning."
    )
    new_mappings = 0
    unresolved = 0
    completed = 0
    with ThreadPoolExecutor(
        max_workers=MAX_SEARCH_WORKERS
    ) as executor:
        futures = {
            executor.submit(
                choose_yahoo_symbol,
                get_issuer(record),
                get_ticker(record),
            ): (
                key,
                record,
            )
            for key, record in pending
        }
        for future in as_completed(
            futures
        ):
            completed += 1
            key, record = futures[
                future
            ]
            issuer = get_issuer(
                record
            )
            ticker = get_ticker(
                record
            )
            isin = get_isin(
                record
            )
            lei = get_lei(
                record
            )
            try:
                (
                    yahoo_symbol,
                    source,
                ) = future.result()
            except Exception as exc:
                yahoo_symbol = None
                source = "error"
                print(
                    "Mappning: FEL - "
                    f"{issuer} "
                    f"({type(exc).__name__})"
                )
            if yahoo_symbol is None:
                unresolved += 1
                mapping[key][
                    "mapping_source"
                ] = source
                print(
                    "Mappning: ej hittad - "
                    f"{issuer} "
                    f"ticker={ticker} "
                    f"isin={isin} "
                    f"[{completed}/{total}]"
                )
                continue
            mapping[key][
                "yahoo_symbol"
            ] = yahoo_symbol
            mapping[key][
                "mapping_source"
            ] = source
            new_mappings += 1
            print(
                "Mappning: "
                f"{issuer} "
                f"ticker={ticker} "
                f"isin={isin} "
                f"→ {yahoo_symbol} "
                f"({source}) "
                f"[{completed}/{total}]"
            )
    return (
        mapping,
        new_mappings,
        unresolved,
    )
def save_instrument_map(
    mapping: dict,
) -> Path:
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
    instruments: list[
        dict
    ] = []
    seen_symbols: set[
        str
    ] = set()
    for key, item in mapping.items():
        if not isinstance(
            item,
            dict,
        ):
            continue
        yahoo_symbol = str(
            item.get(
                "yahoo_symbol",
                "",
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
                "mapping_source": item.get(
                    "mapping_source"
                ),
            }
        )
    return instruments
