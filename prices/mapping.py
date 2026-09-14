"""
Hanterar mappningen mellan Finansinspektionens emittenter
och Yahoo Finance-symboler.
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
) -> str | None:

    quotes = yahoo_search(
        issuer
    )

    candidates: list[
        tuple
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

    if best_similarity < 0.35:
        return None

    return symbol


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


def get_latest_fi_records(
    fi_records: list[dict],
) -> dict[str, dict]:

    latest_by_lei: dict[
        str,
        dict,
    ] = {}

    for record in fi_records:

        lei = str(
            record.get(
                "lei",
                "",
            )
        ).strip()

        if not lei:
            continue

        snapshot_date = str(
            record.get(
                "snapshot_date",
                "",
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

    return latest_by_lei


def resolve_one_mapping(
    item: tuple[str, dict],
) -> tuple[
    str,
    str,
    str | None,
]:

    lei, record = item

    issuer = str(
        record.get(
            "issuer",
            "",
        )
    ).strip()

    if not issuer:
        return (
            lei,
            issuer,
            None,
        )

    symbol = choose_yahoo_symbol(
        issuer
    )

    return (
        lei,
        issuer,
        symbol,
    )


def build_instrument_map(
    existing: dict,
    fi_records: list[dict],
) -> tuple[
    dict,
    int,
    int,
]:

    mapping = dict(
        existing
    )

    latest_by_lei = (
        get_latest_fi_records(
            fi_records
        )
    )

    pending: list[
        tuple[str, dict]
    ] = []

    for lei, record in sorted(
        latest_by_lei.items()
    ):

        issuer = str(
            record.get(
                "issuer",
                "",
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

        pending.append(
            (
                lei,
                record,
            )
        )

    total = len(
        pending
    )

    if total == 0:

        print(
            "Mappning: inga nya "
            "emittenter att söka."
        )

        return (
            mapping,
            0,
            0,
        )

    print(
        "Mappning: "
        f"{total} emittenter behöver "
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
                resolve_one_mapping,
                item,
            ): item
            for item in pending
        }

        for future in as_completed(
            futures
        ):

            completed += 1

            try:

                (
                    lei,
                    issuer,
                    symbol,
                ) = future.result()

            except Exception as exc:

                item = futures[
                    future
                ]

                lei = item[0]

                issuer = str(
                    item[1].get(
                        "issuer",
                        "",
                    )
                )

                symbol = None

                print(
                    "Mappning: FEL - "
                    f"{issuer} "
                    f"({type(exc).__name__})"
                )

            if symbol is None:

                unresolved += 1

                print(
                    "Mappning: ej hittad - "
                    f"{issuer} "
                    f"[{completed}/{total}]"
                )

                continue

            existing_item = mapping.get(
                lei
            )

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
                f"{issuer} → {symbol} "
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
            }
        )

    return instruments
