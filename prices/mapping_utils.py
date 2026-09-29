from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


MAPPING_PATH = Path(
    "data/analysis/instrument_map.json"
)

FI_SNAPSHOT_DIR = Path(
    "data/raw/fi/aggregate/snapshots"
)

FI_RECONSTRUCTED_PATH = Path(
    "data/processed/fi/aggregate/reconstructed.jsonl"
)

KNOWN_YAHOO_SYMBOLS_PATH = Path(
    "prices/known_yahoo_symbols.jsonl"
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def clean_value(
    value: Any,
) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.lower() in {
        "none",
        "null",
        "nan",
        "nat",
        "n/a",
        "na",
        "-",
    }:
        return None

    return text


def normalize_name(
    value: Any,
) -> str:
    text = clean_value(value) or ""

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    text = text.lower()

    text = re.sub(
        r"\b("
        r"ab|aktiebolag|publ|plc|inc|corp|corporation|"
        r"ltd|limited|sa|se|nv|ag|holding|holdings|group"
        r")\b",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_ticker(
    value: Any,
) -> str | None:
    text = clean_value(value)

    if not text:
        return None

    return text.upper().strip()


def normalize_isin(
    value: Any,
) -> str | None:
    text = clean_value(value)

    if not text:
        return None

    text = text.upper().replace(
        " ",
        "",
    )

    if re.fullmatch(
        r"[A-Z0-9]{12}",
        text,
    ):
        return text

    return None


def name_similarity(
    a: Any,
    b: Any,
) -> float:
    a_norm = normalize_name(a)
    b_norm = normalize_name(b)

    if not a_norm or not b_norm:
        return 0.0

    return SequenceMatcher(
        None,
        a_norm,
        b_norm,
    ).ratio()


# ---------------------------------------------------------------------------
# Yahoo symbol conversion
# ---------------------------------------------------------------------------

def yahoo_symbol(
    ticker: Any,
    exchange: Any = None,
) -> str | None:
    """
    Convert a ticker + exchange into a Yahoo Finance symbol.

    We deliberately do not assume Stockholm when exchange is missing.
    """

    ticker_norm = normalize_ticker(
        ticker
    )

    if not ticker_norm:
        return None

    if "." in ticker_norm:
        return ticker_norm

    exchange_norm = clean_value(
        exchange
    )

    if not exchange_norm:
        return None

    exchange_norm = (
        exchange_norm.upper()
        .strip()
    )

    exchange_suffixes = {
        # Nasdaq Stockholm
        "STO": ".ST",
        "STOCKHOLM": ".ST",
        "NASDAQ STOCKHOLM": ".ST",
        "NASDAQ OMX STOCKHOLM": ".ST",

        # Nasdaq Copenhagen
        "CPH": ".CO",
        "COPENHAGEN": ".CO",
        "NASDAQ COPENHAGEN": ".CO",
        "NASDAQ OMX COPENHAGEN": ".CO",

        # Nasdaq Helsinki
        "HEL": ".HE",
        "HELSINKI": ".HE",
        "NASDAQ HELSINKI": ".HE",
        "NASDAQ OMX HELSINKI": ".HE",

        # Oslo
        "OSL": ".OL",
        "OSLO": ".OL",
        "OSLO BORS": ".OL",

        # London
        "LSE": ".L",
        "LONDON": ".L",
        "LONDON STOCK EXCHANGE": ".L",

        # Frankfurt / Xetra
        "FRA": ".F",
        "FRANKFURT": ".F",
        "XETRA": ".DE",

        # Paris
        "PAR": ".PA",
        "PARIS": ".PA",
        "EURONEXT PARIS": ".PA",

        # Amsterdam
        "AMS": ".AS",
        "AMSTERDAM": ".AS",
        "EURONEXT AMSTERDAM": ".AS",

        # Brussels
        "BRU": ".BR",
        "BRUSSELS": ".BR",
        "EURONEXT BRUSSELS": ".BR",

        # Milan
        "MIL": ".MI",
        "MILAN": ".MI",
        "BORSA ITALIANA": ".MI",

        # Madrid
        "MAD": ".MC",
        "MADRID": ".MC",
        "BME": ".MC",

        # Zurich
        "ZRH": ".SW",
        "ZURICH": ".SW",
        "SIX": ".SW",
        "SIX SWISS": ".SW",
    }

    suffix = exchange_suffixes.get(
        exchange_norm
    )

    if suffix:
        return f"{ticker_norm}{suffix}"

    return None


# ---------------------------------------------------------------------------
# Known Yahoo mappings
# ---------------------------------------------------------------------------

KNOWN_YAHOO_SYMBOLS_BY_ISIN: dict[str, str] = {
    "SE0000108227": "SKF-B.ST",
}


def _load_known_yahoo_symbols() -> dict[str, str]:
    if not KNOWN_YAHOO_SYMBOLS_PATH.exists():
        return {}

    result: dict[str, str] = {}

    try:
        with KNOWN_YAHOO_SYMBOLS_PATH.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(
                        line
                    )
                except json.JSONDecodeError:
                    continue

                if not isinstance(
                    record,
                    dict,
                ):
                    continue

                name = clean_value(
                    record.get("name")
                )

                symbol = clean_value(
                    record.get(
                        "yahoo_symbol"
                    )
                )

                if not name or not symbol:
                    continue

                normalized_name = (
                    normalize_name(name)
                )

                if not normalized_name:
                    continue

                result[
                    normalized_name
                ] = symbol

    except OSError:
        return {}

    return result


KNOWN_YAHOO_SYMBOLS: dict[str, str] = (
    _load_known_yahoo_symbols()
)


def _persist_known_yahoo_mapping(
    issuer: str | None,
    symbol: str | None,
) -> bool:
    name = clean_value(
        issuer
    )

    yahoo = clean_value(
        symbol
    )

    if not name or not yahoo:
        return False

    normalized_name = normalize_name(
        name
    )

    if not normalized_name:
        return False

    existing_symbol = (
        KNOWN_YAHOO_SYMBOLS.get(
            normalized_name
        )
    )

    if existing_symbol:
        if existing_symbol == yahoo:
            return False

        print(
            "Mappning: konflikt i "
            f"{KNOWN_YAHOO_SYMBOLS_PATH}: "
            f"{name!r} har redan "
            f"{existing_symbol}, ignorerar "
            f"{yahoo}."
        )

        return False

    KNOWN_YAHOO_SYMBOLS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record = {
        "name": normalized_name,
        "yahoo_symbol": yahoo,
    }

    with KNOWN_YAHOO_SYMBOLS_PATH.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )

    KNOWN_YAHOO_SYMBOLS[
        normalized_name
    ] = yahoo

    return True
