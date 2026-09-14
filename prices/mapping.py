from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import yfinance as yf


MAPPING_PATH = Path(
    "data/analysis/instrument_map.json"
)

FI_SNAPSHOT_DIR = Path(
    "data/raw/fi/aggregate/snapshots"
)

FI_RECONSTRUCTED_PATH = Path(
    "data/processed/fi/aggregate/reconstructed.jsonl"
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

    text = text.upper().strip()

    if text.endswith(".ST"):
        text = text[:-3]

    return text


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


def yahoo_symbol(
    ticker: Any,
) -> str | None:
    ticker_norm = normalize_ticker(
        ticker
    )

    if not ticker_norm:
        return None

    if ticker_norm.endswith(
        ".ST"
    ):
        return ticker_norm

    return (
        f"{ticker_norm}.ST"
    )


# ---------------------------------------------------------------------------
# Known Swedish / historical Yahoo mappings
#
# These are deliberately explicit.
# A known mapping is safer than fuzzy matching.
# ---------------------------------------------------------------------------

KNOWN_YAHOO_SYMBOLS: dict[str, str] = {
    # Large Swedish companies
    "autoliv": "ALIV-SDB.ST",
    "autoliv inc": "ALIV-SDB.ST",
    "atlas copco": "ATCO-B.ST",
    "atlas copco a": "ATCO-A.ST",
    "atlas copco b": "ATCO-B.ST",
    "skf": "SKF-B.ST",
    "skf b": "SKF-B.ST",
    "epiroc": "EPRO-B.ST",
    "epiroc b": "EPRO-B.ST",
    "investor": "INVE-B.ST",
    "investor b": "INVE-B.ST",
    "holmen": "HOLM-B.ST",
    "holmen b": "HOLM-B.ST",
    "ica gruppen": "ICA.ST",
    "ica": "ICA.ST",
    "ncc": "NCC-B.ST",
    "ncc b": "NCC-B.ST",
    "stora enso": "STE-R.ST",
    "stora enso r": "STE-R.ST",
    "tethys oil": "TETY.ST",
    "millicom international cellular": "TIGO",
    "millicom": "TIGO",
    "kindred group": "KIND-SDB.ST",
    "kindred": "KIND-SDB.ST",
    "resurs holding": "RESURS.ST",
    "resurs": "RESURS.ST",
    "sas": "SAS-DKK.CO",
    "sas ab": "SAS-DKK.CO",
    "lagercrantz": "LAGR-B.ST",
    "lagercrantz group": "LAGR-B.ST",
    "lundbergforetagen": "LUND-B.ST",
    "l e lundbergforetagen": "LUND-B.ST",
    "lundbergforetagen b": "LUND-B.ST",
    "nederman": "NMAN.ST",
    "nederman holding": "NMAN.ST",
    "stendorren": "STEFB.ST",
    "stendorren fastigheter": "STEFB.ST",
    "prevas": "PREV-B.ST",
    "prevas aktiebolag": "PREV-B.ST",
    "bjorn borg": "BORG.ST",
    "bjorn borg ab": "BORG.ST",
    "billerud": "BILL.ST",
    "billerud ab": "BILL.ST",
    "c reades": "CRED-A.ST",
    "creades": "CRED-A.ST",
    "cortus energy": "CE.ST",
    "haldex": "HLDX.ST",
    "kancera": "KAN.ST",
    "maha energy": "MAHA-A.ST",
    "medivir": "MVIR.ST",
    "nobina": "NOBINA.ST",
    "oscar properties": "OP.ST",
    "prostalund": "PLUN.ST",
    "resurs": "RESURS.ST",
    "recipharm": "RECI-B.ST",
    "swedol": "SWDL.ST",
    "terranet": "TERRNT-B.ST",
    "veoneer": "VNE-SDB.ST",
    "veoneer inc": "VNE-SDB.ST",

    # Historical / distressed / delisted names
    "fingerprint cards": "FING-B.ST",
    "fingerprint cards ab": "FING-B.ST",
    "klövern": "KLOV-B.ST",
    "klovern": "KLOV-B.ST",
    "ica gruppen aktiebolag": "ICA.ST",
    "nobina ab": "NOBINA.ST",
    "recipharm ab": "RECI-B.ST",
    "swedol ab": "SWDL.ST",
    "veoneer inc": "VNE-SDB.ST",

    # Companies where current Yahoo symbols are known
    "addnode group": "ANOD-B.ST",
    "billerud": "BILL.ST",
    "clas ohlson": "CLAS-B.ST",
    "diös fastigheter": "DIOS.ST",
    "eolus vind": "EOLU-B.ST",
    "enea": "ENEA.ST",
    "essity": "ESSITY-B.ST",
    "grangex": "GRANGX.ST",
    "granges": "GRNG.ST",
    "gränges": "GRNG.ST",
    "heba": "HEBA-B.ST",
    "latour": "LATO-B.ST",
    "knowit": "KNOW.ST",
    "orrön energy": "ORRON.ST",
    "roko": "ROKO-B.ST",
    "sectra": "SECT-B.ST",
    "sbb": "SBB-B.ST",
    "smart eye": "SEYE.ST",
    "zinzino": "ZZ-B.ST",
    "axichem": "AXIC-A.ST",
}


# ---------------------------------------------------------------------------
# FI field extraction
# ---------------------------------------------------------------------------

def _find_value_by_key(
    record: dict[str, Any],
    candidates: set[str],
) -> str | None:
    normalized_candidates = {
        re.sub(
            r"[^a-z0-9]",
            "",
            candidate.lower(),
        )
        for candidate in candidates
    }

    for key, value in record.items():
        normalized_key = re.sub(
            r"[^a-z0-9]",
            "",
            str(key).lower(),
        )

        if normalized_key in normalized_candidates:
            result = clean_value(value)

            if result:
                return result

    return None


def _find_nested_value(
    record: Any,
    candidates: set[str],
    *,
    max_depth: int = 5,
    depth: int = 0,
) -> str | None:
    if depth > max_depth:
        return None

    if isinstance(record, dict):
        direct = _find_value_by_key(
            record,
            candidates,
        )

        if direct:
            return direct

        for value in record.values():
            result = _find_nested_value(
                value,
                candidates,
                max_depth=max_depth,
                depth=depth + 1,
            )

            if result:
                return result

    elif isinstance(record, list):
        for value in record:
            result = _find_nested_value(
                value,
                candidates,
                max_depth=max_depth,
                depth=depth + 1,
            )

            if result:
                return result

    return None


def get_isin(
    record: dict[str, Any],
) -> str | None:
    value = _find_nested_value(
        record,
        {
            "isin",
            "ISIN",
            "instrument_isin",
            "instrumentIsin",
            "security_isin",
            "securityIsin",
        },
    )

    return normalize_isin(
        value
    )


def get_lei(
    record: dict[str, Any],
) -> str | None:
    return _find_nested_value(
        record,
        {
            "lei",
            "LEI",
            "issuer_lei",
            "issuerLei",
            "entity_lei",
            "entityLei",
        },
    )


def get_ticker(
    record: dict[str, Any],
) -> str | None:
    return _find_nested_value(
        record,
        {
            "ticker",
            "Ticker",
            "symbol",
            "Symbol",
            "instrument_ticker",
            "instrumentTicker",
            "security_ticker",
            "securityTicker",
        },
    )


def get_issuer(
    record: dict[str, Any],
) -> str | None:
    return _find_nested_value(
        record,
        {
            "issuer",
            "Issuer",
            "issuer_name",
            "issuerName",
            "company_name",
            "companyName",
        },
    )


def get_date(
    record: dict[str, Any],
) -> str | None:
    return _find_nested_value(
        record,
        {
            "snapshot_date",
            "snapshotDate",
            "position_date",
            "positionDate",
            "date",
            "Date",
        },
    )


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------

def _iter_jsonl(
    path: Path,
):
    if not path.exists():
        return

    with path.open(
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

            if isinstance(
                record,
                dict,
            ):
                yield record


# ---------------------------------------------------------------------------
# FI data
# ---------------------------------------------------------------------------

def read_all_fi_data() -> list[
    dict[str, Any]
]:
    records: list[
        dict[str, Any]
    ] = []

    files = sorted(
        FI_SNAPSHOT_DIR.glob(
            "fi_aggregate_*.jsonl"
        )
    )

    for path in files:
        for record in _iter_jsonl(
            path
        ):
            records.append(
                record
            )

    return records


# ---------------------------------------------------------------------------
# Reconstructed FI identity
# ---------------------------------------------------------------------------

def _read_reconstructed_identity() -> dict[
    str,
    dict[str, Any],
]:
    result: dict[
        str,
        dict[str, Any],
    ] = {}

    if not FI_RECONSTRUCTED_PATH.exists():
        return result

    for record in _iter_jsonl(
        FI_RECONSTRUCTED_PATH
    ):
        isin = get_isin(record)
        lei = get_lei(record)
        issuer = get_issuer(record)

        if not isin:
            continue

        key = None

        if lei:
            key = f"LEI:{lei}"

        elif issuer:
            key = (
                "ISSUER:"
                f"{normalize_name(issuer)}"
            )

        if not key:
            continue

        existing = result.get(
            key
        )

        if existing is None:
            result[key] = {
                "isin": isin,
                "lei": lei,
                "issuer": issuer,
            }

    return result


# ---------------------------------------------------------------------------
# Instrument identity
# ---------------------------------------------------------------------------

def instrument_key(
    *,
    isin: str | None,
    lei: str | None,
    ticker: str | None,
    issuer: str | None,
) -> str:
    if isin:
        return f"ISIN:{isin}"

    if lei:
        return f"LEI:{lei}"

    if ticker:
        return (
            "TICKER:"
            f"{normalize_ticker(ticker)}"
        )

    if issuer:
        return (
            "ISSUER:"
            f"{normalize_name(issuer)}"
        )

    raise ValueError(
        "Cannot identify FI instrument."
    )


def get_latest_fi_instruments(
    fi_records: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    latest: dict[
        str,
        dict[str, Any],
    ] = {}

    reconstructed = (
        _read_reconstructed_identity()
    )

    for record in fi_records:
        lei = get_lei(record)
        issuer = get_issuer(record)
        ticker = get_ticker(record)
        isin = get_isin(record)
        snapshot_date = get_date(
            record
        )

        identity = None

        if lei:
            identity = reconstructed.get(
                f"LEI:{lei}"
            )

        if (
            identity is None
            and issuer
        ):
            identity = reconstructed.get(
                "ISSUER:"
                + normalize_name(
                    issuer
                )
            )

        if identity and not isin:
            isin = identity.get(
                "isin"
            )

        try:
            key = instrument_key(
                isin=isin,
                lei=lei,
                ticker=ticker,
                issuer=issuer,
            )
        except ValueError:
            continue

        candidate = {
            "map_key": key,
            "isin": isin,
            "lei": lei,
            "issuer": issuer,
            "ticker": ticker,
            "date": snapshot_date,
        }

        existing = latest.get(
            key
        )

        if existing is None:
            latest[key] = candidate
            continue

        existing_date = existing.get(
            "date"
        )

        if (
            snapshot_date
            and (
                not existing_date
                or str(snapshot_date)
                > str(existing_date)
            )
        ):
            latest[key] = candidate

    instruments = sorted(
        latest.values(),
        key=lambda item: (
            item.get("issuer") or "",
            item.get("isin") or "",
            item.get("lei") or "",
        ),
    )

    print(
        "Mappning: "
        f"{len(instruments)} "
        "FI-instrument identifierade."
    )

    return instruments


# ---------------------------------------------------------------------------
# Existing mapping
# ---------------------------------------------------------------------------

def load_instrument_map() -> dict[
    str,
    dict[str, Any],
]:
    if not MAPPING_PATH.exists():
        return {}

    try:
        with MAPPING_PATH.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(
                handle
            )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}

    if not isinstance(
        data,
        dict,
    ):
        return {}

    return data


def save_instrument_map(
    mapping: dict[
        str,
        dict[str, Any],
    ],
) -> Path:
    MAPPING_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MAPPING_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            mapping,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

        handle.write("\n")

    return MAPPING_PATH


# ---------------------------------------------------------------------------
# Existing mapping migration
# ---------------------------------------------------------------------------

def _migrate_existing_mapping(
    existing: dict[
        str,
        dict[str, Any],
    ],
    fi_instruments: list[
        dict[str, Any]
    ],
) -> dict[
    str,
    dict[str, Any],
]:
    by_isin: dict[
        str,
        dict[str, Any],
    ] = {}

    by_lei: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    by_issuer: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for item in existing.values():
        if not isinstance(
            item,
            dict,
        ):
            continue

        isin = normalize_isin(
            item.get("isin")
        )

        lei = clean_value(
            item.get("lei")
        )

        issuer = clean_value(
            item.get("issuer")
        )

        if isin:
            by_isin[isin] = item

        if lei:
            by_lei.setdefault(
                lei,
                [],
            ).append(item)

        if issuer:
            by_issuer.setdefault(
                normalize_name(
                    issuer
                ),
                [],
            ).append(item)

    migrated: dict[
        str,
        dict[str, Any],
    ] = {}

    for instrument in fi_instruments:
        isin = normalize_isin(
            instrument.get("isin")
        )

        lei = clean_value(
            instrument.get("lei")
        )

        issuer = clean_value(
            instrument.get("issuer")
        )

        key = instrument[
            "map_key"
        ]

        old = None

        # 1. Exact ISIN
        if isin:
            old = by_isin.get(
                isin
            )

        # 2. Unique LEI
        if old is None and lei:
            candidates = by_lei.get(
                lei,
                [],
            )

            if len(candidates) == 1:
                old = candidates[0]

        # 3. Unique issuer
        if old is None and issuer:
            candidates = by_issuer.get(
                normalize_name(
                    issuer
                ),
                [],
            )

            if len(candidates) == 1:
                old = candidates[0]

        if old is not None:
            migrated[key] = {
                **old,
                "map_key": key,
                "isin": isin
                or old.get("isin"),
                "lei": lei
                or old.get("lei"),
                "issuer": issuer
                or old.get("issuer"),
            }

    return migrated


# ---------------------------------------------------------------------------
# Explicit mapping
# ---------------------------------------------------------------------------

def _explicit_mapping(
    instrument: dict[str, Any],
) -> tuple[
    str | None,
    str | None,
]:
    issuer = normalize_name(
        instrument.get("issuer")
    )

    if not issuer:
        return None, None

    symbol = KNOWN_YAHOO_SYMBOLS.get(
        issuer
    )

    if symbol:
        return (
            symbol,
            "known_name",
        )

    return None, None


# ---------------------------------------------------------------------------
# Ticker based mapping
# ---------------------------------------------------------------------------

def _ticker_mapping(
    instrument: dict[str, Any],
) -> tuple[
    str | None,
    str | None,
]:
    ticker = normalize_ticker(
        instrument.get("ticker")
    )

    if not ticker:
        return None, None

    symbol = yahoo_symbol(
        ticker
    )

    if not symbol:
        return None, None

    return (
        symbol,
        "fi_ticker",
    )


# ---------------------------------------------------------------------------
# Yahoo search
# ---------------------------------------------------------------------------

def _search_yahoo(
    query: str,
) -> list[dict[str, Any]]:
    try:
        search = yf.Search(
            query,
            max_results=10,
        )

        quotes = getattr(
            search,
            "quotes",
            None,
        )

        if not quotes:
            return []

        return [
            quote
            for quote in quotes
            if isinstance(
                quote,
                dict,
            )
        ]

    except Exception as exc:
        print(
            "Mappning: Yahoo-sökning "
            f"misslyckades för "
            f"{query!r}: {exc}"
        )

        return []


def _candidate_symbol(
    quote: dict[str, Any],
) -> str | None:
    symbol = clean_value(
        quote.get("symbol")
    )

    if not symbol:
        return None

    symbol = symbol.upper()

    # Swedish equities.
    if symbol.endswith(
        ".ST"
    ):
        return symbol

    # Some Yahoo results omit the
    # exchange suffix.
    exchange = clean_value(
        quote.get("exchange")
    )

    if (
        exchange
        and exchange.upper()
        in {
            "STO",
            "STOCKHOLM",
        }
    ):
        return (
            f"{symbol}.ST"
        )

    return None


def _search_candidates(
    instrument: dict[str, Any],
) -> list[
    tuple[
        str,
        float,
        str,
    ]
]:
    issuer = clean_value(
        instrument.get("issuer")
    )

    isin = normalize_isin(
        instrument.get("isin")
    )

    queries: list[
        tuple[str, str]
    ] = []

    if issuer:
        queries.append(
            (
                issuer,
                "issuer_search",
            )
        )

    normalized = normalize_name(
        issuer
    )

    if normalized:
        queries.append(
            (
                normalized,
                "normalized_name_search",
            )
        )

    if isin:
        queries.append(
            (
                isin,
                "isin_search",
            )
        )

    candidates: list[
        tuple[
            str,
            float,
            str,
        ]
    ] = []

    seen: set[str] = set()

    for query, source in queries:
        quotes = _search_yahoo(
            query
        )

        for quote in quotes:
            symbol = _candidate_symbol(
                quote
            )

            if not symbol:
                continue

            if symbol in seen:
                continue

            seen.add(symbol)

            quote_name = (
                quote.get("longname")
                or quote.get("shortname")
                or quote.get("name")
                or ""
            )

            score = name_similarity(
                issuer,
                quote_name,
            )

            # Exact-ish name match gets a
            # substantial boost.
            issuer_norm = normalize_name(
                issuer
            )

            quote_norm = normalize_name(
                quote_name
            )

            if (
                issuer_norm
                and quote_norm
                and (
                    issuer_norm
                    == quote_norm
                )
            ):
                score = 1.0

            candidates.append(
                (
                    symbol,
                    score,
                    source,
                )
            )

    candidates.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return candidates


def _search_mapping(
    instrument: dict[str, Any],
) -> tuple[
    str | None,
    str | None,
]:
    candidates = _search_candidates(
        instrument
    )

    if not candidates:
        return None, None

    best_symbol, best_score, source = (
        candidates[0]
    )

    # Conservative threshold.
    #
    # We prefer an unresolved instrument
    # over a dangerous false positive.
    if best_score >= 0.90:
        return (
            best_symbol,
            source,
        )

    # Exact Yahoo name may occasionally
    # have weak normalization similarity
    # because of legal-name differences.
    if best_score >= 0.80:
        if (
            len(candidates) == 1
            or (
                best_score
                - candidates[1][1]
                >= 0.10
            )
        ):
            return (
                best_symbol,
                source,
            )

    return None, None


# ---------------------------------------------------------------------------
# Build mapping
# ---------------------------------------------------------------------------

def build_instrument_map(
    *,
    existing: dict[
        str,
        dict[str, Any],
    ],
    fi_records: list[
        dict[str, Any]
    ],
) -> tuple[
    dict[str, dict[str, Any]],
    int,
    int,
]:
    fi_instruments = (
        get_latest_fi_instruments(
            fi_records
        )
    )

    migrated = (
        _migrate_existing_mapping(
            existing,
            fi_instruments,
        )
    )

    mapping: dict[
        str,
        dict[str, Any],
    ] = dict(
        migrated
    )

    new_mappings = 0
    unresolved = 0

    needs_search = [
        instrument
        for instrument in fi_instruments
        if instrument["map_key"]
        not in mapping
    ]

    print(
        "Mappning: "
        f"{len(needs_search)} instrument "
        "behöver Yahoo-sökning."
    )

    for index, instrument in enumerate(
        needs_search,
        start=1,
    ):
        key = instrument[
            "map_key"
        ]

        issuer = (
            instrument.get(
                "issuer"
            )
            or "?"
        )

        isin = instrument.get(
            "isin"
        )

        ticker = instrument.get(
            "ticker"
        )

        symbol = None
        source = None

        # ---------------------------------------------------------------
        # 1. Explicit known mapping
        # ---------------------------------------------------------------

        symbol, source = (
            _explicit_mapping(
                instrument
            )
        )

        # ---------------------------------------------------------------
        # 2. FI ticker -> .ST
        # ---------------------------------------------------------------

        if symbol is None:
            symbol, source = (
                _ticker_mapping(
                    instrument
                )
            )

        # ---------------------------------------------------------------
        # 3. Yahoo search
        # ---------------------------------------------------------------

        if symbol is None:
            symbol, source = (
                _search_mapping(
                    instrument
                )
            )

        if symbol:
            mapping[key] = {
                "map_key": key,
                "isin": isin,
                "lei": instrument.get(
                    "lei"
                ),
                "issuer": issuer,
                "ticker": ticker,
                "yahoo_symbol": symbol,
                "mapping_source": source,
            }

            new_mappings += 1

            print(
                "Mappning: hittad - "
                f"{issuer} "
                f"ticker={ticker} "
                f"isin={isin} "
                f"-> {symbol} "
                f"[{index}/"
                f"{len(needs_search)}]"
            )

        else:
            mapping[key] = {
                "map_key": key,
                "isin": isin,
                "lei": instrument.get(
                    "lei"
                ),
                "issuer": issuer,
                "ticker": ticker,
                "yahoo_symbol": None,
                "mapping_source": None,
            }

            unresolved += 1

            print(
                "Mappning: ej hittad - "
                f"{issuer} "
                f"ticker={ticker} "
                f"isin={isin} "
                f"[{index}/"
                f"{len(needs_search)}]"
            )

    print(
        "Mappning: "
        f"{new_mappings} nya, "
        f"{unresolved} olösta, "
        f"{len(mapping)} totalt"
    )

    return (
        mapping,
        new_mappings,
        unresolved,
    )


# ---------------------------------------------------------------------------
# Yahoo symbols
# ---------------------------------------------------------------------------

def get_yahoo_symbols(
    mapping: dict[
        str,
        dict[str, Any],
    ],
) -> list[
    dict[str, Any]
]:
    instruments: list[
        dict[str, Any]
    ] = []

    for item in mapping.values():
        if not isinstance(
            item,
            dict,
        ):
            continue

        symbol = clean_value(
            item.get(
                "yahoo_symbol"
            )
        )

        if not symbol:
            continue

        instruments.append(
            item
        )

    # Avoid duplicate Yahoo symbols.
    unique: dict[
        str,
        dict[str, Any],
    ] = {}

    for item in instruments:
        symbol = item[
            "yahoo_symbol"
        ]

        unique[symbol] = item

    return [
        unique[symbol]
        for symbol in sorted(
            unique
        )
    ]
