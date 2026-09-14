from __future__ import annotations
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
import yfinance as yf
MAPPING_PATH = Path("data/analysis/instrument_map.json")
# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def clean_value(value: Any) -> str | None:
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
def normalize_name(value: Any) -> str:
    text = clean_value(value) or ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    text = text.lower()
    # Remove common legal suffixes.
    text = re.sub(
        r"\b("
        r"ab|aktiebolag|publ|plc|inc|corp|corporation|ltd|limited|"
        r"sa|se|nv|ag|holding|holdings|group"
        r")\b",
        " ",
        text,
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
def normalize_ticker(value: Any) -> str | None:
    text = clean_value(value)
    if not text:
        return None
    text = text.upper().strip()
    # FI/Yahoo ticker values occasionally contain exchange suffixes.
    if text.endswith(".ST"):
        text = text[:-3]
    return text
def name_similarity(a: Any, b: Any) -> float:
    a_norm = normalize_name(a)
    b_norm = normalize_name(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()
# ---------------------------------------------------------------------------
# FI field extraction
#
# FI has changed/varied field naming over time. Instead of assuming one
# exact field name, we inspect the actual record keys.
# ---------------------------------------------------------------------------
def _find_value_by_key(
    record: dict[str, Any],
    candidates: set[str],
) -> str | None:
    """
    Find a value using normalized key names.
    Example:
        "ISIN"              -> isin
        "instrumentIsin"    -> instrumentisin
        "instrument_isin"   -> instrumentisin
    """
    normalized_candidates = {
        re.sub(r"[^a-z0-9]", "", candidate.lower())
        for candidate in candidates
    }
    for key, value in record.items():
        normalized_key = re.sub(
            r"[^a-z0-9]",
            "",
            str(key).lower(),
        )
        if normalized_key in normalized_candidates:
            cleaned = clean_value(value)
            if cleaned:
                return cleaned
    return None
def _find_nested_value(
    record: Any,
    candidates: set[str],
    *,
    max_depth: int = 4,
    _depth: int = 0,
) -> str | None:
    """
    Recursively search dictionaries/lists for a field.
    This is intentionally defensive because the FI source format can contain
    nested instrument/security structures.
    """
    if _depth > max_depth:
        return None
    if isinstance(record, dict):
        direct = _find_value_by_key(record, candidates)
        if direct:
            return direct
        for value in record.values():
            result = _find_nested_value(
                value,
                candidates,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            if result:
                return result
    elif isinstance(record, list):
        for value in record:
            result = _find_nested_value(
                value,
                candidates,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            if result:
                return result
    return None
def get_isin(record: dict[str, Any]) -> str | None:
    value = _find_nested_value(
        record,
        {
            "isin",
            "ISIN",
            "instrument_isin",
            "instrumentIsin",
            "security_isin",
            "securityIsin",
            "instrumentisin",
            "securityisin",
        },
    )
    if not value:
        return None
    value = value.upper().replace(" ", "")
    # ISIN is exactly 12 alphanumeric characters.
    if re.fullmatch(r"[A-Z0-9]{12}", value):
        return value
    return None
def get_lei(record: dict[str, Any]) -> str | None:
    return _find_nested_value(
        record,
        {
            "lei",
            "LEI",
            "issuer_lei",
            "issuerLei",
            "entity_lei",
            "entityLei",
            "instrument_lei",
            "instrumentLei",
        },
    )
def get_ticker(record: dict[str, Any]) -> str | None:
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
            "short_name",
            "shortName",
        },
    )
def get_issuer(record: dict[str, Any]) -> str | None:
    return _find_nested_value(
        record,
        {
            "issuer",
            "Issuer",
            "issuer_name",
            "issuerName",
            "issuername",
            "company_name",
            "companyName",
            "name",
            "Name",
        },
    )
# ---------------------------------------------------------------------------
# Instrument identity
# ---------------------------------------------------------------------------
def instrument_key(
    *,
    isin: str | None,
    lei: str | None,
    ticker: str | None,
) -> str:
    """
    ISIN is the primary identity.
    If ISIN is genuinely unavailable, use a deterministic fallback.
    """
    if isin:
        return f"ISIN:{isin}"
    if lei and ticker:
        return f"LEI:{lei}:TICKER:{normalize_ticker(ticker)}"
    if lei:
        return f"LEI:{lei}"
    if ticker:
        return f"TICKER:{normalize_ticker(ticker)}"
    raise ValueError(
        "Cannot create instrument identity without ISIN, LEI or ticker."
    )
# ---------------------------------------------------------------------------
# FI snapshots
# ---------------------------------------------------------------------------
def _iter_jsonl_files() -> list[Path]:
    directory = Path(
        "data/raw/fi/aggregate/snapshots"
    )
    if not directory.exists():
        return []
    return sorted(directory.glob("fi_aggregate_*.jsonl"))
def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
def get_latest_fi_instruments() -> list[dict[str, Any]]:
    """
    Build the latest FI observation for every security/instrument.
    IMPORTANT:
    This is deliberately NOT grouped only by LEI.
    One issuer may have multiple listed securities/classes.
    """
    latest: dict[str, dict[str, Any]] = {}
    files = _iter_jsonl_files()
    if not files:
        raise FileNotFoundError(
            "No FI aggregate snapshot files found."
        )
    for path in files:
        for record in _read_jsonl(path):
            isin = get_isin(record)
            lei = get_lei(record)
            ticker = get_ticker(record)
            issuer = get_issuer(record)
            try:
                key = instrument_key(
                    isin=isin,
                    lei=lei,
                    ticker=ticker,
                )
            except ValueError:
                continue
            date = (
                record.get("date")
                or record.get("Date")
                or record.get("snapshot_date")
                or record.get("snapshotDate")
            )
            existing = latest.get(key)
            if existing is None:
                latest[key] = {
                    "map_key": key,
                    "isin": isin,
                    "lei": lei,
                    "issuer": issuer,
                    "ticker": ticker,
                    "date": date,
                }
                continue
            # Prefer the latest dated observation when possible.
            existing_date = existing.get("date")
            if date and (
                not existing_date
                or str(date) > str(existing_date)
            ):
                latest[key] = {
                    "map_key": key,
                    "isin": isin,
                    "lei": lei,
                    "issuer": issuer,
                    "ticker": ticker,
                    "date": date,
                }
    instruments = sorted(
        latest.values(),
        key=lambda item: (
            item.get("issuer") or "",
            item.get("isin") or "",
            item.get("ticker") or "",
        ),
    )
    print(
        f"Mappning: {len(instruments)} FI-instrument identifierade."
    )
    return instruments
# ---------------------------------------------------------------------------
# Existing mapping
# ---------------------------------------------------------------------------
def load_instrument_map() -> dict[str, dict[str, Any]]:
    if not MAPPING_PATH.exists():
        return {}
    try:
        with MAPPING_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data
def save_instrument_map(mapping: dict[str, dict[str, Any]]) -> None:
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
def _migrate_existing_mapping(
    existing: dict[str, dict[str, Any]],
    fi_instruments: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Reuse old Yahoo mappings where possible.
    Priority:
      1. exact ISIN
      2. exact LEI + ticker
      3. exact LEI, only when there is no ambiguity
    """
    by_isin: dict[str, dict[str, Any]] = {}
    by_lei_ticker: dict[tuple[str, str], dict[str, Any]] = {}
    by_lei: dict[str, list[dict[str, Any]]] = {}
    for old_key, item in existing.items():
        if not isinstance(item, dict):
            continue
        isin = clean_value(item.get("isin"))
        lei = clean_value(item.get("lei"))
        ticker = normalize_ticker(item.get("ticker"))
        if isin:
            by_isin[isin] = item
        if lei and ticker:
            by_lei_ticker[(lei, ticker)] = item
        if lei:
            by_lei.setdefault(lei, []).append(item)
    migrated: dict[str, dict[str, Any]] = {}
    for instrument in fi_instruments:
        isin = instrument.get("isin")
        lei = instrument.get("lei")
        ticker = normalize_ticker(instrument.get("ticker"))
        key = instrument["map_key"]
        old = None
        if isin:
            old = by_isin.get(isin)
        if old is None and lei and ticker:
            old = by_lei_ticker.get((lei, ticker))
        # Only use LEI fallback when there is exactly one old mapping.
        if old is None and lei:
            candidates = by_lei.get(lei, [])
            if len(candidates) == 1:
                old = candidates[0]
        result = {
            "map_key": key,
            "isin": isin,
            "lei": lei,
            "issuer": instrument.get("issuer"),
            "ticker": instrument.get("ticker"),
            "yahoo_symbol": None,
            "mapping_source": None,
        }
        if old:
            yahoo_symbol = clean_value(
                old.get("yahoo_symbol")
            )
            if yahoo_symbol:
                result["yahoo_symbol"] = yahoo_symbol
            result["mapping_source"] = (
                old.get("mapping_source")
                or "migrated"
            )
        migrated[key] = result
    return migrated
# ---------------------------------------------------------------------------
# Yahoo lookup
# ---------------------------------------------------------------------------
def _valid_yahoo_symbol(symbol: Any) -> bool:
    value = clean_value(symbol)
    if not value:
        return False
    if value.upper() in {
        "NONE",
        "NULL",
        "NAN",
        "NAT",
    }:
        return False
    return True
def _search_yahoo(
    *,
    issuer: str | None,
    ticker: str | None,
    isin: str | None,
) -> tuple[str | None, str | None]:
    """
    Return:
        (yahoo_symbol, mapping_source)
    Mapping priority:
      1. exact FI ticker -> .ST
      2. Yahoo search ticker
      3. Yahoo search issuer name
      4. ISIN search if Yahoo exposes it
    """
    normalized_ticker = normalize_ticker(ticker)
    # ---------------------------------------------------------------
    # 1. Direct Stockholm ticker
    # ---------------------------------------------------------------
    if normalized_ticker:
        candidate = f"{normalized_ticker}.ST"
        try:
            search = yf.Search(candidate)
            quotes = getattr(search, "quotes", []) or []
            for quote in quotes:
                symbol = clean_value(
                    quote.get("symbol")
                )
                quote_type = str(
                    quote.get("quoteType") or ""
                ).upper()
                if (
                    symbol
                    and symbol.upper() == candidate.upper()
                    and (
                        not quote_type
                        or quote_type == "EQUITY"
                    )
                ):
                    return symbol, "ticker"
        except Exception:
            pass
        # Yahoo sometimes does not return search results but the direct
        # ticker still exists.
        try:
            ticker_obj = yf.Ticker(candidate)
            info = ticker_obj.fast_info
            if info:
                return candidate, "ticker-direct"
        except Exception:
            pass
    # ---------------------------------------------------------------
    # 2. Search by issuer/ticker
    # ---------------------------------------------------------------
    queries = []
    if normalized_ticker:
        queries.append(normalized_ticker)
    if issuer:
        queries.append(issuer)
    if isin:
        queries.append(isin)
    seen_queries: set[str] = set()
    for query in queries:
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)
        try:
            search = yf.Search(query)
            quotes = getattr(search, "quotes", []) or []
        except Exception:
            continue
        candidates = []
        for quote in quotes:
            symbol = clean_value(
                quote.get("symbol")
            )
            if not _valid_yahoo_symbol(symbol):
                continue
            # We only want Swedish listed equities here.
            if not symbol.upper().endswith(".ST"):
                continue
            quote_type = str(
                quote.get("quoteType") or ""
            ).upper()
            if quote_type and quote_type != "EQUITY":
                continue
            quote_name = (
                quote.get("longname")
                or quote.get("shortname")
                or quote.get("name")
            )
            score = 0.0
            if normalized_ticker:
                yahoo_ticker = normalize_ticker(symbol)
                if yahoo_ticker == normalized_ticker:
                    score += 1.0
            if issuer and quote_name:
                score += name_similarity(
                    issuer,
                    quote_name,
                )
            candidates.append(
                (
                    score,
                    symbol,
                    quote_name,
                )
            )
        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )
        if candidates:
            score, symbol, _ = candidates[0]
            # Exact ticker match is always accepted.
            if normalized_ticker:
                if (
                    normalize_ticker(symbol)
                    == normalized_ticker
                ):
                    return symbol, "ticker-search"
            # Name matching needs a meaningful score.
            if issuer and score >= 0.35:
                return symbol, "issuer-search"
    return None, None
# ---------------------------------------------------------------------------
# Build mapping
# ---------------------------------------------------------------------------
def build_instrument_map(
    *,
    existing: dict[str, dict[str, Any]] | None,
    fi_records: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    if fi_records is None:
        fi_records = get_latest_fi_instruments()
    existing = existing or {}
    mapping = _migrate_existing_mapping(
        existing,
        fi_records,
    )
    unresolved = [
        item
        for item in mapping.values()
        if not _valid_yahoo_symbol(
            item.get("yahoo_symbol")
        )
    ]
    print(
        f"Mappning: {len(unresolved)} instrument behöver "
        f"Yahoo-sökning."
    )
    newly_mapped = 0
    unresolved_count = 0
    for index, item in enumerate(unresolved, start=1):
        issuer = item.get("issuer")
        ticker = item.get("ticker")
        isin = item.get("isin")
        symbol, source = _search_yahoo(
            issuer=issuer,
            ticker=ticker,
            isin=isin,
        )
        if symbol:
            item["yahoo_symbol"] = symbol
            item["mapping_source"] = source
            newly_mapped += 1
            print(
                "Mappning: hittad - "
                f"{issuer} "
                f"ticker={ticker} "
                f"isin={isin} "
                f"-> {symbol} "
                f"[{index}/{len(unresolved)}]"
            )
        else:
            item["yahoo_symbol"] = None
            item["mapping_source"] = None
            unresolved_count += 1
            print(
                "Mappning: ej hittad - "
                f"{issuer} "
                f"ticker={ticker} "
                f"isin={isin} "
                f"[{index}/{len(unresolved)}]"
            )
    print(
        f"Mappning: {newly_mapped} nya, "
        f"{unresolved_count} olösta, "
        f"{len(mapping)} totalt"
    )
    return mapping
# ---------------------------------------------------------------------------
# Yahoo instruments for price fetching
# ---------------------------------------------------------------------------
def get_yahoo_symbols(
    mapping: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    instruments = []
    for map_key, item in mapping.items():
        symbol = clean_value(
            item.get("yahoo_symbol")
        )
        # Never pass invalid placeholders to yfinance.
        if not _valid_yahoo_symbol(symbol):
            continue
        instruments.append(
            {
                "map_key": map_key,
                "isin": clean_value(item.get("isin")),
                "lei": clean_value(item.get("lei")),
                "issuer": clean_value(item.get("issuer")),
                "ticker": clean_value(item.get("ticker")),
                "yahoo_symbol": symbol,
                "mapping_source": item.get(
                    "mapping_source"
                ),
            }
        )
    # One Yahoo symbol should only be fetched once.
    unique: dict[str, dict[str, Any]] = {}
    for instrument in instruments:
        symbol = instrument["yahoo_symbol"]
        if symbol not in unique:
            unique[symbol] = instrument
    return sorted(
        unique.values(),
        key=lambda item: item["yahoo_symbol"],
    )
