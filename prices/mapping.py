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
def normalize_ticker(value: Any) -> str | None:
    text = clean_value(value)
    if not text:
        return None
    text = text.upper().strip()
    if text.endswith(".ST"):
        text = text[:-3]
    return text
def normalize_isin(value: Any) -> str | None:
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
    return normalize_isin(value)
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
# FI data
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
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record
def read_all_fi_data() -> list[dict[str, Any]]:
    """
    Read all FI aggregate snapshot observations.
    This function intentionally returns the observations rather than
    immediately collapsing them to instruments. __main__.py expects
    the complete FI observation list.
    Aggregate FI data contains issuer/LEI information but normally
    does NOT contain ISIN or ticker.
    """
    records: list[dict[str, Any]] = []
    files = sorted(
        FI_SNAPSHOT_DIR.glob(
            "fi_aggregate_*.jsonl"
        )
    )
    for path in files:
        for record in _iter_jsonl(path):
            records.append(record)
    return records
# ---------------------------------------------------------------------------
# Reconstructed FI identity data
# ---------------------------------------------------------------------------
def _read_reconstructed_identity() -> dict[str, dict[str, Any]]:
    """
    Read reconstructed FI history and build the best available
    issuer/LEI -> ISIN identity lookup.
    The reconstructed history is the correct place to obtain ISIN
    because the aggregate snapshots themselves do not contain it.
    """
    result: dict[str, dict[str, Any]] = {}
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
                f"ISSUER:"
                f"{normalize_name(issuer)}"
            )
        if not key:
            continue
        existing = result.get(key)
        if existing is None:
            result[key] = {
                "isin": isin,
                "lei": lei,
                "issuer": issuer,
            }
            continue
        # If the same LEI has multiple ISINs, don't overwrite blindly.
        # Keep the first identity here; instrument-specific historical
        # mapping is handled separately through existing mappings.
        if existing.get("isin") == isin:
            continue
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
    """
    Prefer ISIN.
    If aggregate FI data does not contain ISIN, fall back to LEI.
    """
    if isin:
        return f"ISIN:{isin}"
    if lei:
        return f"LEI:{lei}"
    if ticker:
        return (
            f"TICKER:"
            f"{normalize_ticker(ticker)}"
        )
    if issuer:
        return (
            f"ISSUER:"
            f"{normalize_name(issuer)}"
        )
    raise ValueError(
        "Cannot identify FI instrument."
    )
def get_latest_fi_instruments(
    fi_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build one current FI instrument record per issuer/LEI.
    IMPORTANT:
    Aggregate FI snapshots are issuer-level data. They do not contain
    ISIN/ticker. Therefore this function does not invent those fields.
    """
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
        snapshot_date = (
            record.get("snapshot_date")
            or record.get("snapshotDate")
        )
        # Enrich with reconstructed identity when possible.
        identity = None
        if lei:
            identity = reconstructed.get(
                f"LEI:{lei}"
            )
        if identity is None and issuer:
            identity = reconstructed.get(
                "ISSUER:"
                + normalize_name(issuer)
            )
        if identity and not isin:
            isin = identity.get("isin")
        try:
            key = instrument_key(
                isin=isin,
                lei=lei,
                ticker=ticker,
                issuer=issuer,
            )
        except ValueError:
            continue
        existing = latest.get(key)
        if existing is None:
            latest[key] = {
                "map_key": key,
                "isin": isin,
                "lei": lei,
                "issuer": issuer,
                "ticker": ticker,
                "date": snapshot_date,
            }
            continue
        existing_date = existing.get(
            "date"
        )
        if snapshot_date and (
            not existing_date
            or str(snapshot_date)
            > str(existing_date)
        ):
            latest[key] = {
                "map_key": key,
                "isin": isin,
                "lei": lei,
                "issuer": issuer,
                "ticker": ticker,
                "date": snapshot_date,
            }
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
        f"{len(instruments)} FI-instrument "
        "identifierade."
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
            data = json.load(handle)
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}
    if not isinstance(data, dict):
        return {}
    return data
def save_instrument_map(
    mapping: dict[str, dict[str, Any]],
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
# Mapping migration
# ---------------------------------------------------------------------------
def _migrate_existing_mapping(
    existing: dict[str, dict[str, Any]],
    fi_instruments: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Reuse known Yahoo mappings.
    Matching priority:
      1. exact ISIN
      2. exact LEI + ticker
      3. exact LEI, only if unique
      4. exact issuer, only if unique
    """
    by_isin: dict[
        str,
        dict[str, Any],
    ] = {}
    by_lei_ticker: dict[
        tuple[str, str],
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
        if not isinstance(item, dict):
            continue
        isin = normalize_isin(
            item.get("isin")
        )
        lei = clean_value(
            item.get("lei")
        )
        ticker = normalize_ticker(
            item.get("ticker")
        )
        issuer = clean_value(
            item.get("issuer")
        )
        if isin:
            by_isin[isin] = item
        if lei and ticker:
            by_lei_ticker[
                (lei, ticker)
            ] = item
        if lei:
            by_lei.setdefault(
                lei,
                [],
            ).append(item)
        if issuer:
            by_issuer.setdefault(
                normalize_name(issuer),
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
        ticker = normalize_ticker(
            instrument.get("ticker")
        )
        issuer = clean_value(
            instrument.get("issuer")
        )
        key = instrument["map_key"]
        old = None
        # 1. ISIN
        if isin:
            old = by_isin.get(isin)
        # 2. LEI + ticker
        if (
            old is None
            and lei
            and ticker
        ):
            old = by_lei_ticker.get(
                (lei, ticker)
            )
        # 3. Unique LEI
        if old is None and lei:
            candidates = by_lei.get(
                lei,
                [],
            )
            if len(candidates) == 1:
                old = candidates[0]
        # 4. Unique issuer
        if old is None and issuer:
            candidates = by_issuer.get(
                normalize_name(issuer),
                [],
            )
            if len(candidates) == 1:
                old = candidates[0]
        result = {
            "map_key": key,
            "isin": isin,
            "lei": lei,
            "issuer": issuer,
            "ticker": ticker,
            "yahoo_symbol": None,
            "mapping_source": None,
        }
        if old:
            yahoo_symbol = clean_value(
                old.get("yahoo_symbol")
            )
            if (
                yahoo_symbol
                and _valid_yahoo_symbol(
                    yahoo_symbol
                )
            ):
                result[
                    "yahoo_symbol"
                ] = yahoo_symbol
            result[
                "mapping_source"
            ] = (
                old.get(
                    "mapping_source"
                )
                or "migrated"
            )
        migrated[key] = result
    return migrated
# ---------------------------------------------------------------------------
# Yahoo mapping
# ---------------------------------------------------------------------------
def _valid_yahoo_symbol(
    symbol: Any,
) -> bool:
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
) -> tuple[
    str | None,
    str | None,
]:
    """
    Search Yahoo for a Swedish equity.
    Returns:
        (yahoo_symbol, mapping_source)
    """
    normalized_ticker = (
        normalize_ticker(ticker)
    )
    # ---------------------------------------------------------------
    # 1. Direct ticker
    # ---------------------------------------------------------------
    if normalized_ticker:
        candidate = (
            f"{normalized_ticker}.ST"
        )
        try:
            search = yf.Search(
                candidate
            )
            quotes = (
                getattr(
                    search,
                    "quotes",
                    [],
                )
                or []
            )
            for quote in quotes:
                symbol = clean_value(
                    quote.get("symbol")
                )
                quote_type = str(
                    quote.get(
                        "quoteType"
                    )
                    or ""
                ).upper()
                if (
                    symbol
                    and symbol.upper()
                    == candidate.upper()
                    and (
                        not quote_type
                        or quote_type
                        == "EQUITY"
                    )
                ):
                    return (
                        symbol,
                        "ticker",
                    )
        except Exception:
            pass
    # ---------------------------------------------------------------
    # 2. Search issuer
    # ---------------------------------------------------------------
    queries: list[str] = []
    if issuer:
        queries.append(issuer)
    if normalized_ticker:
        queries.append(
            normalized_ticker
        )
    if isin:
        queries.append(isin)
    seen: set[str] = set()
    for query in queries:
        if (
            not query
            or query in seen
        ):
            continue
        seen.add(query)
        try:
            search = yf.Search(
                query
            )
            quotes = (
                getattr(
                    search,
                    "quotes",
                    [],
                )
                or []
            )
        except Exception:
            continue
        candidates = []
        for quote in quotes:
            symbol = clean_value(
                quote.get("symbol")
            )
            if not _valid_yahoo_symbol(
                symbol
            ):
                continue
            if not symbol.upper().endswith(
                ".ST"
            ):
                continue
            quote_type = str(
                quote.get(
                    "quoteType"
                )
                or ""
            ).upper()
            if (
                quote_type
                and quote_type != "EQUITY"
            ):
                continue
            quote_name = (
                quote.get("longname")
                or quote.get("shortname")
                or quote.get("name")
            )
            score = 0.0
            if (
                normalized_ticker
                and normalize_ticker(
                    symbol
                )
                == normalized_ticker
            ):
                score += 1.0
            if (
                issuer
                and quote_name
            ):
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
        if not candidates:
            continue
        score, symbol, _ = (
            candidates[0]
        )
        if (
            normalized_ticker
            and normalize_ticker(
                symbol
            )
            == normalized_ticker
        ):
            return (
                symbol,
                "ticker-search",
            )
        if (
            issuer
            and score >= 0.35
        ):
            return (
                symbol,
                "issuer-search",
            )
    return (
        None,
        None,
    )
# ---------------------------------------------------------------------------
# Build complete mapping
# ---------------------------------------------------------------------------
def build_instrument_map(
    *,
    existing: dict[str, dict[str, Any]] | None,
    fi_records: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    int,
    int,
]:
    """
    Build instrument mapping.
    Return value deliberately matches prices/__main__.py:
        mapping,
        new_mappings,
        unresolved
    """
    existing = existing or {}
    fi_instruments = (
        get_latest_fi_instruments(
            fi_records
        )
    )
    mapping = _migrate_existing_mapping(
        existing,
        fi_instruments,
    )
    unresolved_items = [
        item
        for item in mapping.values()
        if not _valid_yahoo_symbol(
            item.get(
                "yahoo_symbol"
            )
        )
    ]
    print(
        "Mappning: "
        f"{len(unresolved_items)} instrument "
        "behöver Yahoo-sökning."
    )
    new_mappings = 0
    unresolved = 0
    for index, item in enumerate(
        unresolved_items,
        start=1,
    ):
        issuer = item.get(
            "issuer"
        )
        ticker = item.get(
            "ticker"
        )
        isin = item.get(
            "isin"
        )
        symbol, source = (
            _search_yahoo(
                issuer=issuer,
                ticker=ticker,
                isin=isin,
            )
        )
        if symbol:
            item[
                "yahoo_symbol"
            ] = symbol
            item[
                "mapping_source"
            ] = source
            new_mappings += 1
            print(
                "Mappning: hittad - "
                f"{issuer} "
                f"ticker={ticker} "
                f"isin={isin} "
                f"-> {symbol} "
                f"[{index}/"
                f"{len(unresolved_items)}]"
            )
        else:
            item[
                "yahoo_symbol"
            ] = None
            item[
                "mapping_source"
            ] = None
            unresolved += 1
            print(
                "Mappning: ej hittad - "
                f"{issuer} "
                f"ticker={ticker} "
                f"isin={isin} "
                f"[{index}/"
                f"{len(unresolved_items)}]"
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
# Instruments used by price fetch
# ---------------------------------------------------------------------------
def get_yahoo_symbols(
    mapping: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Return only instruments with valid Yahoo symbols.
    Invalid values such as None/NONE are never passed to yfinance.
    """
    instruments: list[
        dict[str, Any]
    ] = []
    for map_key, item in mapping.items():
        symbol = clean_value(
            item.get(
                "yahoo_symbol"
            )
        )
        if not _valid_yahoo_symbol(
            symbol
        ):
            continue
        instruments.append(
            {
                "map_key": map_key,
                "isin": normalize_isin(
                    item.get("isin")
                ),
                "lei": clean_value(
                    item.get("lei")
                ),
                "issuer": clean_value(
                    item.get("issuer")
                ),
                "ticker": normalize_ticker(
                    item.get("ticker")
                ),
                "yahoo_symbol": symbol,
                "mapping_source": item.get(
                    "mapping_source"
                ),
            }
        )
    # Fetch each Yahoo symbol only once.
    unique: dict[
        str,
        dict[str, Any],
    ] = {}
    for instrument in instruments:
        symbol = instrument[
            "yahoo_symbol"
        ]
        if symbol not in unique:
            unique[symbol] = instrument
    return sorted(
        unique.values(),
        key=lambda item:
            item["yahoo_symbol"],
    )
