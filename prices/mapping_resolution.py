from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yfinance as yf

from prices.mapping_utils import (
    KNOWN_YAHOO_SYMBOLS,
    KNOWN_YAHOO_SYMBOLS_BY_ISIN,
    _persist_known_yahoo_mapping,
    clean_value,
    name_similarity,
    normalize_isin,
    normalize_name,
    normalize_ticker,
    yahoo_symbol,
)


HISTORICAL_SYMBOLS_PATH = Path(
    "prices/historical_symbols.jsonl"
)


# ---------------------------------------------------------------------------
# Verified historical Yahoo mappings
# ---------------------------------------------------------------------------
#
# Historical mappings are stored as data in historical_symbols.jsonl.
#
# The file is indexed by both ISIN and LEI. This is important because some
# historical FI records no longer contain the original ISIN, while the LEI
# can still identify the issuer.
#
# The mapping is deliberately conservative. An unresolved instrument is
# safer than a wrong historical price series.
# ---------------------------------------------------------------------------

def _load_historical_yahoo_symbols() -> tuple[
    dict[str, str],
    dict[str, str],
]:
    by_isin: dict[
        str,
        str,
    ] = {}

    by_lei: dict[
        str,
        str,
    ] = {}

    if not HISTORICAL_SYMBOLS_PATH.exists():
        return (
            by_isin,
            by_lei,
        )

    try:
        with HISTORICAL_SYMBOLS_PATH.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line_number, line in enumerate(
                handle,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(
                        line
                    )

                except json.JSONDecodeError as exc:
                    print(
                        "Mappning: ogiltig historisk "
                        "mapping på rad "
                        f"{line_number}: {exc}"
                    )
                    continue

                if not isinstance(
                    record,
                    dict,
                ):
                    continue

                symbol = clean_value(
                    record.get(
                        "yahoo_symbol"
                    )
                )

                if not symbol:
                    continue

                symbol = symbol.upper()

                isin = normalize_isin(
                    record.get("isin")
                )

                lei = clean_value(
                    record.get("lei")
                )

                if isin:
                    existing = by_isin.get(
                        isin
                    )

                    if (
                        existing is not None
                        and existing != symbol
                    ):
                        print(
                            "Mappning: konflikt i "
                            f"{HISTORICAL_SYMBOLS_PATH}: "
                            f"ISIN {isin} har både "
                            f"{existing} och "
                            f"{symbol}. "
                            "Den nya raden ignoreras."
                        )

                    else:
                        by_isin[isin] = symbol

                if lei:
                    existing = by_lei.get(
                        lei
                    )

                    if (
                        existing is not None
                        and existing != symbol
                    ):
                        print(
                            "Mappning: konflikt i "
                            f"{HISTORICAL_SYMBOLS_PATH}: "
                            f"LEI {lei} har både "
                            f"{existing} och "
                            f"{symbol}. "
                            "Den nya raden ignoreras."
                        )

                    else:
                        by_lei[lei] = symbol

    except OSError as exc:
        print(
            "Mappning: kunde inte läsa "
            f"{HISTORICAL_SYMBOLS_PATH}: {exc}"
        )

    return (
        by_isin,
        by_lei,
    )


(
    HISTORICAL_YAHOO_SYMBOLS_BY_ISIN,
    HISTORICAL_YAHOO_SYMBOLS_BY_LEI,
) = _load_historical_yahoo_symbols()


# ---------------------------------------------------------------------------
# Existing mapping migration
# ---------------------------------------------------------------------------

def migrate_existing_mapping(
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

        if isin:
            old = by_isin.get(
                isin
            )

        if old is None and lei:
            candidates = by_lei.get(
                lei,
                [],
            )

            if len(candidates) == 1:
                old = candidates[0]

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
                "isin": (
                    isin
                    or old.get("isin")
                ),
                "lei": (
                    lei
                    or old.get("lei")
                ),
                "issuer": (
                    issuer
                    or old.get("issuer")
                ),
                "ticker": (
                    instrument.get("ticker")
                    or old.get("ticker")
                ),
                "exchange": (
                    instrument.get("exchange")
                    or old.get("exchange")
                ),
            }

    return migrated


# ---------------------------------------------------------------------------
# Mapping status
# ---------------------------------------------------------------------------

def mapping_status(
    instrument: dict[str, Any],
) -> str:
    if any(
        (
            normalize_isin(
                instrument.get("isin")
            ),
            clean_value(
                instrument.get("lei")
            ),
            normalize_ticker(
                instrument.get("ticker")
            ),
        )
    ):
        return "identity_available"

    return "identity_incomplete"


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

def mapping_confidence(
    source: str | None,
) -> str | None:
    if source in {
        "known_isin",
        "known_name",
        "historical_isin",
        "historical_lei",
        "fi_ticker_exchange",
    }:
        return "high"

    if source in {
        "issuer_search",
        "normalized_name_search",
        "ticker_search",
        "isin_search",
    }:
        return "medium"

    return None


# ---------------------------------------------------------------------------
# Explicit mapping
# ---------------------------------------------------------------------------

def explicit_mapping(
    instrument: dict[str, Any],
) -> tuple[
    str | None,
    str | None,
]:
    isin = normalize_isin(
        instrument.get("isin")
    )

    if isin:
        symbol = (
            KNOWN_YAHOO_SYMBOLS_BY_ISIN.get(
                isin
            )
        )

        if symbol:
            return (
                symbol,
                "known_isin",
            )

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
# Historical mapping
# ---------------------------------------------------------------------------

def historical_mapping(
    instrument: dict[str, Any],
) -> tuple[
    str | None,
    str | None,
]:
    isin = normalize_isin(
        instrument.get("isin")
    )

    if isin:
        symbol = (
            HISTORICAL_YAHOO_SYMBOLS_BY_ISIN.get(
                isin
            )
        )

        if symbol:
            return (
                symbol,
                "historical_isin",
            )

    lei = clean_value(
        instrument.get("lei")
    )

    if lei:
        symbol = (
            HISTORICAL_YAHOO_SYMBOLS_BY_LEI.get(
                lei
            )
        )

        if symbol:
            return (
                symbol,
                "historical_lei",
            )

    return None, None


# ---------------------------------------------------------------------------
# FI ticker mapping
# ---------------------------------------------------------------------------

def ticker_mapping(
    instrument: dict[str, Any],
) -> tuple[
    str | None,
    str | None,
]:
    ticker = normalize_ticker(
        instrument.get("ticker")
    )

    exchange = instrument.get(
        "exchange"
    )

    if not ticker:
        return None, None

    symbol = yahoo_symbol(
        ticker,
        exchange,
    )

    if not symbol:
        return None, None

    return (
        symbol,
        "fi_ticker_exchange",
    )


# ---------------------------------------------------------------------------
# Yahoo search
# ---------------------------------------------------------------------------

def search_yahoo(
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


def candidate_symbol(
    quote: dict[str, Any],
) -> str | None:
    symbol = clean_value(
        quote.get("symbol")
    )

    if not symbol:
        return None

    symbol = symbol.upper()

    quote_type = clean_value(
        quote.get("quoteType")
    )

    if quote_type:
        quote_type = quote_type.upper()

        if quote_type not in {
            "EQUITY",
            "STOCK",
        }:
            return None

    return symbol


def candidate_exchange(
    quote: dict[str, Any],
) -> str | None:
    exchange = clean_value(
        quote.get("exchange")
    )

    if exchange:
        return exchange.upper()

    return None


def candidate_matches_exchange(
    quote: dict[str, Any],
    instrument: dict[str, Any],
) -> bool:
    requested = clean_value(
        instrument.get("exchange")
    )

    if not requested:
        return True

    requested = requested.upper()

    actual = candidate_exchange(
        quote
    )

    if not actual:
        return False

    aliases = {
        "STO": {
            "STO",
            "STOCKHOLM",
        },
        "STOCKHOLM": {
            "STO",
            "STOCKHOLM",
        },
        "CPH": {
            "CPH",
            "COPENHAGEN",
        },
        "COPENHAGEN": {
            "CPH",
            "COPENHAGEN",
        },
        "HEL": {
            "HEL",
            "HELSINKI",
        },
        "HELSINKI": {
            "HEL",
            "HELSINKI",
        },
        "LSE": {
            "LSE",
            "LONDON",
        },
        "LONDON": {
            "LSE",
            "LONDON",
        },
    }

    accepted = aliases.get(
        requested,
        {requested},
    )

    return actual in accepted


def search_candidates(
    instrument: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
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

    ticker = normalize_ticker(
        instrument.get("ticker")
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

    if ticker:
        queries.append(
            (
                ticker,
                "ticker_search",
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

    if diagnostics is not None:
        diagnostics["queries"] = []
        diagnostics["raw_candidates"] = 0
        diagnostics["invalid_quote_types"] = 0
        diagnostics["exchange_rejected"] = []
        diagnostics["accepted_candidates"] = []

    for query, source in queries:
        quotes = search_yahoo(
            query
        )

        if diagnostics is not None:
            diagnostics["queries"].append(
                {
                    "query": query,
                    "source": source,
                    "result_count": len(quotes),
                }
            )

        for quote in quotes:
            symbol = candidate_symbol(
                quote
            )

            if not symbol:
                if diagnostics is not None:
                    diagnostics[
                        "invalid_quote_types"
                    ] += 1

                continue

            if diagnostics is not None:
                diagnostics[
                    "raw_candidates"
                ] += 1

            actual_exchange = (
                candidate_exchange(
                    quote
                )
            )

            if not candidate_matches_exchange(
                quote,
                instrument,
            ):
                if diagnostics is not None:
                    diagnostics[
                        "exchange_rejected"
                    ].append(
                        {
                            "symbol": symbol,
                            "exchange": actual_exchange,
                            "source": source,
                        }
                    )

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

            issuer_norm = normalize_name(
                issuer
            )

            quote_norm = normalize_name(
                quote_name
            )

            if (
                issuer_norm
                and quote_norm
                and issuer_norm == quote_norm
            ):
                score = 1.0

            score += 0.10

            quote_symbol = normalize_ticker(
                symbol
            )

            if (
                ticker
                and quote_symbol
                and (
                    quote_symbol == ticker
                    or quote_symbol.startswith(
                        ticker + "."
                    )
                )
            ):
                score += 0.15

            score = min(
                score,
                1.0,
            )

            candidates.append(
                (
                    symbol,
                    score,
                    source,
                )
            )

            if diagnostics is not None:
                diagnostics[
                    "accepted_candidates"
                ].append(
                    {
                        "symbol": symbol,
                        "score": score,
                        "source": source,
                        "exchange": actual_exchange,
                        "name": quote_name,
                    }
                )

    candidates.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return candidates


def search_mapping(
    instrument: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> tuple[
    str | None,
    str | None,
]:
    candidates = search_candidates(
        instrument,
        diagnostics,
    )

    if not candidates:
        if diagnostics is not None:
            raw_candidates = diagnostics.get(
                "raw_candidates",
                0,
            )

            invalid_quote_types = diagnostics.get(
                "invalid_quote_types",
                0,
            )

            exchange_rejected = diagnostics.get(
                "exchange_rejected",
                [],
            )

            if (
                raw_candidates == 0
                and invalid_quote_types > 0
            ):
                diagnostics["reason"] = (
                    "no_equity_candidates"
                )

            elif (
                raw_candidates > 0
                and exchange_rejected
            ):
                diagnostics["reason"] = (
                    "all_wrong_exchange"
                )

            else:
                diagnostics["reason"] = (
                    "no_candidates"
                )

        return None, None

    best_symbol, best_score, source = (
        candidates[0]
    )

    if best_score >= 0.95:
        if diagnostics is not None:
            diagnostics["reason"] = (
                "accepted_high_score"
            )

        return (
            best_symbol,
            source,
        )

    if best_score >= 0.85:
        if (
            len(candidates) == 1
            or (
                best_score
                - candidates[1][1]
                >= 0.10
            )
        ):
            if diagnostics is not None:
                diagnostics["reason"] = (
                    "accepted_clear_leader"
                )

            return (
                best_symbol,
                source,
            )

        if diagnostics is not None:
            diagnostics["reason"] = (
                "multiple_close_candidates"
            )

        return None, None

    if diagnostics is not None:
        diagnostics["reason"] = (
            "best_score_below_threshold"
        )

    return None, None


# ---------------------------------------------------------------------------
# Unified resolution
# ---------------------------------------------------------------------------

def resolve_instrument(
    instrument: dict[str, Any],
) -> tuple[
    str | None,
    str | None,
    dict[str, Any],
]:
    # 1. Exact known mappings.
    symbol, source = explicit_mapping(
        instrument
    )

    if symbol:
        return (
            symbol,
            source,
            {},
        )

    # 2. Verified historical mappings.
    symbol, source = historical_mapping(
        instrument
    )

    if symbol:
        _persist_known_yahoo_mapping(
            instrument.get("issuer"),
            symbol,
        )

        return (
            symbol,
            source,
            {},
        )

    # 3. Direct FI ticker + exchange.
    symbol, source = ticker_mapping(
        instrument
    )

    if symbol:
        _persist_known_yahoo_mapping(
            instrument.get("issuer"),
            symbol,
        )

        return (
            symbol,
            source,
            {},
        )

    # 4. Yahoo search.
    diagnostics: dict[str, Any] = {}

    symbol, source = search_mapping(
        instrument,
        diagnostics,
    )

    if symbol:
        _persist_known_yahoo_mapping(
            instrument.get("issuer"),
            symbol,
        )

        return (
            symbol,
            source,
            diagnostics,
        )

    return (
        None,
        None,
        diagnostics,
    )


__all__ = [
    "HISTORICAL_YAHOO_SYMBOLS_BY_ISIN",
    "HISTORICAL_YAHOO_SYMBOLS_BY_LEI",
    "mapping_confidence",
    "mapping_status",
    "migrate_existing_mapping",
    "resolve_instrument",
]
