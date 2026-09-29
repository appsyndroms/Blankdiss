from __future__ import annotations

import json
from typing import Any

import yfinance as yf

from prices.fi_identity import (
    get_latest_fi_instruments,
    read_all_fi_data,
)
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
    MAPPING_PATH,
)


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
) -> Any:
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
                "isin": isin
                or old.get("isin"),
                "lei": lei
                or old.get("lei"),
                "issuer": issuer
                or old.get("issuer"),
                "ticker": (
                    instrument.get(
                        "ticker"
                    )
                    or old.get(
                        "ticker"
                    )
                ),
                "exchange": (
                    instrument.get(
                        "exchange"
                    )
                    or old.get(
                        "exchange"
                    )
                ),
            }

    return migrated


# ---------------------------------------------------------------------------
# Mapping diagnostics
# ---------------------------------------------------------------------------

def _mapping_status(
    instrument: dict[str, Any],
) -> str:
    """
    Explain whether an unresolved instrument has usable identity data.

    This does not infer whether an instrument is currently listed or
    delisted. Listing history is a separate concern.
    """

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


def _mapping_confidence(
    source: str | None,
) -> str | None:
    if source in {
        "known_name",
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

def _explicit_mapping(
    instrument: dict[str, Any],
) -> tuple[
    str | None,
    str | None,
]:
    isin = normalize_isin(
        instrument.get("isin")
    )

    if isin:
        symbol = KNOWN_YAHOO_SYMBOLS_BY_ISIN.get(
            isin
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


def _candidate_exchange(
    quote: dict[str, Any],
) -> str | None:
    exchange = clean_value(
        quote.get("exchange")
    )

    if exchange:
        return exchange.upper()

    return None


def _candidate_matches_exchange(
    quote: dict[str, Any],
    instrument: dict[str, Any],
) -> bool:
    requested = clean_value(
        instrument.get("exchange")
    )

    if not requested:
        return True

    requested = requested.upper()

    actual = _candidate_exchange(
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


def _search_candidates(
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
        diagnostics["exchange_rejected"] = []
        diagnostics["accepted_candidates"] = []

    for query, source in queries:
        quotes = _search_yahoo(
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
            symbol = _candidate_symbol(
                quote
            )

            if not symbol:
                continue

            actual_exchange = (
                _candidate_exchange(
                    quote
                )
            )

            if not _candidate_matches_exchange(
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
                and (
                    issuer_norm
                    == quote_norm
                )
            ):
                score = 1.0

            # Exchange agreement is now a prerequisite, not just
            # a small score bonus.
            score += 0.10

            # Ticker agreement is a strong signal.
            quote_symbol = normalize_ticker(
                symbol
            )

            if (
                ticker
                and quote_symbol
                and (
                    quote_symbol
                    == ticker
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

            candidate = (
                symbol,
                score,
                source,
            )

            candidates.append(
                candidate
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


def _search_mapping(
    instrument: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> tuple[
    str | None,
    str | None,
]:
    candidates = _search_candidates(
        instrument,
        diagnostics,
    )

    if not candidates:
        if diagnostics is not None:
            diagnostics["reason"] = (
                "no_accepted_candidates"
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


def _print_mapping_diagnostic(
    instrument: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    issuer = (
        instrument.get("issuer")
        or "?"
    )

    isin = instrument.get(
        "isin"
    )

    lei = instrument.get(
        "lei"
    )

    ticker = instrument.get(
        "ticker"
    )

    exchange = instrument.get(
        "exchange"
    )

    print(
        "Mappning diagnostik: "
        f"{issuer}"
    )

    print(
        "  identitet: "
        f"isin={isin} "
        f"lei={lei} "
        f"ticker={ticker} "
        f"exchange={exchange}"
    )

    for query in diagnostics.get(
        "queries",
        [],
    ):
        print(
            "  sökning: "
            f"{query['source']} "
            f"{query['query']!r} "
            f"-> "
            f"{query['result_count']} "
            "Yahoo-resultat"
        )

    rejected = diagnostics.get(
        "exchange_rejected",
        [],
    )

    if rejected:
        print(
            "  fel börs: "
            + ", ".join(
                (
                    f"{item['symbol']} "
                    f"({item['exchange']})"
                )
                for item in rejected
            )
        )

    accepted = diagnostics.get(
        "accepted_candidates",
        [],
    )

    if accepted:
        print(
            "  accepterade kandidater: "
            + ", ".join(
                (
                    f"{item['symbol']} "
                    f"score={item['score']:.3f}"
                )
                for item in accepted
            )
        )

    print(
        "  diagnos: "
        f"{diagnostics.get('reason')}"
    )


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

    needs_mapping = [
        instrument
        for instrument in fi_instruments
        if (
            instrument["map_key"] not in mapping
            or not clean_value(
                mapping[
                    instrument["map_key"]
                ].get(
                    "yahoo_symbol"
                )
            )
        )
    ]

    print(
        "Mappning: "
        f"{len(needs_mapping)} instrument "
        "behöver Yahoo-mappning."
    )

    diagnostic_counts = {
        "no_accepted_candidates": 0,
        "best_score_below_threshold": 0,
        "multiple_close_candidates": 0,
        "exchange_only": 0,
        "other": 0,
    }

    for index, instrument in enumerate(
        needs_mapping,
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

        exchange = instrument.get(
            "exchange"
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
        # 2. FI ticker + exchange
        #
        # Never blindly assume .ST.
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

        diagnostics: dict[str, Any] = {}

        if symbol is None:
            symbol, source = (
                _search_mapping(
                    instrument,
                    diagnostics,
                )
            )

        if symbol:
            _persist_known_yahoo_mapping(
                instrument.get(
                    "issuer"
                ),
                symbol,
            )

            mapping[key] = {
                "map_key": key,
                "isin": isin,
                "lei": instrument.get(
                    "lei"
                ),
                "issuer": issuer,
                "ticker": ticker,
                "exchange": exchange,
                "yahoo_symbol": symbol,
                "mapping_source": source,
                "mapping_confidence": _mapping_confidence(
                    source
                ),
                "mapping_status": "mapped",
            }

            new_mappings += 1

            print(
                "Mappning: hittad - "
                f"{issuer} "
                f"ticker={ticker} "
                f"exchange={exchange} "
                f"isin={isin} "
                f"-> {symbol} "
                f"[{index}/"
                f"{len(needs_mapping)}]"
            )

        else:
            old = mapping.get(
                key,
                {}
            )

            mapping[key] = {
                **old,
                "map_key": key,
                "isin": isin,
                "lei": instrument.get(
                    "lei"
                ),
                "issuer": issuer,
                "ticker": ticker,
                "exchange": exchange,
                "yahoo_symbol": None,
                "mapping_source": None,
                "mapping_confidence": None,
                "mapping_status": _mapping_status(
                    instrument
                ),
            }

            unresolved += 1

            print(
                "Mappning: ej hittad - "
                f"{issuer} "
                f"ticker={ticker} "
                f"exchange={exchange} "
                f"isin={isin} "
                f"[{index}/"
                f"{len(needs_mapping)}]"
            )

            if diagnostics:
                _print_mapping_diagnostic(
                    instrument,
                    diagnostics,
                )

                reason = diagnostics.get(
                    "reason"
                )

                if reason in diagnostic_counts:
                    diagnostic_counts[
                        reason
                    ] += 1
                else:
                    diagnostic_counts[
                        "other"
                    ] += 1

                if (
                    diagnostics.get(
                        "exchange_rejected"
                    )
                    and not diagnostics.get(
                        "accepted_candidates"
                    )
                ):
                    diagnostic_counts[
                        "exchange_only"
                    ] += 1

    print(
        "Mappning diagnostik:"
    )

    print(
        "  inga accepterade kandidater: "
        f"{diagnostic_counts['no_accepted_candidates']}"
    )

    print(
        "  bästa score under tröskel: "
        f"{diagnostic_counts['best_score_below_threshold']}"
    )

    print(
        "  flera nära kandidater: "
        f"{diagnostic_counts['multiple_close_candidates']}"
    )

    print(
        "  endast kandidater från fel börs: "
        f"{diagnostic_counts['exchange_only']}"
    )

    print(
        "  övrigt: "
        f"{diagnostic_counts['other']}"
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


# ---------------------------------------------------------------------------
# Compatibility exports
# ---------------------------------------------------------------------------
#
# These imports deliberately remain available through prices.mapping.
# Existing callers therefore do not need to be changed just because
# the implementation has been split into smaller modules.
# ---------------------------------------------------------------------------

__all__ = [
    "build_instrument_map",
    "get_latest_fi_instruments",
    "get_yahoo_symbols",
    "load_instrument_map",
    "name_similarity",
    "normalize_isin",
    "normalize_name",
    "normalize_ticker",
    "read_all_fi_data",
    "save_instrument_map",
    "yahoo_symbol",
]
