from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from prices.mapping_utils import (
    FI_RECONSTRUCTED_PATH,
    FI_SNAPSHOT_DIR,
    clean_value,
    normalize_isin,
    normalize_name,
    normalize_ticker,
)


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
            result = clean_value(
                value
            )

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


def get_exchange(
    record: dict[str, Any],
) -> str | None:
    """
    Extract exchange/market information when FI data contains it.

    FI schemas have changed over time, so several common field names
    are supported.
    """

    return _find_nested_value(
        record,
        {
            "exchange",
            "Exchange",
            "exchange_code",
            "exchangeCode",
            "market",
            "Market",
            "market_code",
            "marketCode",
            "trading_venue",
            "tradingVenue",
            "trading_venue_code",
            "tradingVenueCode",
            "venue",
            "Venue",
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
        exchange = get_exchange(record)
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
            "exchange": exchange,
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


__all__ = [
    "get_date",
    "get_exchange",
    "get_isin",
    "get_issuer",
    "get_latest_fi_instruments",
    "get_lei",
    "get_ticker",
    "read_all_fi_data",
]
