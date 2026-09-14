"""Normalisering av FI-data."""

from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

from .config import FI_URL, STOCKHOLM


def now_stockholm() -> datetime:
    return datetime.now(STOCKHOLM)


def fetched_at() -> str:
    """Tidpunkt då FI-data hämtades."""

    return now_stockholm().isoformat(
        timespec="seconds"
    )


def normalize_text(value) -> str:
    if value is None:
        return ""

    return str(value).replace(
        "\xa0",
        " ",
    ).strip()


def normalize_percent(
    value,
) -> float | None:
    """
    Normaliserar svensk procentnotation.

    Exempel:

        7,1   -> 7.1
        0,49  -> 0.49
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    text = normalize_text(value)

    if not text:
        return None

    text = (
        text
        .replace("%", "")
        .replace("\xa0", "")
        .replace(" ", "")
        .replace(",", ".")
    )

    text = re.sub(
        r"[^0-9.\-]",
        "",
        text,
    )

    if not text:
        return None

    try:
        result = float(text)
    except ValueError:
        return None

    if result < 0 or result > 100:
        return None

    return result


def normalize_date(
    value,
) -> str | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    text = normalize_text(value)

    if not text:
        return None

    parsed = pd.to_datetime(
        text,
        errors="coerce",
        dayfirst=False,
    )

    if pd.isna(parsed):
        parsed = pd.to_datetime(
            text,
            errors="coerce",
            dayfirst=True,
        )

    if pd.isna(parsed):
        return None

    return parsed.strftime("%Y-%m-%d")


def resolve_column(
    columns,
    patterns: list[str],
) -> str | None:

    normalized = {
        column: normalize_text(column).lower()
        for column in columns
    }

    for pattern in patterns:
        for original, value in normalized.items():
            if pattern in value:
                return original

    return None


def normalize_records(
    table: pd.DataFrame,
    fetched_at_value: str,
    source_date: str,
) -> list[dict]:

    if table.empty:
        return []

    issuer_column = resolve_column(
        table.columns,
        [
            "emittentens namn",
            "bolagsnamn",
            "emittent",
            "issuer",
        ],
    )

    lei_column = resolve_column(
        table.columns,
        [
            "emittentens lei-kod",
            "lei-kod",
            "lei",
        ],
    )

    position_date_column = resolve_column(
        table.columns,
        [
            "positionsdatum senaste position",
            "positionsdatum",
            "datum",
            "date",
            "snapshot",
        ],
    )

    short_interest_column = resolve_column(
        table.columns,
        [
            "summa blankning %",
            "summa blankning",
            "aggregerad",
            "aggregate",
        ],
    )

    if not issuer_column:
        return []

    if not short_interest_column:
        return []

    records: list[dict] = []

    for _, row in table.iterrows():

        issuer = normalize_text(
            row.get(issuer_column)
        )

        if not issuer:
            continue

        lei = ""

        if lei_column:
            lei = normalize_text(
                row.get(lei_column)
            )

        position_date = source_date

        if position_date_column:
            candidate = normalize_date(
                row.get(position_date_column)
            )

            if candidate:
                position_date = candidate

        short_interest_pct = normalize_percent(
            row.get(short_interest_column)
        )

        if short_interest_pct is None:
            continue

        records.append(
            {
                "fetched_at": fetched_at_value,
                "source_date": source_date,
                "position_date": position_date,
                "lei": lei,
                "issuer": issuer,
                "short_interest_pct": short_interest_pct,
                "source": FI_URL,
            }
        )

    return records
