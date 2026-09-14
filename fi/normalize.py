"""Normalisering av Finansinspektionens blankningsdata."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd


STOCKHOLM = ZoneInfo(
    "Europe/Stockholm"
)


def now_stockholm() -> datetime:
    """Aktuell tid i Europe/Stockholm."""

    return datetime.now(
        STOCKHOLM
    )


def fetched_at() -> str:
    """
    ISO-tidsstämpel för när FI-data hämtades.
    """

    return now_stockholm().isoformat(
        timespec="seconds"
    )


def normalize_text(
    value: object,
) -> str | None:
    """Normaliserar text från FI-tabellen."""

    if value is None:
        return None

    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def normalize_percent(
    value: object,
) -> float | None:
    """
    Normaliserar procentvärden från FI.

    Exempel:

        '7,10 %' -> 7.10
        '0,49 %' -> 0.49
        7.1      -> 7.1

    Värdet representerar procentenheter,
    inte decimalform.
    """

    if value is None:
        return None

    if pd.isna(value):
        return None

    if isinstance(
        value,
        (int, float),
    ):
        result = float(value)
    else:
        text = str(value).strip()

        if not text:
            return None

        text = (
            text.replace("%", "")
            .replace("\xa0", "")
            .replace(" ", "")
            .replace(",", ".")
        )

        try:
            result = float(text)
        except ValueError:
            return None

    if result < 0 or result > 100:
        raise ValueError(
            f"Orimligt procentvärde från FI: {result}"
        )

    return round(
        result,
        6,
    )


def normalize_date(
    value: object,
) -> str | None:
    """Normaliserar ett datum till YYYY-MM-DD."""

    if value is None:
        return None

    if pd.isna(value):
        return None

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.date().isoformat()

    text = str(value).strip()

    if not text:
        return None

    parsed = pd.to_datetime(
        text,
        dayfirst=True,
        errors="coerce",
    )

    if pd.isna(parsed):
        return None

    return parsed.date().isoformat()


def normalize_column_name(
    value: object,
) -> str:
    """Normaliserar ett kolumnnamn för jämförelser."""

    text = normalize_text(value)

    if text is None:
        return ""

    text = text.lower()

    text = (
        text.replace("å", "a")
        .replace("ä", "a")
        .replace("ö", "o")
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def resolve_column(
    columns: Iterable[object],
    candidates: Iterable[str],
) -> object | None:
    """
    Hittar en kolumn utifrån flera möjliga namn.
    """

    normalized = {
        normalize_column_name(column): column
        for column in columns
    }

    for candidate in candidates:
        key = normalize_column_name(
            candidate
        )

        if key in normalized:
            return normalized[key]

    return None


def normalize_records(
    table: pd.DataFrame,
    fetched_at_value: str,
    source_date: str,
) -> list[dict]:
    """
    Gör FI-tabellen till Blankdiss standardformat.
    """

    issuer_column = resolve_column(
        table.columns,
        [
            "Emittentens namn",
            "Emittent",
        ],
    )

    lei_column = resolve_column(
        table.columns,
        [
            "Emittentens LEI-kod",
            "LEI-kod",
            "LEI",
        ],
    )

    position_date_column = resolve_column(
        table.columns,
        [
            "Positionsdatum senaste position",
            "Positionsdatum",
        ],
    )

    short_interest_column = resolve_column(
        table.columns,
        [
            "Summa blankning %",
            "Summa blankning",
        ],
    )

    if not issuer_column:
        raise ValueError(
            "Kolumnen för emittent saknas."
        )

    if not lei_column:
        raise ValueError(
            "Kolumnen för LEI saknas."
        )

    if not position_date_column:
        raise ValueError(
            "Kolumnen för positionsdatum saknas."
        )

    if not short_interest_column:
        raise ValueError(
            "Kolumnen för summa blankning saknas."
        )

    records: list[dict] = []

    for _, row in table.iterrows():
        issuer = normalize_text(
            row[issuer_column]
        )

        lei = normalize_text(
            row[lei_column]
        )

        position_date = normalize_date(
            row[position_date_column]
        )

        short_interest = normalize_percent(
            row[short_interest_column]
        )

        if not issuer:
            continue

        if not lei:
            continue

        if short_interest is None:
            continue

        records.append(
            {
                "fetched_at": fetched_at_value,
                "source_date": source_date,
                "position_date": position_date,
                "lei": lei,
                "issuer": issuer,
                "short_interest_pct": short_interest,
            }
        )

    records.sort(
        key=lambda record: (
            record["issuer"].lower(),
            record["lei"],
        )
    )

    return records
