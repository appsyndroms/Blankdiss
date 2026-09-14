"""Normalisering av Finansinspektionens blankningsdata."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

from zoneinfo import ZoneInfo

import pandas as pd


STOCKHOLM = ZoneInfo("Europe/Stockholm")


def now_stockholm() -> datetime:
    """Returnerar aktuell tid i svensk tidszon."""
    return datetime.now(STOCKHOLM)


def normalize_column_name(
    value: Any,
) -> str:
    """Normaliserar ett kolumnnamn för robust matchning."""
    if value is None:
        return ""

    text = str(value)

    if text.lower() == "nan":
        return ""

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip().lower()


def find_column(
    columns: list[Any],
    candidates: list[str],
) -> Any | None:
    """
    Hittar en kolumn genom normaliserade
    namn och delsträngar.
    """
    normalized = [
        (
            column,
            normalize_column_name(column),
        )
        for column in columns
    ]

    candidate_names = [
        normalize_column_name(candidate)
        for candidate in candidates
    ]

    # Först: exakt matchning.
    for candidate in candidate_names:
        for column, normalized_name in normalized:
            if normalized_name == candidate:
                return column

    # Därefter: delsträngsmatchning.
    for candidate in candidate_names:
        for column, normalized_name in normalized:
            if candidate in normalized_name:
                return column

    return None


def find_issuer_column(
    columns: list[Any],
) -> Any | None:
    """Hittar emittentkolumnen."""
    return find_column(
        columns,
        [
            "Emittentens namn",
            "Namn på emittent",
            "Issuer",
            "Name of the issuer",
            "Emittent",
        ],
    )


def find_lei_column(
    columns: list[Any],
) -> Any | None:
    """Hittar LEI-kolumnen."""
    return find_column(
        columns,
        [
            "Emittentens LEI-kod",
            "LEI-kod",
            "LEI",
            "Issuer LEI",
        ],
    )


def find_position_date_column(
    columns: list[Any],
) -> Any | None:
    """Hittar positionsdatumkolumnen."""
    return find_column(
        columns,
        [
            "Positionsdatum senaste position",
            "Positionsdatum",
            "Position date",
            "Date of position",
        ],
    )


def find_short_interest_column(
    columns: list[Any],
) -> Any | None:
    """Hittar kolumnen för aggregerad blankning."""
    return find_column(
        columns,
        [
            "Summa blankning %",
            "Summa blankning",
            "Aggregerad blankningsposition i procent",
            "Aggregerad blankningsosition i procent",
            "Aggregate net short position in per cent",
            "Aggregate net short position in percent",
            "Aggregate net short position",
            "Short interest",
        ],
    )


def clean_text(
    value: Any,
) -> str | None:
    """Rensar textvärden."""
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return None

    return text


def clean_lei(
    value: Any,
) -> str | None:
    """Normaliserar LEI."""
    text = clean_text(value)

    if text is None:
        return None

    return text.upper()


def parse_percent(
    value: Any,
) -> float | None:
    """Tolkar blankningsprocent."""
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    if not text:
        return None

    text = text.replace("%", "")
    text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return None


def parse_date(
    value: Any,
) -> str | None:
    """Tolkar datum till ISO-format."""
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()

    if hasattr(value, "date") and not isinstance(
        value,
        str,
    ):
        try:
            return value.date().isoformat()
        except (AttributeError, ValueError):
            pass

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return None

    parsed = pd.to_datetime(
        text,
        errors="coerce",
        dayfirst=True,
    )

    if pd.isna(parsed):
        return None

    return parsed.date().isoformat()


def normalize_records(
    table: pd.DataFrame,
    fetched_at: str,
    source_date: str | None = None,
) -> list[dict]:
    """
    Normaliserar en rå FI-tabell till Blankdiss-format.

    Stöder både moderna svenska rubriker
    och äldre historiska FI-filer med
    engelska rubriker inom parentes.
    """
    if table.empty:
        return []

    columns = list(table.columns)

    issuer_column = find_issuer_column(
        columns
    )

    if issuer_column is None:
        raise ValueError(
            "Kunde inte hitta kolumn "
            "för emittent."
        )

    lei_column = find_lei_column(
        columns
    )

    position_date_column = (
        find_position_date_column(
            columns
        )
    )

    short_interest_column = (
        find_short_interest_column(
            columns
        )
    )

    if short_interest_column is None:
        raise ValueError(
            "Kunde inte hitta kolumn "
            "för summa/aggregerad "
            "blankning."
        )

    records: list[dict] = []

    for _, row in table.iterrows():
        issuer = clean_text(
            row.get(issuer_column)
        )

        if issuer is None:
            continue

        short_interest_pct = parse_percent(
            row.get(
                short_interest_column
            )
        )

        if short_interest_pct is None:
            continue

        if lei_column is not None:
            lei = clean_lei(
                row.get(lei_column)
            )
        else:
            lei = None

        if position_date_column is not None:
            position_date = parse_date(
                row.get(
                    position_date_column
                )
            )
        else:
            position_date = source_date

        if position_date is None:
            position_date = source_date

        records.append(
            {
                "fetched_at": fetched_at,
                "source_date": source_date,
                "position_date": position_date,
                "lei": lei,
                "issuer": issuer,
                "short_interest_pct": (
                    short_interest_pct
                ),
            }
        )

    return records
