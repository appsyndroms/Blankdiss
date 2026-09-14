"""Hämtning och normalisering av aktuell FI-data."""
from __future__ import annotations
from io import StringIO
import pandas as pd
from .client import fetch_html
from .errors import FIError
from .normalize import (
    normalize_records,
    now_stockholm,
)
def find_current_table(
    tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Hittar FI:s aktuella aggregerade blankningstabell.
    """
    required_columns = {
        "emittent",
        "lei",
        "positionsdatum",
        "summa blankning",
    }
    for table in tables:
        if table.empty:
            continue
        normalized_columns = set()
        for column in table.columns:
            text = str(column).strip().lower()
            text = (
                text.replace("å", "a")
                .replace("ä", "a")
                .replace("ö", "o")
            )
            normalized_columns.add(
                text
            )
        has_issuer = any(
            "emittentens namn" in column
            or column == "emittent"
            for column in normalized_columns
        )
        has_lei = any(
            "emittentens lei" in column
            or column == "lei"
            or "lei-kod" in column
            for column in normalized_columns
        )
        has_position_date = any(
            "positionsdatum" in column
            for column in normalized_columns
        )
        has_short_interest = any(
            "summa blankning" in column
            for column in normalized_columns
        )
        if (
            has_issuer
            and has_lei
            and has_position_date
            and has_short_interest
        ):
            return table
    raise FIError(
        "Kunde inte hitta FI:s aktuella "
        "aggregerade blankningstabell."
    )
def fetch_current() -> list[dict]:
    """
    Hämtar aktuell aggregerad blankning från FI.
    """
    fetched_at_value = (
        now_stockholm()
        .isoformat()
    )
    html = fetch_html()
    try:
        tables = pd.read_html(
            StringIO(html),
            decimal=",",
            thousands=" ",
        )
    except Exception as exc:
        raise FIError(
            "Kunde inte tolka FI:s HTML-tabeller."
        ) from exc
    if not tables:
        raise FIError(
            "FI-sidan innehöll inga tabeller."
        )
    table = find_current_table(
        tables
    )
    source_date = (
        fetched_at_value[:10]
    )
    try:
        records = normalize_records(
            table,
            fetched_at_value,
            source_date,
        )
    except ValueError as exc:
        raise FIError(
            f"Kunde inte normalisera FI-data: {exc}"
        ) from exc
    if not records:
        raise FIError(
            "FI-tabellen innehöll inga "
            "giltiga observationer."
        )
    return records
