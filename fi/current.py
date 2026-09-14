"""Hämtning och normalisering av aktuell FI-data."""

from __future__ import annotations

from io import StringIO

import pandas as pd

from .client import fetch_html
from .errors import FIError
from .normalize import (
    fetched_at,
    normalize_records,
    normalize_text,
    resolve_column,
)


def find_current_table(
    tables: list[pd.DataFrame],
) -> pd.DataFrame:

    for table in tables:

        if table.empty:
            continue

        issuer_column = resolve_column(
            table.columns,
            [
                "emittentens namn",
                "emittent",
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
            ],
        )

        short_interest_column = resolve_column(
            table.columns,
            [
                "summa blankning %",
                "summa blankning",
            ],
        )

        if (
            issuer_column
            and lei_column
            and position_date_column
            and short_interest_column
        ):
            return table

    raise FIError(
        "Kunde inte hitta FI:s blankningstabell."
    )


def fetch_current() -> list[dict]:

    fetched_at_value = fetched_at()

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

    table = find_current_table(tables)

    source_date = fetched_at_value[:10]

    return normalize_records(
        table,
        fetched_at_value,
        source_date,
    )
