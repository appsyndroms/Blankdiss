# fi/normalize.py

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


STOCKHOLM_TZ = ZoneInfo("Europe/Stockholm")


def now_stockholm() -> datetime:
    """
    Returnerar aktuell tid i Europe/Stockholm.
    """

    return datetime.now(
        STOCKHOLM_TZ
    )


def fetched_at() -> str:
    """
    Returnerar aktuell tid i Europe/Stockholm
    som ISO-sträng.
    """

    return now_stockholm().isoformat(
        timespec="seconds"
    )


def normalize_percent(
    value: object,
) -> float | None:
    if value is None:
        return None

    if pd.isna(value):
        return None

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

    if not 0 <= result <= 100:
        return None

    return result


def normalize_date(
    value: object,
) -> str | None:
    """
    Normaliserar ett datum till YYYY-MM-DD.

    FI levererar dessa datum som ISO-format:
    YYYY-MM-DD.
    """

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
        format="%Y-%m-%d",
        errors="coerce",
    )

    if pd.isna(parsed):
        return None

    return parsed.date().isoformat()


def normalize_lei(
    value: object,
) -> str | None:
    if value is None:
        return None

    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def normalize_issuer(
    value: object,
) -> str | None:
    if value is None:
        return None

    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def normalize_records(
    dataframe: pd.DataFrame,
    fetched_at: str,
    source_date: str,
) -> list[dict[str, Any]]:
    columns = {
        str(column).strip(): column
        for column in dataframe.columns
    }

    issuer_column = next(
        (
            columns[name]
            for name in (
                "Emittentens namn",
                "Emittent",
            )
            if name in columns
        ),
        None,
    )

    lei_column = next(
        (
            columns[name]
            for name in (
                "Emittentens LEI-kod",
                "LEI-kod",
                "LEI",
            )
            if name in columns
        ),
        None,
    )

    position_date_column = next(
        (
            columns[name]
            for name in (
                "Positionsdatum senaste position",
                "Positionsdatum",
            )
            if name in columns
        ),
        None,
    )

    short_interest_column = next(
        (
            columns[name]
            for name in (
                "Summa blankning %",
                "Summa blankning",
            )
            if name in columns
        ),
        None,
    )

    if issuer_column is None:
        raise ValueError(
            "Kunde inte hitta kolumn för emittent."
        )

    if position_date_column is None:
        raise ValueError(
            "Kunde inte hitta kolumn för positionsdatum."
        )

    if short_interest_column is None:
        raise ValueError(
            "Kunde inte hitta kolumn för summa blankning."
        )

    records: list[dict[str, Any]] = []

    for _, row in dataframe.iterrows():
        issuer = normalize_issuer(
            row[issuer_column]
        )

        position_date = normalize_date(
            row[position_date_column]
        )

        short_interest_pct = normalize_percent(
            row[short_interest_column]
        )

        lei = (
            normalize_lei(row[lei_column])
            if lei_column is not None
            else None
        )

        if issuer is None:
            continue

        if position_date is None:
            continue

        if short_interest_pct is None:
            continue

        records.append(
            {
                "fetched_at": fetched_at,
                "source_date": source_date,
                "position_date": position_date,
                "lei": lei,
                "issuer": issuer,
                "short_interest_pct": short_interest_pct,
            }
        )

    return records
