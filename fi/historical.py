"""Historisk aggregerad FI-data."""

from __future__ import annotations

from datetime import date

from .errors import FIError


def find_historical_source() -> str:
    """
    Hitta FI:s historiska aggregerade datakälla.

    FI:s aktuella webbplats visar att "Aggregerade
    positioner" är en separat datamängd. Den historiska
    endpointen är dock inte exponerad som en vanlig
    href i den HTML som klienten får.

    Vi ska därför inte gissa en endpoint här.
    """

    raise FIError(
        "FI:s historiska aggregerade datakälla kunde "
        "inte verifieras automatiskt. "
        "Backfill avbryts i stället för att använda "
        "fel datamängd."
    )


def fetch_historical(
    start_date: date,
    end_date: date,
) -> list[dict]:

    if start_date > end_date:
        raise FIError(
            "Startdatum är senare än slutdatum."
        )

    source = find_historical_source()

    # Medvetet oanvänd tills den riktiga FI-källan
    # har verifierats.
    raise FIError(
        "Historisk FI-källa är ännu inte implementerad: "
        f"{source}"
    )
