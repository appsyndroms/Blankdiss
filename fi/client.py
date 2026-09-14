"""Hämtning och normalisering av FI:s aggregatfil."""

from __future__ import annotations

from io import BytesIO

import pandas as pd

from .client import fetch_aggregate
from .errors import FIError
from .normalize import (
    fetched_at,
    normalize_records,
)


def read_aggregate_file(
    data: bytes,
) -> pd.DataFrame:
    """
    Läser FI:s aggregerade ODS-fil.

    FI:s aggregatfil har sex metadata-/rubrikrader
    före själva tabellen.
    """

    try:
        dataframe = pd.read_excel(
            BytesIO(data),
            engine="odf",
            skiprows=6,
            header=None,
        )
    except ImportError as exc:
        raise FIError(
            "ODS-stöd saknas. Installera odfpy."
        ) from exc
    except Exception as exc:
        raise FIError(
            "Kunde inte läsa FI:s aggregerade "
            "ODS-fil."
        ) from exc

    if dataframe.empty:
        raise FIError(
            "FI:s aggregatfil innehöll inga rader."
        )

    if len(dataframe.columns) != 4:
        raise FIError(
            "FI:s aggregatfil hade oväntat antal "
            f"kolumner: {len(dataframe.columns)} "
            "(förväntade 4)."
        )

    dataframe.columns = [
        "Emittentens namn",
        "Emittentens LEI-kod",
        "Summa blankning %",
        "Positionsdatum",
    ]

    return dataframe


def fetch_current() -> list[dict]:
    """
    Hämtar FI:s aktuella aggregerade blankning.

    Källa:
        GetBlankningsregisterAggregat

    Datan är FI:s egen aggregering och ska därför
    inte rekonstrueras från positionsinnehavare.
    """

    fetched_at_value = fetched_at()

    data = fetch_aggregate()

    dataframe = read_aggregate_file(
        data
    )

    try:
        source_dates = (
            pd.to_datetime(
                dataframe["Positionsdatum"],
                errors="coerce",
            )
            .dropna()
        )

        if source_dates.empty:
            raise ValueError(
                "Ingen giltig positionsdatumkolumn."
            )

        source_date = (
            source_dates
            .max()
            .date()
            .isoformat()
        )

        records = normalize_records(
            dataframe,
            fetched_at_value,
            source_date,
        )

    except ValueError as exc:
        raise FIError(
            "Kunde inte normalisera FI:s "
            f"aggregatdata: {exc}"
        ) from exc

    if not records:
        raise FIError(
            "FI:s aggregatfil innehöll inga "
            "giltiga observationer."
        )

    return records
