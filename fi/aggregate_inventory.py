"""Analysera datumfördelningen i FI:s aktuella aggregatfil."""
from __future__ import annotations
import argparse
from io import BytesIO
import pandas as pd
import requests
from .config import (
    FI_AGGREGATE_TIMEOUT,
    FI_AGGREGATE_URL,
    HEADERS,
)
from .errors import FIError
def fetch_aggregate() -> bytes:
    """Hämtar FI:s aktuella aggregatfil."""
    response = requests.get(
        FI_AGGREGATE_URL,
        headers=HEADERS,
        timeout=FI_AGGREGATE_TIMEOUT,
    )
    print(
        "HTTP:",
        response.status_code,
    )
    print(
        "Content-Type:",
        response.headers.get(
            "Content-Type",
            "",
        ),
    )
    print(
        "Bytes:",
        len(response.content),
    )
    print(
        "URL:",
        response.url,
    )
    if response.status_code != 200:
        raise FIError(
            "FI returnerade HTTP "
            f"{response.status_code}."
        )
    return response.content
def read_aggregate(
    data: bytes,
) -> pd.DataFrame:
    """
    Läser FI:s ODS-fil.
    FI:s aggregatfil har sex metadata-/rubrikrader
    före själva tabellen.
    """
    try:
        table = pd.read_excel(
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
            "Kunde inte läsa FI:s aggregatfil: "
            f"{exc}"
        ) from exc
    if table.empty:
        raise FIError(
            "FI:s aggregatfil innehöll inga rader."
        )
    if len(table.columns) < 4:
        raise FIError(
            "FI:s aggregatfil hade oväntat "
            f"antal kolumner: {len(table.columns)}."
        )
    table = table.iloc[:, :4].copy()
    table.columns = [
        "issuer",
        "lei",
        "short_interest_pct",
        "position_date",
    ]
    return table
def normalize_dates(
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Normaliserar positionsdatum."""
    table = table.copy()
    table["position_date"] = pd.to_datetime(
        table["position_date"],
        errors="coerce",
    )
    return table
def print_summary(
    table: pd.DataFrame,
) -> None:
    """Skriver en sammanfattning av datumfördelningen."""
    total_rows = len(table)
    valid_dates = table[
        table["position_date"].notna()
    ].copy()
    unique_dates = sorted(
        valid_dates["position_date"]
        .dt.date
        .unique()
    )
    print()
    print("=" * 72)
    print("FI AGGREGATE INVENTORY")
    print("=" * 72)
    print()
    print(
        f"Råa rader:              {total_rows:,}"
    )
    print(
        "Rader med giltigt datum: "
        f"{len(valid_dates):,}"
    )
    print(
        "Ogiltiga/saknade datum:  "
        f"{total_rows - len(valid_dates):,}"
    )
    print(
        "Unika positionsdatum:    "
        f"{len(unique_dates):,}"
    )
    if not unique_dates:
        raise FIError(
            "FI-filen innehöll inga giltiga "
            "positionsdatum."
        )
    print(
        "Första datum:            "
        f"{unique_dates[0]}"
    )
    print(
        "Sista datum:             "
        f"{unique_dates[-1]}"
    )
    print()
    print("-" * 72)
    print("Antal observationer per positionsdatum")
    print("-" * 72)
    counts = (
        valid_dates
        .groupby(
            valid_dates["position_date"].dt.date
        )
        .size()
        .sort_index()
    )
    for position_date, count in counts.items():
        print(
            f"{position_date}    {count:>6,}"
        )
    print()
    print("-" * 72)
    print("Årssammanfattning")
    print("-" * 72)
    yearly = (
        valid_dates
        .groupby(
            valid_dates["position_date"].dt.year
        )
        .size()
        .sort_index()
    )
    for year, count in yearly.items():
        print(
            f"{year}    {count:>10,}"
        )
def print_transition(
    table: pd.DataFrame,
    start: str,
    end: str,
) -> None:
    """
    Skriver detaljer kring en specifik
    övergång i historiken.
    """
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    subset = table[
        table["position_date"].between(
            start_date,
            end_date,
        )
    ].copy()
    if subset.empty:
        print()
        print(
            f"Inga observationer mellan "
            f"{start} och {end}."
        )
        return
    counts = (
        subset
        .groupby(
            subset["position_date"].dt.date
        )
        .size()
        .sort_index()
    )
    print()
    print("-" * 72)
    print(
        f"Detaljer: {start} → {end}"
    )
    print("-" * 72)
    for position_date, count in counts.items():
        print(
            f"{position_date}    {count:>6,}"
        )
def print_gaps(
    table: pd.DataFrame,
) -> None:
    """
    Identifierar luckor mellan observerade datum.
    Detta är diagnostik. Alla kalenderdagar behöver
    inte ha en publicerad observation.
    """
    dates = (
        table["position_date"]
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )
    if dates.empty:
        return
    expected = pd.date_range(
        dates.iloc[0],
        dates.iloc[-1],
        freq="D",
    )
    observed = pd.DatetimeIndex(
        dates
    )
    missing = expected.difference(
        observed
    )
    print()
    print("-" * 72)
    print("Luckor i kalenderdatum")
    print("-" * 72)
    print(
        f"Förväntade kalenderdagar: "
        f"{len(expected):,}"
    )
    print(
        f"Observerade datum:        "
        f"{len(observed):,}"
    )
    print(
        f"Saknade kalenderdatum:    "
        f"{len(missing):,}"
    )
    if missing.empty:
        print(
            "Inga kalenderluckor."
        )
        return
    print()
    print(
        "Första 100 saknade datum:"
    )
    for value in missing[:100]:
        print(
            value.date()
        )
    if len(missing) > 100:
        print(
            f"... och ytterligare "
            f"{len(missing) - 100:,}."
        )
def parse_args() -> argparse.Namespace:
    """Tolkar kommandoradsargument."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyserar datumfördelningen i "
            "FI:s aktuella aggregatfil."
        )
    )
    parser.add_argument(
        "--transition-start",
        default="2022-05-25",
        help=(
            "Startdatum för detaljerad "
            "övergångsanalys."
        ),
    )
    parser.add_argument(
        "--transition-end",
        default="2022-06-15",
        help=(
            "Slutdatum för detaljerad "
            "övergångsanalys."
        ),
    )
    return parser.parse_args()
def main() -> int:
    """Kör inventeringen."""
    args = parse_args()
    try:
        data = fetch_aggregate()
        table = read_aggregate(
            data
        )
        table = normalize_dates(
            table
        )
        print_summary(
            table
        )
        print_transition(
            table,
            args.transition_start,
            args.transition_end,
        )
        print_gaps(
            table
        )
    except (
        FIError,
        requests.RequestException,
    ) as exc:
        print(
            f"FI aggregate inventory FEL: {exc}"
        )
        return 1
    return 0
if __name__ == "__main__":
    raise SystemExit(
        main()
    )
