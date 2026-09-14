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
    print("HTTP:", response.status_code)
    print(
        "Content-Type:",
        response.headers.get("Content-Type", ""),
    )
    print("Bytes:", len(response.content))
    print("URL:", response.url)
    if response.status_code != 200:
        raise FIError(
            "FI returnerade HTTP "
            f"{response.status_code}."
        )
    return response.content
def read_aggregate(
    data: bytes,
) -> pd.DataFrame:
    """Läser FI:s ODS-fil."""
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
def get_counts(
    table: pd.DataFrame,
) -> pd.Series:
    """Returnerar antal observationer per positionsdatum."""
    valid_dates = table[
        table["position_date"].notna()
    ].copy()
    return (
        valid_dates
        .groupby(
            valid_dates["position_date"].dt.date
        )
        .size()
        .sort_index()
    )
def print_summary(
    table: pd.DataFrame,
) -> None:
    """Skriver en kort diagnostisk sammanfattning."""
    total_rows = len(table)
    valid_dates = table[
        table["position_date"].notna()
    ].copy()
    counts = get_counts(table)
    if counts.empty:
        raise FIError(
            "FI-filen innehöll inga giltiga "
            "positionsdatum."
        )
    print()
    print("=" * 72)
    print("FI AGGREGATE INVENTORY")
    print("=" * 72)
    print(
        f"Råa rader:              "
        f"{total_rows:,}"
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
        f"{len(counts):,}"
    )
    print(
        "Första datum:            "
        f"{counts.index[0]}"
    )
    print(
        "Sista datum:             "
        f"{counts.index[-1]}"
    )
    print()
    print("-" * 72)
    print(
        "TOPP 30 DATUM EFTER ANTAL OBSERVATIONER"
    )
    print("-" * 72)
    top_counts = (
        counts
        .sort_values(
            ascending=False
        )
        .head(30)
    )
    for position_date, count in top_counts.items():
        print(
            f"{position_date}    "
            f"{count:>6,}"
        )
    print()
    print("-" * 72)
    print("DATUM RUNT 2022-06-09")
    print("-" * 72)
    transition = counts.loc[
        (
            counts.index
            >= pd.Timestamp(
                "2022-05-25"
            ).date()
        )
        & (
            counts.index
            <= pd.Timestamp(
                "2022-06-15"
            ).date()
        )
    ]
    if transition.empty:
        print(
            "Inga observationer i intervallet."
        )
    else:
        for position_date, count in (
            transition.items()
        ):
            print(
                f"{position_date}    "
                f"{count:>6,}"
            )
    print()
    print("-" * 72)
    print("ÅRSSAMMANFATTNING")
    print("-" * 72)
    yearly = (
        valid_dates
        .groupby(
            valid_dates[
                "position_date"
            ].dt.year
        )
        .size()
        .sort_index()
    )
    for year, count in yearly.items():
        print(
            f"{year}    "
            f"{count:>10,}"
        )
def print_gaps(
    table: pd.DataFrame,
) -> None:
    """Identifierar luckor mellan observerade datum."""
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
        "Förväntade kalenderdagar: "
        f"{len(expected):,}"
    )
    print(
        "Observerade datum:        "
        f"{len(observed):,}"
    )
    print(
        "Saknade kalenderdatum:    "
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
            "... och ytterligare "
            f"{len(missing) - 100:,}."
        )
def parse_args() -> argparse.Namespace:
    """Tolkar kommandoradsargument."""
    return argparse.ArgumentParser(
        description=(
            "Analyserar datumfördelningen i "
            "FI:s aktuella aggregatfil."
        )
    ).parse_args()
def main() -> int:
    """Kör inventeringen."""
    parse_args()
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
        print_gaps(
            table
        )
    except (
        FIError,
        requests.RequestException,
    ) as exc:
        print(
            f"FI aggregate inventory FEL: "
            f"{exc}"
        )
        return 1
    return 0
if __name__ == "__main__":
    raise SystemExit(
        main()
    )
