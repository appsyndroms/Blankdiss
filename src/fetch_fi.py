"""
Hämtar aktuella blankningspositioner från Finansinspektionen.

Datakälla:
    https://www.fi.se/sv/vara-register/blankningsregistret/

FI-sidan hämtas först med requests.
Därefter analyseras den hämtade HTML-texten med pandas.

Viktigt:
    pandas.read_html() får aldrig FI_URL direkt.
    Detta undviker att pandas/lxml försöker läsa URL:en
    som en lokal fil eller göra ett eget HTTP-anrop.

Resultatet sparas som en daterad JSONL-fil:

    data/raw/fi_aggregate_YYYY-MM-DD.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"

FI_URL = (
    "https://www.fi.se/sv/vara-register/"
    "blankningsregistret/"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(compatible; Blankdiss/1.0; "
        "+https://github.com/appsyndroms/Blankdiss)"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": (
        "sv-SE,sv;q=0.9,en-US;q=0.8,en;q=0.7"
    ),
}


def normalize_text(value) -> str:
    """Normaliserar text för jämförelser."""

    if value is None:
        return ""

    return (
        str(value)
        .replace("\xa0", " ")
        .strip()
    )


def normalize_percent(value) -> float | None:
    """
    Konverterar en procentangivelse till float.

    Exempel:

        "3,42 %" -> 3.42
        "3.42%"  -> 3.42
        3.42     -> 3.42
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
        return float(text)
    except ValueError:
        return None


def normalize_date(value) -> str | None:
    """
    Normaliserar datum till YYYY-MM-DD.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    parsed = pd.to_datetime(
        value,
        errors="coerce",
        dayfirst=True,
    )

    if pd.isna(parsed):
        return None

    return parsed.strftime(
        "%Y-%m-%d"
    )


def fetch_html() -> str:
    """
    Hämtar FI:s HTML med requests.

    Pandas/lxml används inte för själva HTTP-anropet.
    """

    print(
        f"Hämtar FI: {FI_URL}"
    )

    response = requests.get(
        FI_URL,
        headers=HEADERS,
        timeout=30,
    )

    print(
        f"FI HTTP-status: {response.status_code}"
    )

    response.raise_for_status()

    html = response.text

    print(
        f"FI HTML-längd: {len(html)} tecken"
    )

    if not html.strip():
        raise RuntimeError(
            "FI returnerade ett tomt HTML-svar."
        )

    return html


def find_target_table(
    tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Hittar tabellen som innehåller FI:s aktuella
    blankningspositioner.
    """

    for table_number, table in enumerate(tables):

        if table.empty:
            continue

        columns = [
            normalize_text(column).lower()
            for column in table.columns
        ]

        column_text = " | ".join(columns)

        issuer_match = (
            "emittentens namn" in column_text
            or "emittent" in column_text
        )

        lei_match = (
            "emittentens lei-kod" in column_text
            or "lei-kod" in column_text
            or "lei" in column_text
        )

        date_match = (
            "positionsdatum senaste position"
            in column_text
            or "positionsdatum" in column_text
        )

        short_match = (
            "summa blankning %" in column_text
            or "summa blankning" in column_text
        )

        if (
            issuer_match
            and lei_match
            and date_match
            and short_match
        ):
            print(
                "FI måltabell hittad: "
                f"tabell {table_number}"
            )

            print(
                "FI-kolumner: "
                + ", ".join(columns)
            )

            return table

    available = []

    for index, table in enumerate(tables):

        columns = [
            normalize_text(column)
            for column in table.columns
        ]

        available.append(
            f"Tabell {index}: {columns}"
        )

    details = "\n".join(
        available
    )

    raise RuntimeError(
        "Kunde inte hitta FI:s blankningstabell.\n"
        "Tillgängliga tabeller:\n"
        f"{details}"
    )


def resolve_column(
    columns,
    patterns: list[str],
) -> str | None:
    """
    Hittar den faktiska kolumnrubriken utifrån
    möjliga delar av kolumnnamnet.
    """

    normalized = {
        column: normalize_text(column).lower()
        for column in columns
    }

    for pattern in patterns:

        for original, value in normalized.items():

            if pattern in value:
                return original

    return None


def fetch_current() -> list[dict]:
    """
    Hämtar aktuella aggregerade blankningsnivåer från FI.
    """

    html = fetch_html()

    print(
        "Tolkar hämtad FI-HTML..."
    )

    try:
        # OBS:
        # Här skickas HTML-INNEHÅLLET till pandas.
        # FI_URL skickas aldrig till read_html().
        tables = pd.read_html(
            html
        )
    except Exception as exc:
        raise RuntimeError(
            "Kunde inte tolka FI:s hämtade HTML "
            "som tabeller."
        ) from exc

    print(
        f"FI-tabeller hittade: {len(tables)}"
    )

    table = find_target_table(
        tables
    )

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

    if not issuer_column:
        raise RuntimeError(
            "Kunde inte hitta kolumnen "
            "för emittent."
        )

    if not lei_column:
        raise RuntimeError(
            "Kunde inte hitta LEI-kolumnen."
        )

    if not position_date_column:
        raise RuntimeError(
            "Kunde inte hitta "
            "positionsdatum."
        )

    if not short_interest_column:
        raise RuntimeError(
            "Kunde inte hitta kolumnen "
            "Summa blankning %."
        )

    records: list[dict] = []

    snapshot_date = date.today().isoformat()

    for _, row in table.iterrows():

        issuer = normalize_text(
            row.get(
                issuer_column
            )
        )

        lei = normalize_text(
            row.get(
                lei_column
            )
        )

        position_date = normalize_date(
            row.get(
                position_date_column
            )
        )

        short_interest_pct = normalize_percent(
            row.get(
                short_interest_column
            )
        )

        if not issuer:
            continue

        if not lei:
            continue

        if not position_date:
            continue

        if short_interest_pct is None:
            continue

        records.append(
            {
                "snapshot_date": (
                    snapshot_date
                ),

                "position_date": (
                    position_date
                ),

                "lei": lei,

                "issuer": issuer,

                "short_interest_pct": (
                    short_interest_pct
                ),

                "source": FI_URL,
            }
        )

    return records


def write_jsonl(
    records: list[dict],
) -> Path:
    """
    Skriver resultatet som en ny daterad JSONL-fil.
    """

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        RAW_DIR
        / (
            "fi_aggregate_"
            f"{date.today().isoformat()}.jsonl"
        )
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for record in records:

            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    return path


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Hämta aktuella aggregerade "
            "blankningspositioner från FI."
        )
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "Visa endast de N första "
            "observationerna."
        ),
    )

    args = parser.parse_args()

    records = fetch_current()

    if args.sample is not None:
        records = records[
            :args.sample
        ]

    path = write_jsonl(
        records
    )

    print(
        f"FI: {len(records)} observationer "
        f"→ {path}"
    )


if __name__ == "__main__":
    main()
