"""
Hämtar aktuella blankningspositioner från Finansinspektionen.

Datakälla:
    https://www.fi.se/sv/vara-register/blankningsregistret/

FI-sidan hämtas med requests.

HTML-innehållet skickas till pandas via StringIO så att
pandas/lxml inte försöker tolka HTML-texten som en filväg.

Normal körning ger kort och användbar logg.

Full traceback kan aktiveras med:

    python -m src.fetch_fi --debug

Resultatet sparas som:

    data/raw/fi_aggregate_YYYY-MM-DD.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from io import StringIO
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


class FetchFIError(Exception):
    """Kontrollerat fel vid hämtning eller tolkning av FI-data."""


def normalize_text(value) -> str:
    """Normaliserar text för jämförelser."""

    if value is None:
        return ""

    return (
        str(value)
        .replace("\xa0", " ")
        .strip()
    )


def normalize_percent(
    value,
) -> float | None:
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

    except (
        TypeError,
        ValueError,
    ):
        pass

    text = normalize_text(
        value
    )

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


def normalize_date(
    value,
) -> str | None:
    """Normaliserar datum till YYYY-MM-DD."""

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

    except (
        TypeError,
        ValueError,
    ):
        pass

    text = normalize_text(
        value
    )

    if not text:
        return None

    # FI:s aktuella tabell använder normalt
    # ISO-format YYYY-MM-DD.
    parsed = pd.to_datetime(
        text,
        errors="coerce",
        dayfirst=False,
    )

    # Fallback för eventuella svenska datumformat.
    if pd.isna(parsed):

        parsed = pd.to_datetime(
            text,
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

    Returnerar endast HTML-innehållet.
    HTML skrivs aldrig ut i loggen.
    """

    try:

        response = requests.get(
            FI_URL,
            headers=HEADERS,
            timeout=30,
        )

    except requests.RequestException as exc:

        raise FetchFIError(
            "HTTP-fel vid hämtning från FI: "
            f"{type(exc).__name__}"
        ) from exc

    if response.status_code != 200:

        raise FetchFIError(
            "FI svarade med HTTP "
            f"{response.status_code}."
        )

    html = response.text

    if not html.strip():

        raise FetchFIError(
            "FI returnerade ett tomt HTML-svar."
        )

    return html


def resolve_column(
    columns,
    patterns: list[str],
) -> str | None:
    """
    Hittar den faktiska kolumnrubriken utifrån
    möjliga delar av kolumnnamnet.
    """

    normalized = {
        column: normalize_text(
            column
        ).lower()
        for column in columns
    }

    for pattern in patterns:

        for original, value in normalized.items():

            if pattern in value:
                return original

    return None


def find_target_table(
    tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Hittar FI-tabellen med:

        Emittentens namn
        Emittentens LEI-kod
        Positionsdatum senaste position
        Summa blankning %
    """

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

    raise FetchFIError(
        "Kunde inte hitta FI:s blankningstabell."
    )


def fetch_current() -> list[dict]:
    """Hämtar aktuella aggregerade blankningsnivåer från FI."""

    html = fetch_html()

    try:

        tables = pd.read_html(
            StringIO(html)
        )

    except Exception as exc:

        raise FetchFIError(
            "Kunde inte tolka FI:s HTML "
            "som tabeller."
        ) from exc

    if not tables:

        raise FetchFIError(
            "FI-sidan innehöll inga HTML-tabeller."
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

        raise FetchFIError(
            "Kolumnen för emittent saknas."
        )

    if not lei_column:

        raise FetchFIError(
            "LEI-kolumnen saknas."
        )

    if not position_date_column:

        raise FetchFIError(
            "Kolumnen för positionsdatum saknas."
        )

    if not short_interest_column:

        raise FetchFIError(
            "Kolumnen för Summa blankning % saknas."
        )

    records: list[dict] = []

    snapshot_date = (
        date.today().isoformat()
    )

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

        short_interest_pct = (
            normalize_percent(
                row.get(
                    short_interest_column
                )
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
    """Skriver resultatet som en daterad JSONL-fil."""

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
            "Spara endast de N första "
            "observationerna."
        ),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Visa full traceback vid fel."
        ),
    )

    args = parser.parse_args()

    try:

        records = fetch_current()

        if args.sample is not None:

            records = records[
                :args.sample
            ]

        path = write_jsonl(
            records
        )

        print(
            f"FI: OK - {len(records)} "
            f"observationer → {path}"
        )

    except FetchFIError as exc:

        print(
            f"FI: FEL - {exc}",
            file=sys.stderr,
        )

        if args.debug:
            raise

        sys.exit(1)

    except Exception as exc:

        print(
            "FI: OVÄNTAT FEL - "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        if args.debug:
            raise

        sys.exit(1)


if __name__ == "__main__":
    main()
