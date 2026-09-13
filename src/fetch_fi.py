"""
Hämtar aktuell aggregerad blankningsdata från Finansinspektionen.

FI:s blankningsregister visar:
- emittentens namn
- LEI
- senaste positionsdatum
- summa blankning %

Vi sparar varje körning som en separat JSONL-fil.

Historiska råfiler ska aldrig skrivas över.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import requests


FI_URL = "https://www.fi.se/sv/vara-register/blankningsregistret/"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(compatible; Blankdiss/0.1; research project)"
    )
}


def normalize_percent(value) -> float | None:
    """Konverterar FI:s svenska procentformat till float."""

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    text = text.replace("%", "")
    text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return None


def normalize_date(value) -> str | None:
    """Normaliserar datum till YYYY-MM-DD."""

    if value is None:
        return None

    parsed = pd.to_datetime(
        value,
        errors="coerce",
        dayfirst=False,
    )

    if pd.isna(parsed):
        return None

    return parsed.strftime("%Y-%m-%d")


def find_target_table(html: str) -> pd.DataFrame:
    """
    Hittar tabellen med:
    Emittentens namn / LEI / Positionsdatum / Summa blankning %.
    """

    tables = pd.read_html(html)

    for table in tables:
        columns = {
            str(column).strip()
            for column in table.columns
        }

        required = {
            "Emittentens namn",
            "Emittentens LEI-kod",
            "Positionsdatum senaste position",
            "Summa blankning %",
        }

        if required.issubset(columns):
            return table

    raise RuntimeError(
        "Kunde inte hitta FI:s aggregerade blankningstabell."
    )


def fetch_current() -> list[dict]:
    """Hämtar aktuell aggregerad blankning."""

    response = requests.get(
        FI_URL,
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    table = find_target_table(response.text)

    records: list[dict] = []

    for _, row in table.iterrows():
        issuer = str(
            row["Emittentens namn"]
        ).strip()

        lei = str(
            row["Emittentens LEI-kod"]
        ).strip()

        position_date = normalize_date(
            row["Positionsdatum senaste position"]
        )

        short_interest = normalize_percent(
            row["Summa blankning %"]
        )

        if (
            not issuer
            or issuer.lower() == "nan"
            or not lei
            or lei.lower() == "nan"
            or position_date is None
            or short_interest is None
        ):
            continue

        records.append(
            {
                "snapshot_date": date.today().isoformat(),
                "position_date": position_date,
                "lei": lei,
                "issuer": issuer,
                "short_interest_pct": short_interest,
                "source": FI_URL,
            }
        )

    return records


def write_jsonl(records: list[dict]) -> Path:
    """Skriver en ny daterad råfil."""

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        RAW_DIR
        / f"fi_aggregate_{date.today().isoformat()}.jsonl"
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
        description="Hämta FI:s aggregerade blankning."
    )

    parser.add_argument(
        "--sample",
        action="store_true",
        help="Skapa sample-data istället för att hämta FI.",
    )

    args = parser.parse_args()

    if args.sample:
        records = [
            {
                "snapshot_date": "2026-09-13",
                "position_date": "2026-09-11",
                "lei": "SAMPLE001",
                "issuer": "Sample Aktiebolag",
                "short_interest_pct": 3.42,
                "source": "sample",
            }
        ]
    else:
        records = fetch_current()

    path = write_jsonl(records)

    print(
        f"FI: {len(records)} poster → {path}"
    )


if __name__ == "__main__":
    main()
