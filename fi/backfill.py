"""Backfill av Finansinspektionens historiska aggregerade blankning."""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

from .config import HEADERS, RAW_DIR
from .errors import FIError
from .normalize import normalize_records, now_stockholm


DEFAULT_START_DATE = date(2022, 5, 25)

HISTORICAL_DIR = (
    RAW_DIR
    / "historical"
)


def parse_date(
    value: str,
) -> date:
    """Tolkar YYYY-MM-DD."""

    try:
        return datetime.strptime(
            value,
            "%Y-%m-%d",
        ).date()

    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Ogiltigt datum: {value}. "
            "Använd YYYY-MM-DD."
        ) from exc


def date_range(
    start: date,
    end: date,
):
    """Itererar över alla kalenderdagar."""

    current = start

    while current <= end:
        yield current
        current += timedelta(days=1)


def historical_urls(
    source_date: date,
) -> list[str]:
    """
    Returnerar möjliga FI-URL:er för en daterad
    historisk aggregatfil.

    FI:s historiska filer följer formatet:

        aggregerade-blankningspositioner-YYYY-MM-DD.xlsx
    """

    date_text = source_date.isoformat()

    base = (
        "https://www.fi.se/contentassets/"
        "79e6c3558bd9473fb70a418f51df48d0/"
    )

    stem = (
        "aggregerade-blankningspositioner-"
        f"{date_text}"
    )

    return [
        f"{base}{stem}.xlsx",
        f"{base}{stem}.ods",
    ]


def download_historical_file(
    session: requests.Session,
    source_date: date,
) -> tuple[bytes, str] | None:
    """
    Hämtar en daterad FI-fil.

    Returnerar None om FI inte har någon fil
    för datumet.
    """

    for url in historical_urls(source_date):

        try:
            response = session.get(
                url,
                headers=HEADERS,
                timeout=60,
            )

        except requests.RequestException as exc:
            raise FIError(
                f"FI-nätverksfel för "
                f"{source_date}: {exc}"
            ) from exc

        if response.status_code == 404:
            continue

        if response.status_code != 200:
            raise FIError(
                f"FI returnerade HTTP "
                f"{response.status_code} "
                f"för {source_date}."
            )

        content_type = response.headers.get(
            "Content-Type",
            "",
        ).lower()

        if len(response.content) < 100:
            continue

        if "html" in content_type:
            continue

        return (
            response.content,
            url,
        )

    return None


def read_aggregate_file(
    data: bytes,
    source_date: date,
) -> pd.DataFrame:
    """
    Läser en historisk FI-aggregatfil.

    Stödjer både XLSX och ODS.
    """

    errors: list[str] = []

    for engine in (
        "openpyxl",
        "odf",
    ):
        try:
            table = pd.read_excel(
                BytesIO(data),
                engine=engine,
                header=None,
            )

            # Försök först hitta rubrikraden
            # dynamiskt.
            for header_row in range(
                min(15, len(table))
            ):
                candidate = (
                    table
                    .iloc[header_row]
                    .astype(str)
                    .str.strip()
                )

                text = (
                    " | ".join(
                        candidate.tolist()
                    )
                    .lower()
                )

                if (
                    "emittentens namn" in text
                    and "summa blankning" in text
                ):
                    result = (
                        table
                        .iloc[
                            header_row + 1 :
                        ]
                        .copy()
                    )

                    result.columns = (
                        candidate.tolist()
                    )

                    return result.reset_index(
                        drop=True
                    )

            # Fallback för FI:s äldre fasta
            # filstruktur.
            if table.shape[1] >= 4:

                result = (
                    table
                    .iloc[7:, :4]
                    .copy()
                )

                result.columns = [
                    "Emittentens namn",
                    "Emittentens LEI-kod",
                    "Summa blankning %",
                    "Positionsdatum",
                ]

                return result.reset_index(
                    drop=True
                )

        except Exception as exc:
            errors.append(
                f"{engine}: {exc}"
            )

    raise FIError(
        "Kunde inte läsa historisk "
        f"FI-fil för {source_date}: "
        f"{'; '.join(errors)}"
    )


def output_path(
    source_date: date,
) -> Path:
    """Returnerar JSONL-sökvägen för ett datum."""

    return (
        HISTORICAL_DIR
        / (
            "fi_aggregate_"
            f"{source_date.isoformat()}"
            ".jsonl"
        )
    )


def write_historical_snapshot(
    records: list[dict],
    source_date: date,
) -> Path:
    """Skriver en historisk JSONL-snapshot."""

    HISTORICAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_path(
        source_date
    )

    lines = [
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
        for record in records
    ]

    path.write_text(
        "".join(lines),
        encoding="utf-8",
    )

    return path


def backfill(
    start: date = DEFAULT_START_DATE,
    end: date | None = None,
    delay: float = 0.15,
) -> tuple[int, int]:
    """
    Hämtar hela den historiska period som
    FI:s daterade filer täcker.

    Vi provar varje kalenderdag eftersom FI
    först publicerade aggregatet veckovis och
    senare gick över till fortlöpande publicering.
    Dagar utan fil hoppas över.
    """

    if end is None:
        end = (
            now_stockholm().date()
            - timedelta(days=1)
        )

    if start > end:
        raise FIError(
            f"Startdatum {start} ligger "
            f"efter slutdatum {end}."
        )

    found = 0
    rows = 0

    session = requests.Session()

    for source_date in date_range(
        start,
        end,
    ):

        existing = output_path(
            source_date
        )

        # Idempotens:
        # finns datumet redan hoppar vi över
        # nätverksanropet.
        if existing.exists():

            found += 1

            try:
                with existing.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    rows += sum(
                        1
                        for _ in file
                    )

            except OSError:
                pass

            continue

        result = download_historical_file(
            session,
            source_date,
        )

        if result is None:
            time.sleep(delay)
            continue

        data, url = result

        table = read_aggregate_file(
            data,
            source_date,
        )

        fetched_at_value = (
            now_stockholm().isoformat(
                timespec="seconds"
            )
        )

        try:
            records = normalize_records(
                table,
                fetched_at_value,
                source_date.isoformat(),
            )

        except ValueError as exc:
            raise FIError(
                "Kunde inte normalisera "
                f"FI-data för {source_date}: "
                f"{exc}"
            ) from exc

        if not records:
            raise FIError(
                "FI-filen för "
                f"{source_date} innehöll "
                "inga observationer."
            )

        path = write_historical_snapshot(
            records,
            source_date,
        )

        found += 1
        rows += len(records)

        print(
            f"FI backfill: {source_date}: "
            f"{len(records)} observationer "
            f"-> {path}"
        )

        print(
            f"FI backfill: källa = {url}"
        )

        time.sleep(delay)

    return found, rows


def parse_args() -> argparse.Namespace:
    """Parsar kommandoradsargument."""

    parser = argparse.ArgumentParser(
        description=(
            "Hämta historiska "
            "aggregerade "
            "blankningspositioner "
            "från Finansinspektionen."
        )
    )

    parser.add_argument(
        "--from",
        dest="start",
        type=parse_date,
        default=DEFAULT_START_DATE,
        help=(
            "Första datum, YYYY-MM-DD. "
            f"Standard: "
            f"{DEFAULT_START_DATE}."
        ),
    )

    parser.add_argument(
        "--to",
        dest="end",
        type=parse_date,
        default=None,
        help=(
            "Sista datum, YYYY-MM-DD. "
            "Standard: gårdagen."
        ),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.15,
        help=(
            "Paus mellan FI-anrop "
            "i sekunder. "
            "Standard: 0.15."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """CLI-entrypoint."""

    args = parse_args()

    try:
        found, rows = backfill(
            start=args.start,
            end=args.end,
            delay=args.delay,
        )

        print(
            "FI backfill klart: "
            f"{found} filer, "
            f"{rows} observationer."
        )

        return 0

    except FIError as exc:

        print(
            f"FI backfill: FEL: {exc}"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
