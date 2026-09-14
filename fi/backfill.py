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
    Vi provar både XLSX och ODS eftersom FI:s
    äldre historiska filer kan ha annat format.
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
def detect_file_format(
    data: bytes,
) -> str:
    """
    Identifierar filformat utifrån filens bytes.
    Detta är mer tillförlitligt än filändelsen
    eftersom servern kan returnera ett annat
    format än URL:en antyder.
    """
    if data.startswith(
        b"PK\x03\x04"
    ):
        return "zip"
    if data.startswith(
        b"\xd0\xcf\x11\xe0"
    ):
        return "ole"
    if data.startswith(
        b"<?xml"
    ):
        return "xml"
    if data.lstrip().startswith(
        b"<"
    ):
        return "html-or-xml"
    return "unknown"
def describe_content(
    data: bytes,
) -> str:
    """
    Returnerar en kort diagnostisk beskrivning
    av innehållet.
    """
    file_format = detect_file_format(
        data
    )
    preview = (
        data[:120]
        .replace(b"\r", b" ")
        .replace(b"\n", b" ")
    )
    try:
        preview_text = (
            preview
            .decode(
                "utf-8",
                errors="replace",
            )
        )
    except Exception:
        preview_text = repr(
            preview
        )
    return (
        f"format={file_format}, "
        f"bytes={len(data)}, "
        f"preview={preview_text!r}"
    )
def download_historical_file(
    session: requests.Session,
    source_date: date,
) -> tuple[bytes, str] | None:
    """
    Hämtar en daterad FI-fil.
    Returnerar None om FI inte har någon fil
    för datumet.
    Viktigt:
    Funktionen litar inte på filändelsen.
    Den loggar i stället vad FI faktiskt
    returnerar så att formatproblem kan
    diagnostiseras utan gissningar.
    """
    for url in historical_urls(
        source_date
    ):
        print(
            "FI backfill: försöker "
            f"{url}"
        )
        try:
            response = session.get(
                url,
                headers=HEADERS,
                timeout=60,
                allow_redirects=True,
            )
        except requests.RequestException as exc:
            raise FIError(
                f"FI-nätverksfel för "
                f"{source_date}: {exc}"
            ) from exc
        content_type = response.headers.get(
            "Content-Type",
            "",
        )
        print(
            "FI backfill: "
            f"HTTP={response.status_code}, "
            f"Content-Type={content_type!r}, "
            f"bytes={len(response.content)}"
        )
        if response.url != url:
            print(
                "FI backfill: redirect -> "
                f"{response.url}"
            )
        if response.status_code == 404:
            print(
                "FI backfill: fil saknas "
                f"för {source_date}"
            )
            continue
        if response.status_code != 200:
            raise FIError(
                f"FI returnerade HTTP "
                f"{response.status_code} "
                f"för {source_date}."
            )
        data = response.content
        if len(data) < 100:
            print(
                "FI backfill: svaret är "
                "för litet för att vara "
                "en historisk datafil."
            )
            continue
        description = describe_content(
            data
        )
        print(
            "FI backfill: "
            f"{description}"
        )
        lowered_content_type = (
            content_type.lower()
        )
        if (
            "html" in lowered_content_type
            or "text/html" in lowered_content_type
        ):
            raise FIError(
                "FI returnerade HTML i stället "
                "för en historisk datafil "
                f"för {source_date}. "
                f"{description}"
            )
        return (
            data,
            url,
        )
    return None
def read_aggregate_file(
    data: bytes,
    source_date: date,
) -> pd.DataFrame:
    """
    Läser en historisk FI-aggregatfil.
    Stödjer XLSX och ODS.
    Filformatet diagnostiseras först så att
    felmeddelanden blir tydligare.
    """
    file_format = detect_file_format(
        data
    )
    print(
        "FI backfill: läser fil "
        f"{source_date}: "
        f"format={file_format}"
    )
    errors: list[str] = []
    engines: list[str]
    if file_format == "zip":
        engines = [
            "openpyxl",
            "odf",
        ]
    elif file_format == "ole":
        engines = [
            "xlrd",
        ]
    else:
        engines = [
            "openpyxl",
            "odf",
            "xlrd",
        ]
    for engine in engines:
        try:
            table = pd.read_excel(
                BytesIO(data),
                engine=engine,
                header=None,
            )
            print(
                "FI backfill: "
                f"parser={engine}, "
                f"shape={table.shape}"
            )
            # Försök först hitta rubrikraden
            # dynamiskt.
            for header_row in range(
                min(20, len(table))
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
                    print(
                        "FI backfill: "
                        f"rubrikrad={header_row}"
                    )
                    return result.reset_index(
                        drop=True
                    )
            # Alternativ rubrikdetektion.
            for header_row in range(
                min(20, len(table))
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
                    "emittent" in text
                    and (
                        "blankning" in text
                        or "blanknings" in text
                    )
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
                    print(
                        "FI backfill: "
                        "alternativ rubrikrad="
                        f"{header_row}"
                    )
                    return result.reset_index(
                        drop=True
                    )
            # Fallback för FI:s äldre fasta
            # filstruktur.
            if table.shape[1] >= 4:
                print(
                    "FI backfill: "
                    "använder fallback "
                    "för första fyra kolumner."
                )
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
            print(
                "FI backfill: "
                f"parser={engine} misslyckades: "
                f"{exc}"
            )
    raise FIError(
        "Kunde inte läsa historisk "
        f"FI-fil för {source_date}: "
        f"{describe_content(data)}; "
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
    Hämtar historiska aggregerade
    blankningspositioner.
    Vi provar varje kalenderdag eftersom FI
    först publicerade aggregatet veckovis och
    senare gick över till fortlöpande publicering.
    Befintliga datum hoppas över för att
    processen ska vara idempotent.
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
                    existing_rows = sum(
                        1
                        for _ in file
                    )
                rows += existing_rows
                print(
                    "FI backfill: "
                    f"{source_date} finns redan "
                    f"({existing_rows} rader), "
                    "hoppar över."
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
            "FI backfill: källa = "
            f"{url}"
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
