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
def detect_file_format(
    data: bytes,
) -> str:
    """
    Identifierar filformat utifrån filens bytes.
    XLSX och ODS är båda ZIP-baserade, så ZIP
    används som första signal. Det faktiska
    kalkylbladsformatet avgörs därefter av
    parsern.
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
    Funktionen litar inte på filändelsen.
    Den loggar vad FI faktiskt returnerar.
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
def print_table_diagnostic(
    table: pd.DataFrame,
    source_date: date,
) -> None:
    """
    Skriver ut diagnostik för en historisk
    FI-tabell när rubrikerna inte känns igen.
    """
    print(
        "FI backfill: kunde inte identifiera "
        f"kolumnerna för {source_date}."
    )
    print(
        "FI backfill: "
        f"tabellens shape={table.shape}"
    )
    print(
        "FI backfill: kolumnindex="
        f"{list(table.columns)!r}"
    )
    preview_rows = min(
        25,
        len(table),
    )
    print(
        "FI backfill: "
        f"första {preview_rows} rader:"
    )
    for index in range(
        preview_rows
    ):
        values = [
            repr(value)
            for value in table.iloc[index].tolist()
        ]
        print(
            f"  [{index}] "
            + " | ".join(values)
        )
def find_header_row(
    table: pd.DataFrame,
) -> int | None:
    """
    Försöker hitta rubrikraden i en FI-fil.
    Historiska FI-filer kan ha andra rubriker
    än dagens aggregatfil, därför används flera
    signaler.
    """
    for header_row in range(
        min(30, len(table))
    ):
        values = (
            table
            .iloc[header_row]
            .astype(str)
            .str.strip()
        )
        text = (
            " | ".join(
                values.tolist()
            )
            .lower()
        )
        # Dagens/nyare struktur.
        if (
            "emittentens namn" in text
            and "summa blankning" in text
        ):
            return header_row
        # Alternativ stavning.
        if (
            "emittent" in text
            and "blankning" in text
            and (
                "lei" in text
                or "positionsdatum" in text
                or "position" in text
            )
        ):
            return header_row
        # Historiska filer kan använda
        # "issuer" / "short".
        if (
            "issuer" in text
            and (
                "short" in text
                or "position" in text
            )
        ):
            return header_row
    return None
def prepare_table(
    table: pd.DataFrame,
    source_date: date,
) -> pd.DataFrame:
    """
    Försöker omvandla en rå Excel-tabell till
    en tabell med identifierbara kolumner.
    """
    header_row = find_header_row(
        table
    )
    if header_row is None:
        print_table_diagnostic(
            table,
            source_date,
        )
        raise FIError(
            "Kunde inte identifiera "
            "rubrikraden i FI:s historiska "
            f"fil för {source_date}."
        )
    candidate = (
        table
        .iloc[header_row]
        .astype(str)
        .str.strip()
    )
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
    result = result.reset_index(
        drop=True
    )
    print(
        "FI backfill: identifierad "
        f"rubrikrad={header_row}"
    )
    print(
        "FI backfill: kolumner="
        f"{list(result.columns)!r}"
    )
    return result
def read_aggregate_file(
    data: bytes,
    source_date: date,
) -> pd.DataFrame:
    """
    Läser en historisk FI-aggregatfil.
    För ZIP-baserade filer provas först XLSX.
    Om XLSX lyckas läser vi inte filen med ODS
    bara för att rubrikerna inte hittades.
    Detta är viktigt eftersom både XLSX och ODS
    är ZIP-baserade.
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
    # FI-filen vi har sett är XLSX.
    # Prova därför XLSX först.
    if file_format == "zip":
        try:
            table = pd.read_excel(
                BytesIO(data),
                engine="openpyxl",
                header=None,
            )
            print(
                "FI backfill: "
                "parser=openpyxl, "
                f"shape={table.shape}"
            )
            return prepare_table(
                table,
                source_date,
            )
        except FIError:
            # Detta är ett riktigt innehålls-/formatfel
            # och ska inte maskeras av ett senare ODS-fel.
            raise
        except Exception as exc:
            errors.append(
                f"openpyxl: {exc}"
            )
            print(
                "FI backfill: "
                f"parser=openpyxl misslyckades: "
                f"{exc}"
            )
        # Om openpyxl misslyckades helt kan filen
        # fortfarande vara ODS.
        try:
            table = pd.read_excel(
                BytesIO(data),
                engine="odf",
                header=None,
            )
            print(
                "FI backfill: "
                "parser=odf, "
                f"shape={table.shape}"
            )
            return prepare_table(
                table,
                source_date,
            )
        except FIError:
            raise
        except Exception as exc:
            errors.append(
                f"odf: {exc}"
            )
    elif file_format == "ole":
        try:
            table = pd.read_excel(
                BytesIO(data),
                engine="xlrd",
                header=None,
            )
            print(
                "FI backfill: "
                "parser=xlrd, "
                f"shape={table.shape}"
            )
            return prepare_table(
                table,
                source_date,
            )
        except FIError:
            raise
        except Exception as exc:
            errors.append(
                f"xlrd: {exc}"
            )
    else:
        errors.append(
            "okänt filformat"
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
