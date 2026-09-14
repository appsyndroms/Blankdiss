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
CONTINUOUS_START_DATE = date(2022, 6, 9)
HISTORICAL_DIR = RAW_DIR / "historical"
def parse_date(value: str) -> date:
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
def historical_dates(
    start: date,
    end: date,
):
    """
    Itererar över datum då FI förväntas ha
    publicerat historiska aggregatfiler.
    FI:s aggregerade historik började
    publiceras veckovis 2022-05-25.
    Från 2022-06-09 publiceras data
    fortlöpande. Historiska filer är
    handelsdagsrelaterade, så helger
    ska inte frågas efter.
    Det här minskar antalet meningslösa
    404-anrop kraftigt utan att vi
    gissar fram några observationer.
    """
    if start > end:
        return
    first_period_end = (
        CONTINUOUS_START_DATE
        - timedelta(days=1)
    )
    # Veckovisa publiceringar:
    # 2022-05-25, 2022-06-01, 2022-06-08.
    weekly_date = DEFAULT_START_DATE
    while weekly_date <= first_period_end:
        if start <= weekly_date <= end:
            yield weekly_date
        weekly_date += timedelta(days=7)
    # Från 2022-06-09 går vi dag för dag,
    # men endast måndag-fredag.
    current = max(
        start,
        CONTINUOUS_START_DATE,
    )
    while current <= end:
        if current.weekday() < 5:
            yield current
        current += timedelta(days=1)
def historical_urls(
    source_date: date,
) -> list[str]:
    """
    Returnerar möjliga FI-URL:er för
    en daterad historisk aggregatfil.
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
    """Identifierar filformat utifrån filens bytes."""
    if data.startswith(b"PK\x03\x04"):
        return "zip"
    if data.startswith(b"\xd0\xcf\x11\xe0"):
        return "ole"
    if data.startswith(b"<?xml"):
        return "xml"
    if data.lstrip().startswith(b"<"):
        return "html-or-xml"
    return "unknown"
def describe_content(
    data: bytes,
) -> str:
    """Returnerar en kort diagnostisk beskrivning."""
    file_format = detect_file_format(
        data
    )
    preview = (
        data[:120]
        .replace(b"\r", b" ")
        .replace(b"\n", b" ")
    )
    preview_text = preview.decode(
        "utf-8",
        errors="replace",
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
    XLSX är den primära historiska
    filtypen. ODS används endast som
    fallback om XLSX inte finns.
    En vanlig 404 räknas som att datumet
    saknar publicerad fil och är inte ett
    fel.
    """
    urls = historical_urls(
        source_date
    )
    for index, url in enumerate(urls):
        # ODS behöver inte provas efter
        # en vanlig XLSX-404 om vi redan
        # vet att FI:s historiska arkiv
        # använder XLSX för datumet.
        #
        # Vi behåller ändå ODS som fallback
        # eftersom arkivet kan innehålla äldre
        # filer i annat format.
        if index == 1:
            print(
                "FI backfill: XLSX saknas, "
                "provar ODS som fallback."
            )
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
                "FI-nätverksfel för "
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
            if index == 0:
                print(
                    "FI backfill: XLSX saknas "
                    f"för {source_date}"
                )
            else:
                print(
                    "FI backfill: ODS saknas "
                    f"för {source_date}"
                )
            continue
        if response.status_code != 200:
            raise FIError(
                "FI returnerade HTTP "
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
        if "html" in content_type.lower():
            print(
                "FI backfill: HTML-svar "
                f"för {source_date}, "
                "behandlar som saknad fil."
            )
            continue
        return data, url
    return None
def prepare_historical_two_column_table(
    table: pd.DataFrame,
    source_date: date,
) -> pd.DataFrame:
    """Hanterar FI:s äldre tvåkolumnsformat."""
    if table.shape[1] != 2:
        raise FIError(
            "Den historiska FI-filen har "
            f"{table.shape[1]} kolumner. "
            "Förväntade två kolumner."
        )
    result = pd.DataFrame(
        {
            "Emittentens namn": table.iloc[
                :, 0
            ],
            "Emittentens LEI-kod": None,
            "Summa blankning %": pd.to_numeric(
                table.iloc[:, 1],
                errors="coerce",
            ),
            "Positionsdatum": (
                source_date.isoformat()
            ),
        }
    )
    result = result.loc[
        result["Emittentens namn"].notna()
    ].copy()
    result = result.loc[
        result["Summa blankning %"].notna()
    ].copy()
    result = result.reset_index(
        drop=True
    )
    print(
        "FI backfill: identifierat "
        "historiskt tvåkolumnsformat."
    )
    print(
        "FI backfill: "
        f"{len(result)} observationer."
    )
    return result
def find_header_row(
    table: pd.DataFrame,
) -> int | None:
    """
    Försöker hitta rubrikraden.
    Historiska FI-filer kan innehålla
    blandade datatyper, så varje cell
    konverteras uttryckligen till sträng
    innan raden sätts ihop.
    """
    for header_row in range(
        min(30, len(table))
    ):
        raw_values = table.iloc[
            header_row
        ].tolist()
        values = [
            str(value).strip()
            for value in raw_values
        ]
        text = (
            " | ".join(values)
            .lower()
        )
        if (
            "emittentens namn" in text
            and "summa blankning" in text
        ):
            return header_row
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
        if (
            "issuer" in text
            and (
                "short" in text
                or "position" in text
            )
        ):
            return header_row
    return None
def print_table_diagnostic(
    table: pd.DataFrame,
    source_date: date,
) -> None:
    """Skriver diagnostik när formatet inte känns igen."""
    print(
        "FI backfill: okänt tabellformat "
        f"för {source_date}."
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
            for value in table.iloc[
                index
            ].tolist()
        ]
        print(
            f"  [{index}] "
            + " | ".join(values)
        )
def prepare_table(
    table: pd.DataFrame,
    source_date: date,
) -> pd.DataFrame:
    """
    Försöker omvandla en rå Excel-tabell
    till normaliserbara kolumner.
    """
    # FI:s äldre aggregatfiler, exempelvis
    # 2022-05-25, består av exakt två
    # kolumner och saknar rubrikrad.
    if table.shape[1] == 2:
        first_column = table.iloc[
            :, 0
        ]
        second_column = pd.to_numeric(
            table.iloc[:, 1],
            errors="coerce",
        )
        non_empty = (
            first_column.notna()
        )
        numeric_ratio = (
            second_column.notna().sum()
            / max(
                non_empty.sum(),
                1,
            )
        )
        if numeric_ratio >= 0.90:
            return (
                prepare_historical_two_column_table(
                    table,
                    source_date,
                )
            )
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
            "rubrikraden i FI:s "
            "historiska fil för "
            f"{source_date}."
        )
    candidate = [
        str(value).strip()
        for value in table.iloc[
            header_row
        ].tolist()
    ]
    result = (
        table
        .iloc[
            header_row + 1 :
        ]
        .copy()
    )
    result.columns = candidate
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
    """Läser en historisk FI-aggregatfil."""
    file_format = detect_file_format(
        data
    )
    print(
        "FI backfill: läser fil "
        f"{source_date}: "
        f"format={file_format}"
    )
    errors: list[str] = []
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
            raise
        except Exception as exc:
            errors.append(
                f"openpyxl: {exc}"
            )
            print(
                "FI backfill: "
                "parser=openpyxl "
                f"misslyckades: {exc}"
            )
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
    Före 2022-06-09 används FI:s
    kända veckovisa publiceringsdatum.
    Från 2022-06-09 används vardagar.
    Befintliga datum hoppas över för
    att processen ska vara idempotent.
    Ett saknat historiskt datum är
    normalt och räknas inte som ett fel.
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
    dates = historical_dates(
        start,
        end,
    )
    for source_date in dates:
        existing = output_path(
            source_date
        )
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
            f"Standard: {DEFAULT_START_DATE}."
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
