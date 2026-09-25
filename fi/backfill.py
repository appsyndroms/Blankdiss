"""Självläkande backfill av FI:s aggregerade blankningsdata."""

from __future__ import annotations

import argparse
import io
import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
import requests

from .config import (
    FI_AGGREGATE_TIMEOUT,
    FI_AGGREGATE_URL,
    HEADERS,
)
from .errors import FIError
from .normalize import now_stockholm
from .storage import load_manifest, write_snapshot


DEFAULT_START_DATE = date(2022, 6, 9)

COMMONCRAWL_COLLECTIONS_URL = (
    "https://index.commoncrawl.org/collinfo.json"
)

COMMONCRAWL_INDEX_URL = (
    "https://index.commoncrawl.org/"
)

WAYBACK_CDX_URL = (
    "https://web.archive.org/cdx/search/cdx"
)

DEFAULT_DELAY = 1.0
COMMONCRAWL_RETRIES = 3
COMMONCRAWL_RETRY_STATUSES = {
    502,
    503,
    504,
}

# FI:s aggregat-endpoint har historiskt testats med
# flera möjliga parameternamn. Vi provar dem i turordning
# och accepterar endast ett svar som faktiskt innehåller
# det efterfrågade positionsdatumet.
FI_DATE_PARAMETER_NAMES = (
    "date",
    "datum",
    "source_date",
    "position_date",
    "positionDate",
    "positionsdatum",
)


@dataclass(frozen=True)
class Candidate:
    """En möjlig historisk FI-fil."""

    url: str
    source_date: date
    source: str
    archive_timestamp: str | None = None


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


def snapshot_dates() -> set[date]:
    """
    Läser befintliga FI-datum från manifestet.

    source_dates är det kanoniska sättet att
    avgöra vilka FI-datum som redan finns.
    """

    manifest = load_manifest()

    dates: set[date] = set()

    snapshots = manifest.get(
        "snapshots",
        [],
    )

    if not isinstance(
        snapshots,
        list,
    ):
        return dates

    for snapshot in snapshots:
        if not isinstance(
            snapshot,
            dict,
        ):
            continue

        source_dates = snapshot.get(
            "source_dates",
            [],
        )

        if not isinstance(
            source_dates,
            list,
        ):
            continue

        for value in source_dates:
            if not value:
                continue

            try:
                dates.add(
                    date.fromisoformat(
                        str(value)
                    )
                )
            except ValueError:
                continue

    return dates


def easter_sunday(year: int) -> date:
    """Beräknar påskdagen enligt Gregorian computus."""

    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (
        19 * a
        + b
        - d
        - g
        + 15
    ) % 30
    i = c // 4
    k = c % 4
    l = (
        32
        + 2 * e
        + 2 * i
        - h
        - k
    ) % 7
    m = (
        a
        + 11 * h
        + 22 * l
    ) // 451

    month = (
        h + l - 7 * m + 114
    ) // 31

    day = (
        (h + l - 7 * m + 114)
        % 31
    ) + 1

    return date(
        year,
        month,
        day,
    )


def midsummer_day(year: int) -> date:
    """Första lördagen mellan 20 och 26 juni."""

    current = date(
        year,
        6,
        20,
    )

    while current.weekday() != 5:
        current += timedelta(days=1)

    return current


def all_saints_day(year: int) -> date:
    """Första lördagen mellan 31 oktober och 6 november."""

    current = date(
        year,
        10,
        31,
    )

    while current.weekday() != 5:
        current += timedelta(days=1)

    return current


def swedish_holidays(
    year: int,
) -> set[date]:
    """
    Svenska allmänna helgdagar.

    Dessa används endast för att undvika att
    behandla helgdagar som saknade FI-dagar.
    """

    easter = easter_sunday(year)

    return {
        date(year, 1, 1),
        date(year, 1, 6),
        easter - timedelta(days=2),
        easter - timedelta(days=1),
        easter,
        easter + timedelta(days=1),
        date(year, 5, 1),
        easter + timedelta(days=39),
        date(year, 6, 6),
        midsummer_day(year),
        all_saints_day(year),
        date(year, 12, 25),
        date(year, 12, 26),
    }


def is_expected_fi_day(
    value: date,
) -> bool:
    """Returnerar True för vardag som inte är allmän helgdag."""

    if value.weekday() >= 5:
        return False

    if value in swedish_holidays(
        value.year
    ):
        return False

    return True


def missing_weekdays(
    start: date,
    end: date,
) -> list[date]:
    """Returnerar saknade förväntade FI-dagar."""

    existing = snapshot_dates()

    result: list[date] = []

    current = start

    while current <= end:
        if (
            is_expected_fi_day(current)
            and current not in existing
        ):
            result.append(current)

        current += timedelta(days=1)

    return result


def normalize_text(
    value: object,
) -> str:
    """Normaliserar text."""

    if value is None:
        return ""

    text = str(value).strip()

    return re.sub(
        r"\s+",
        " ",
        text,
    )


def normalize_header(
    value: object,
) -> str:
    """Normaliserar kolumnnamn."""

    text = normalize_text(
        value
    ).lower()

    replacements = {
        "å": "a",
        "ä": "a",
        "ö": "o",
        "é": "e",
        "á": "a",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )

    return text.strip("_")


def extract_date_from_url(
    url: str,
) -> date | None:
    """Hittar FI-datum i en historisk fil-URL."""

    match = re.search(
        r"aggregerade[-_]blankningspositioner[-_]"
        r"(\d{4})[-_](\d{2})[-_](\d{2})",
        url,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    try:
        return date(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )

    except ValueError:
        return None


def is_fi_aggregate_url(
    url: str,
) -> bool:
    """Kontrollerar att URL är en FI-aggregatfil."""

    lowered = url.lower()

    return (
        "fi.se/" in lowered
        and "aggregerade-blankningspositioner-"
        in lowered
        and lowered.endswith(".xlsx")
    )


def commoncrawl_index_sort_key(
    index_name: str,
) -> tuple[int, int]:
    """Returnerar (år, vecka) för Common Crawl-index."""

    match = re.fullmatch(
        r"CC-MAIN-(\d{4})-(\d{2})",
        index_name,
    )

    if not match:
        return (-1, -1)

    return (
        int(match.group(1)),
        int(match.group(2)),
    )


def get_commoncrawl_indexes(
    session: requests.Session,
) -> list[str]:
    """Hämtar Common Crawl-index, nyaste först."""

    headers = {
        **HEADERS,
        "User-Agent": (
            "Blankdiss/1.0 "
            "(FI historical backfill; "
            "https://github.com/appsyndroms/Blankdiss)"
        ),
    }

    try:
        response = session.get(
            COMMONCRAWL_COLLECTIONS_URL,
            headers=headers,
            timeout=60,
        )

        response.raise_for_status()

        collections = response.json()

    except (
        requests.RequestException,
        ValueError,
    ) as exc:
        print(
            "FI backfill: Common Crawl-index "
            f"kunde inte hämtas: {exc}"
        )

        return []

    result: list[str] = []

    for collection in collections:
        if not isinstance(
            collection,
            dict,
        ):
            continue

        name = collection.get(
            "id",
            "",
        )

        if re.fullmatch(
            r"CC-MAIN-\d{4}-\d{2}",
            name,
        ):
            result.append(name)

    result.sort(
        key=commoncrawl_index_sort_key,
        reverse=True,
    )

    return result


def query_commoncrawl(
    session: requests.Session,
    index_name: str,
    start: date,
    end: date,
) -> list[Candidate]:
    """Söker FI-filer i Common Crawl."""

    pattern = (
        "https://www.fi.se/contentassets/*/"
        "aggregerade-blankningspositioner-*.xlsx"
    )

    endpoint = (
        f"{COMMONCRAWL_INDEX_URL}"
        f"{index_name}-index"
    )

    params = {
        "url": pattern,
        "output": "json",
        "filter": "status:200",
        "collapse": "urlkey",
    }

    headers = {
        **HEADERS,
        "User-Agent": (
            "Blankdiss/1.0 "
            "(FI historical backfill; "
            "https://github.com/appsyndroms/Blankdiss)"
        ),
    }

    for attempt in range(
        COMMONCRAWL_RETRIES
    ):
        try:
            response = session.get(
                endpoint,
                params=params,
                headers=headers,
                timeout=60,
            )

            if response.status_code == 404:
                return []

            if (
                response.status_code
                in COMMONCRAWL_RETRY_STATUSES
            ):
                if (
                    attempt
                    < COMMONCRAWL_RETRIES - 1
                ):
                    wait_seconds = (
                        2 ** attempt
                    )

                    print(
                        "FI backfill: Common Crawl "
                        f"{index_name} gav "
                        f"{response.status_code}, "
                        f"försöker igen om "
                        f"{wait_seconds}s."
                    )

                    time.sleep(
                        wait_seconds
                    )

                    continue

                print(
                    "FI backfill: Common Crawl "
                    f"{index_name} misslyckades "
                    f"efter {COMMONCRAWL_RETRIES} "
                    f"försök: HTTP "
                    f"{response.status_code}"
                )

                return []

            response.raise_for_status()

            break

        except requests.RequestException as exc:
            if (
                attempt
                < COMMONCRAWL_RETRIES - 1
            ):
                wait_seconds = (
                    2 ** attempt
                )

                print(
                    "FI backfill: Common Crawl "
                    f"{index_name} misslyckades: "
                    f"{exc}. "
                    f"Försöker igen om "
                    f"{wait_seconds}s."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            print(
                "FI backfill: Common Crawl "
                f"{index_name} misslyckades "
                f"efter {COMMONCRAWL_RETRIES} "
                f"försök: {exc}"
            )

            return []

    candidates: list[Candidate] = []

    for line in response.text.splitlines():
        if not line.strip():
            continue

        try:
            item = json.loads(line)

        except json.JSONDecodeError:
            continue

        url = item.get(
            "url",
            "",
        )

        if not is_fi_aggregate_url(
            url
        ):
            continue

        source_date = extract_date_from_url(
            url
        )

        if source_date is None:
            continue

        if not (
            start
            <= source_date
            <= end
        ):
            continue

        candidates.append(
            Candidate(
                url=url,
                source_date=source_date,
                source="commoncrawl",
                archive_timestamp=item.get(
                    "timestamp"
                ),
            )
        )

    return candidates


def query_wayback(
    session: requests.Session,
    start: date,
    end: date,
) -> list[Candidate]:
    """Söker historiska FI-filer i Wayback."""

    pattern = (
        "https://www.fi.se/contentassets/*/"
        "aggregerade-blankningspositioner-*.xlsx"
    )

    params = {
        "url": pattern,
        "output": "json",
        "filter": "statuscode:200",
        "fl": (
            "timestamp,"
            "original,"
            "statuscode"
        ),
        "collapse": "urlkey",
    }

    try:
        response = session.get(
            WAYBACK_CDX_URL,
            params=params,
            timeout=60,
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        print(
            "FI backfill: Wayback "
            f"misslyckades: {exc}"
        )

        return []

    try:
        rows = response.json()

    except ValueError:
        return []

    candidates: list[Candidate] = []

    for row in rows:
        if not isinstance(
            row,
            list,
        ):
            continue

        if len(row) < 2:
            continue

        timestamp = row[0]
        url = row[1]

        if not is_fi_aggregate_url(
            url
        ):
            continue

        source_date = extract_date_from_url(
            url
        )

        if source_date is None:
            continue

        if not (
            start
            <= source_date
            <= end
        ):
            continue

        candidates.append(
            Candidate(
                url=url,
                source_date=source_date,
                source="wayback",
                archive_timestamp=timestamp,
            )
        )

    return candidates


def download_candidate(
    session: requests.Session,
    candidate: Candidate,
) -> bytes | None:
    """Hämtar en kandidat direkt eller från arkiv."""

    urls: list[str] = []

    if candidate.source == "commoncrawl":
        urls.append(candidate.url)

    elif candidate.source == "wayback":
        if candidate.archive_timestamp:
            urls.append(
                "https://web.archive.org/web/"
                f"{candidate.archive_timestamp}"
                "id_/"
                f"{candidate.url}"
            )

    urls.append(candidate.url)

    for url in urls:
        try:
            response = session.get(
                url,
                headers=HEADERS,
                timeout=60,
            )

        except requests.RequestException:
            continue

        if response.status_code != 200:
            continue

        data = response.content

        if len(data) < 1000:
            continue

        first = data[:1000].lower()

        if (
            b"<html" in first
            or b"<!doctype" in first
        ):
            continue

        return data

    return None


def find_column(
    columns: Iterable[object],
    *wanted: str,
) -> str | None:
    """Hittar en kolumn utifrån normaliserat namn."""

    normalized = {
        normalize_header(column): str(column)
        for column in columns
    }

    for wanted_name in wanted:
        wanted_normalized = normalize_header(
            wanted_name
        )

        for (
            normalized_name,
            original,
        ) in normalized.items():

            if (
                normalized_name
                == wanted_normalized
                or wanted_normalized
                in normalized_name
                or normalized_name
                in wanted_normalized
            ):
                return original

    return None


def parse_position(
    value: object,
) -> float | None:
    """Tolkar blankningsprocent."""

    if value is None:
        return None

    if isinstance(
        value,
        (int, float),
    ):
        if pd.isna(value):
            return None

        number = float(value)

        if 0 < abs(number) < 1:
            number *= 100

        return number

    text = normalize_text(value)

    if not text:
        return None

    text = (
        text
        .replace("%", "")
        .replace(" ", "")
        .replace(",", ".")
    )

    try:
        number = float(text)

    except ValueError:
        return None

    if 0 < abs(number) < 1:
        number *= 100

    return number


def parse_position_date(
    value: object,
) -> str | None:
    """Tolkar positionsdatum."""

    if value is None:
        return None

    if isinstance(
        value,
        datetime,
    ):
        return value.date().isoformat()

    if isinstance(
        value,
        date,
    ):
        return value.isoformat()

    text = normalize_text(value)

    if not text:
        return None

    parsed = pd.to_datetime(
        text,
        errors="coerce",
        dayfirst=False,
    )

    if pd.isna(parsed):
        return None

    return parsed.date().isoformat()


def parse_excel_records(
    content: bytes,
    source_date: date,
    source: str,
    source_url: str,
) -> list[dict]:
    """
    Läser FI:s Excel/ODS-data.

    FI:s aktuella aggregat-endpoint returnerar ODS,
    medan arkiverade historiska filer kan vara XLSX.
    Därför provar vi ODS först och XLSX därefter.
    """

    excel: pd.ExcelFile

    try:
        excel = pd.ExcelFile(
            io.BytesIO(content),
            engine="odf",
        )

        skiprows = 6

        frames: list[pd.DataFrame] = []

        for sheet in excel.sheet_names:
            frame = pd.read_excel(
                excel,
                sheet_name=sheet,
                skiprows=skiprows,
                header=None,
            )

            if not frame.empty:
                frames.append(frame)

    except Exception:
        try:
            excel = pd.ExcelFile(
                io.BytesIO(content),
                engine="openpyxl",
            )

            frames = []

            for sheet in excel.sheet_names:
                frame = pd.read_excel(
                    excel,
                    sheet_name=sheet,
                )

                if not frame.empty:
                    frames.append(frame)

        except Exception as exc:
            raise FIError(
                "FI-historik: kunde inte läsa "
                f"Excel/ODS-data: {exc}"
            ) from exc

    if not frames:
        raise FIError(
            "FI-historik: filen innehåller "
            "inga tabeller."
        )

    frame = pd.concat(
        frames,
        ignore_index=True,
    )

    # ODS-endpointen har ingen vanlig header
    # efter skiprows=6. Arkiverade XLSX-filer
    # har däremot normalt header.
    #
    # Om första raden fortfarande innehåller
    # rubriker försöker vi använda den.
    if all(
        normalize_header(value)
        for value in frame.iloc[0].tolist()
    ):
        possible_headers = {
            normalize_header(value)
            for value in frame.iloc[0].tolist()
        }

        if (
            any(
                "emittent" in value
                for value in possible_headers
            )
            or "lei" in possible_headers
            or any(
                "summa_blankning" in value
                for value in possible_headers
            )
        ):
            frame = frame.iloc[1:].copy()

    # För FI:s ODS-format är de första fyra
    # kolumnerna:
    #   emittent
    #   lei
    #   summa blankning
    #   positionsdatum
    #
    # Det formatet har visat sig vara stabilt.
    if len(frame.columns) >= 4:
        first_four = frame.iloc[:, :4].copy()

        first_four.columns = [
            "issuer",
            "lei",
            "short_interest_pct",
            "position_date",
        ]

        frame = first_four

    issuer_col = find_column(
        frame.columns,
        "Emittentens namn",
        "Namn på emittent",
        "Emittent",
        "Issuer",
        "Company",
        "issuer",
    )

    lei_col = find_column(
        frame.columns,
        "Emittentens LEI-kod",
        "LEI",
        "Emittentens LEI",
        "lei",
    )

    position_col = find_column(
        frame.columns,
        "Summa blankning %",
        "Position i procent",
        "Position",
        "Aggregate short position",
        "Short position",
        "short_interest_pct",
    )

    latest_date_col = find_column(
        frame.columns,
        "Positionsdatum senaste position",
        "Positionsdatum",
        "Latest position date",
        "position_date",
    )

    if issuer_col is None:
        raise FIError(
            "FI-historik: kunde inte hitta "
            "emittentkolumn."
        )

    if position_col is None:
        raise FIError(
            "FI-historik: kunde inte hitta "
            "blankningskolumn."
        )

    records: list[dict] = []

    for _, row in frame.iterrows():
        issuer = normalize_text(
            row.get(issuer_col)
        )

        if not issuer:
            continue

        position = parse_position(
            row.get(position_col)
        )

        if position is None:
            continue

        lei = (
            normalize_text(
                row.get(lei_col)
            )
            if lei_col
            else ""
        )

        latest_position_date = (
            parse_position_date(
                row.get(
                    latest_date_col
                )
            )
            if latest_date_col
            else None
        )

        records.append(
            {
                "snapshot_date": (
                    source_date.isoformat()
                ),
                "source_date": (
                    source_date.isoformat()
                ),
                "issuer": issuer,
                "lei": lei or None,
                "short_interest_pct": round(
                    position,
                    6,
                ),
                "position_date": (
                    latest_position_date
                ),
                "fetched_at": (
                    now_stockholm()
                    .isoformat(
                        timespec="seconds"
                    )
                ),
                "source": source,
                "source_url": source_url,
            }
        )

    return records


def parse_xlsx(
    content: bytes,
    source_date: date,
    source: str,
    source_url: str,
) -> list[dict]:
    """Bakåtkompatibelt namn för arkiverade XLSX-filer."""

    return parse_excel_records(
        content,
        source_date,
        source,
        source_url,
    )


def validate_records(
    records: list[dict],
    source_date: date,
) -> None:
    """Validerar en återställd FI-snapshot."""

    if not records:
        raise FIError(
            f"FI-historik {source_date}: "
            "inga observationer."
        )

    issuers = [
        record["issuer"]
        for record in records
    ]

    if len(issuers) != len(
        set(issuers)
    ):
        raise FIError(
            f"FI-historik {source_date}: "
            "dubbletter av emittenter."
        )

    for record in records:
        value = record[
            "short_interest_pct"
        ]

        if value < 0 or value > 100:
            raise FIError(
                f"FI-historik {source_date}: "
                f"ogiltig blankning {value}."
            )


def extract_position_dates(
    content: bytes,
) -> set[date]:
    """
    Läser alla positionsdatum ur ett FI-aggregatsvar.

    Används innan vi accepterar ett svar från
    FI:s historiska endpoint. Det räcker alltså
    inte att endpointen returnerar HTTP 200.
    """

    excel: pd.ExcelFile

    try:
        excel = pd.ExcelFile(
            io.BytesIO(content),
            engine="odf",
        )

        frames: list[pd.DataFrame] = []

        for sheet in excel.sheet_names:
            frame = pd.read_excel(
                excel,
                sheet_name=sheet,
                skiprows=6,
                header=None,
            )

            if not frame.empty:
                frames.append(frame)

    except Exception:
        try:
            excel = pd.ExcelFile(
                io.BytesIO(content),
                engine="openpyxl",
            )

            frames = []

            for sheet in excel.sheet_names:
                frame = pd.read_excel(
                    excel,
                    sheet_name=sheet,
                )

                if not frame.empty:
                    frames.append(frame)

        except Exception:
            return set()

    if not frames:
        return set()

    frame = pd.concat(
        frames,
        ignore_index=True,
    )

    if len(frame.columns) < 4:
        return set()

    values = frame.iloc[:, 3]

    parsed = pd.to_datetime(
        values,
        errors="coerce",
        dayfirst=False,
    )

    return {
        value.date()
        for value in parsed.dropna()
    }


def fetch_fi_historical_endpoint(
    session: requests.Session,
    source_date: date,
) -> bytes | None:
    """
    Försöker hämta ett historiskt FI-aggregat
    direkt från FI:s endpoint.

    Ett svar accepteras endast om:
      1. HTTP-status är 200
      2. svaret går att läsa som Excel/ODS
      3. det efterfrågade datumet faktiskt finns
         i positionsdatumkolumnen.

    Detta är viktigt eftersom FI kan ignorera en
    okänd datumparameter och returnera dagens fil.
    """

    date_text = source_date.isoformat()

    print(
        "FI backfill: provar FI:s historiska "
        f"aggregat-endpoint för {source_date}."
    )

    for parameter_name in (
        FI_DATE_PARAMETER_NAMES
    ):
        params = {
            parameter_name: date_text,
        }

        try:
            response = session.get(
                FI_AGGREGATE_URL,
                params=params,
                headers=HEADERS,
                timeout=FI_AGGREGATE_TIMEOUT,
            )

        except requests.RequestException as exc:
            print(
                "FI backfill: "
                f"{parameter_name} misslyckades: "
                f"{exc}"
            )
            continue

        print(
            "FI backfill: "
            f"{parameter_name} -> "
            f"HTTP {response.status_code}, "
            f"{len(response.content)} bytes."
        )

        if response.status_code != 200:
            continue

        content = response.content

        if len(content) < 1000:
            print(
                "FI backfill: svaret är för litet."
            )
            continue

        position_dates = extract_position_dates(
            content
        )

        if source_date not in position_dates:
            print(
                "FI backfill: "
                f"{parameter_name} gav inget "
                f"positionsdatum {source_date}."
            )
            continue

        print(
            "FI backfill: HISTORIK FUNNEN via "
            f"{parameter_name}={date_text}."
        )

        return content

    print(
        "FI backfill: FI:s historiska "
        f"endpoint gav inget verifierat svar "
        f"för {source_date}."
    )

    return None


def recover_from_fi_endpoint(
    session: requests.Session,
    missing: list[date],
    delay: float,
) -> set[date]:
    """
    Försöker återställa saknade dagar direkt från FI.

    Returnerar de datum som faktiskt sparades.
    """

    recovered_dates: set[date] = set()

    for source_date in missing:
        content = fetch_fi_historical_endpoint(
            session,
            source_date,
        )

        if content is None:
            time.sleep(delay)
            continue

        try:
            records = parse_excel_records(
                content,
                source_date,
                "fi_endpoint",
                FI_AGGREGATE_URL,
            )

            validate_records(
                records,
                source_date,
            )

        except (
            FIError,
            ValueError,
            KeyError,
        ) as exc:
            print(
                "FI backfill: FI-endpointets "
                f"data kunde inte valideras "
                f"för {source_date}: {exc}"
            )

            time.sleep(delay)
            continue

        path = write_snapshot(
            records
        )

        print(
            "FI backfill: återställde "
            f"{source_date} direkt från FI: "
            f"{len(records)} observationer "
            f"-> {path}"
        )

        recovered_dates.add(
            source_date
        )

        time.sleep(delay)

    return recovered_dates


def recover_missing_dates(
    start: date | None = None,
    end: date | None = None,
    delay: float = DEFAULT_DELAY,
) -> tuple[int, int]:
    """
    Försöker återställa saknade FI-dagar.

    Ordning:

    1. FI:s eget historiska aggregat-endpoint.
    2. Common Crawl.
    3. Wayback.

    Varje återställd dag sparas genom samma
    write_snapshot() som används av dagens
    FI-hämtning.

    Returnerar:
        (antal återställda dagar,
         antal dagar som fortfarande saknas)
    """

    existing = snapshot_dates()

    if start is None:
        if existing:
            start = (
                min(existing)
                + timedelta(days=1)
            )
        else:
            start = DEFAULT_START_DATE

    if end is None:
        end = (
            now_stockholm().date()
            - timedelta(days=1)
        )

    if start > end:
        return 0, 0

    missing = missing_weekdays(
        start,
        end,
    )

    if not missing:
        print(
            "FI backfill: inga saknade "
            "förväntade FI-dagar."
        )

        return 0, 0

    print(
        "FI backfill: saknade "
        "förväntade FI-dagar:"
    )

    for value in missing:
        print(
            f"  - {value}"
        )

    session = requests.Session()

    session.headers.update(
        HEADERS
    )

    # ---------------------------------------------------------
    # STEG 1: FI:s eget endpoint
    # ---------------------------------------------------------

    print()
    print(
        "FI backfill: steg 1/3 - "
        "provar FI:s eget aggregat-endpoint."
    )

    recovered_dates = recover_from_fi_endpoint(
        session,
        missing,
        delay,
    )

    remaining = [
        value
        for value in missing
        if value not in recovered_dates
    ]

    if not remaining:
        return (
            len(recovered_dates),
            0,
        )

    print()
    print(
        "FI backfill: kvar efter FI-endpoint: "
        f"{len(remaining)}."
    )

    # ---------------------------------------------------------
    # STEG 2: Common Crawl
    # ---------------------------------------------------------

    search_start = min(remaining)
    search_end = max(remaining)

    candidates: list[Candidate] = []

    print()
    print(
        "FI backfill: steg 2/3 - "
        "söker Common Crawl."
    )

    indexes = get_commoncrawl_indexes(
        session
    )

    print(
        "FI backfill: Common Crawl-index "
        f"att söka: {len(indexes)}"
    )

    for index_name in indexes[:12]:
        print(
            "FI backfill: söker Common Crawl "
            f"{index_name}."
        )

        found = query_commoncrawl(
            session,
            index_name,
            search_start,
            search_end,
        )

        if found:
            print(
                "FI backfill: "
                f"{index_name} gav "
                f"{len(found)} kandidater."
            )

            candidates.extend(
                found
            )

        time.sleep(delay)

        found_dates = {
            candidate.source_date
            for candidate in candidates
        }

        if all(
            value in found_dates
            for value in remaining
        ):
            print(
                "FI backfill: alla kvarvarande "
                "datum hittades i Common Crawl."
            )
            break

    # ---------------------------------------------------------
    # STEG 3: Wayback
    # ---------------------------------------------------------

    unresolved_dates = {
        value
        for value in remaining
        if not any(
            candidate.source_date == value
            for candidate in candidates
        )
    }

    if unresolved_dates:
        print()
        print(
            "FI backfill: steg 3/3 - "
            "söker kvarvarande datum i Wayback."
        )

        candidates.extend(
            query_wayback(
                session,
                min(unresolved_dates),
                max(unresolved_dates),
            )
        )

    by_date: dict[
        date,
        list[Candidate],
    ] = {}

    for candidate in candidates:
        by_date.setdefault(
            candidate.source_date,
            [],
        ).append(candidate)

    recovered_archive = 0
    unresolved = 0

    for source_date in remaining:
        candidates_for_date = (
            by_date.get(
                source_date,
                [],
            )
        )

        candidates_for_date.sort(
            key=lambda candidate: (
                candidate.source
                != "commoncrawl"
            )
        )

        recovered_this_date = False

        for candidate in candidates_for_date:
            print(
                "FI backfill: försöker "
                f"{source_date} via "
                f"{candidate.source}."
            )

            content = download_candidate(
                session,
                candidate,
            )

            if content is None:
                continue

            try:
                records = parse_excel_records(
                    content,
                    source_date,
                    candidate.source,
                    candidate.url,
                )

                validate_records(
                    records,
                    source_date,
                )

            except (
                FIError,
                ValueError,
                KeyError,
            ) as exc:
                print(
                    "FI backfill: fil kunde "
                    f"inte valideras: {exc}"
                )
                continue

            path = write_snapshot(
                records
            )

            print(
                "FI backfill: återställde "
                f"{source_date}: "
                f"{len(records)} observationer "
                f"-> {path}"
            )

            recovered_archive += 1
            recovered_this_date = True

            break

        if not recovered_this_date:
            print(
                "FI backfill: kunde inte "
                f"återställa {source_date}."
            )

            unresolved += 1

        time.sleep(delay)

    recovered = (
        len(recovered_dates)
        + recovered_archive
    )

    return recovered, unresolved


def parse_args() -> argparse.Namespace:
    """Parsar CLI-argument."""

    parser = argparse.ArgumentParser(
        description=(
            "Återställer saknade vardagar "
            "i FI:s aggregerade "
            "blankningshistorik."
        )
    )

    parser.add_argument(
        "--from",
        dest="start",
        type=parse_date,
        default=None,
        help="Första datum.",
    )

    parser.add_argument(
        "--to",
        dest="end",
        type=parse_date,
        default=None,
        help=(
            "Sista datum. Standard är "
            "gårdagen."
        ),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Paus mellan externa anrop.",
    )

    return parser.parse_args()


def main() -> int:
    """CLI-entrypoint."""

    args = parse_args()

    recovered, unresolved = (
        recover_missing_dates(
            start=args.start,
            end=args.end,
            delay=args.delay,
        )
    )

    print()
    print(
        "FI backfill klart:"
    )
    print(
        f"  återställda dagar: {recovered}"
    )
    print(
        f"  kvarvarande luckor: {unresolved}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
