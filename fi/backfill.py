"""Självläkande backfill av FI:s aggregerade blankningsdata."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .config import (
    HEADERS,
    MANIFEST_PATH,
    RAW_DIR,
    SNAPSHOT_DIR,
)
from .errors import FIError
from .normalize import now_stockholm


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

DEFAULT_DELAY = 0.5


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


def output_path(
    source_date: date,
) -> Path:
    """
    Returnerar den kanoniska snapshot-filen
    för ett återställt FI-datum.
    """

    return (
        SNAPSHOT_DIR
        / (
            "fi_aggregate_"
            f"{source_date.isoformat()}"
            "_00-00-00.jsonl"
        )
    )


def snapshot_dates() -> set[date]:
    """
    Läser befintliga FI-datum från snapshots.

    Datumet hämtas från filnamnet.
    """

    dates: set[date] = set()

    for path in SNAPSHOT_DIR.glob(
        "fi_aggregate_*.jsonl"
    ):
        match = re.search(
            r"fi_aggregate_"
            r"(\d{4})-(\d{2})-(\d{2})",
            path.name,
        )

        if not match:
            continue

        try:
            dates.add(
                date(
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                )
            )
        except ValueError:
            continue

    return dates


def easter_sunday(year: int) -> date:
    """
    Beräknar påskdagen enligt
    Gregorian computus.
    """

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
    """
    Första lördagen mellan 20 och 26 juni.
    """

    current = date(
        year,
        6,
        20,
    )

    while current.weekday() != 5:
        current += timedelta(days=1)

    return current


def all_saints_day(year: int) -> date:
    """
    Första lördagen mellan 31 oktober
    och 6 november.
    """

    current = date(
        year,
        10,
        31,
    )

    while current.weekday() != 5:
        current += timedelta(days=1)

    return current


def swedish_holidays(year: int) -> set[date]:
    """
    Svenska allmänna helgdagar samt
    helgaftnar som behandlas som söndagar
    i svensk arbetsrätt.

    Vi använder dessa för att inte skapa
    falska FI-luckor på dagar då en publicering
    normalt inte ska förväntas.
    """

    easter = easter_sunday(year)

    holidays = {
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

        date(year, 12, 24),
        date(year, 12, 25),
        date(year, 12, 26),
        date(year, 12, 31),
    }

    return holidays


def is_expected_fi_day(
    value: date,
) -> bool:
    """
    Returnerar True när datumet är en vardag
    där FI-data kan förväntas.

    Helger och svenska helgdagar/helgaftnar
    räknas inte som luckor.
    """

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


def get_commoncrawl_indexes(
    session: requests.Session,
) -> list[str]:
    """Hämtar relevanta Common Crawl-index."""

    try:
        response = session.get(
            COMMONCRAWL_COLLECTIONS_URL,
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
        name = collection.get(
            "id",
            "",
        )

        if name.startswith(
            "CC-MAIN-"
        ):
            result.append(name)

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

    try:
        response = session.get(
            endpoint,
            params=params,
            timeout=60,
        )

        if response.status_code == 404:
            return []

        response.raise_for_status()

    except requests.RequestException as exc:
        print(
            "FI backfill: Common Crawl "
            f"{index_name} misslyckades: {exc}"
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


def parse_xlsx(
    content: bytes,
    source_date: date,
    source: str,
    source_url: str,
) -> list[dict]:
    """Läser en historisk FI-XLSX."""

    excel = pd.ExcelFile(
        io.BytesIO(content),
        engine="openpyxl",
    )

    frames: list[pd.DataFrame] = []

    for sheet in excel.sheet_names:
        frame = pd.read_excel(
            excel,
            sheet_name=sheet,
        )

        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise FIError(
            "FI-historik: Excel-filen "
            "innehåller inga tabeller."
        )

    frame = pd.concat(
        frames,
        ignore_index=True,
    )

    issuer_col = find_column(
        frame.columns,
        "Emittentens namn",
        "Namn på emittent",
        "Emittent",
        "Issuer",
        "Company",
    )

    lei_col = find_column(
        frame.columns,
        "Emittentens LEI-kod",
        "LEI",
        "Emittentens LEI",
    )

    position_col = find_column(
        frame.columns,
        "Summa blankning %",
        "Position i procent",
        "Position",
        "Aggregate short position",
        "Short position",
    )

    latest_date_col = find_column(
        frame.columns,
        "Positionsdatum senaste position",
        "Positionsdatum",
        "Latest position date",
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


def sha256_bytes(
    data: bytes,
) -> str:
    """SHA-256 för snapshot-data."""

    return hashlib.sha256(
        data
    ).hexdigest()


def register_snapshot(
    path: Path,
    records: list[dict],
) -> None:
    """
    Registrerar en återställd snapshot
    i FI:s manifest.
    """

    if not records:
        return

    try:
        manifest = json.loads(
            MANIFEST_PATH.read_text(
                encoding="utf-8"
            )
        )
    except (
        FileNotFoundError,
        json.JSONDecodeError,
    ):
        manifest = {
            "source": "FI",
            "dataset": (
                "aggregate_short_positions"
            ),
            "last_checked": None,
            "files": {},
            "snapshots": [],
        }

    snapshots = manifest.setdefault(
        "snapshots",
        [],
    )

    relative_path = path.relative_to(
        RAW_DIR
    ).as_posix()

    data = path.read_bytes()

    fetched_at = records[0].get(
        "fetched_at"
    )

    entry = {
        "file": relative_path,
        "fetched_at": fetched_at,
        "source_dates": [
            records[0]["source_date"]
        ],
        "observations": len(records),
        "sha256": sha256_bytes(data),
        "recovered": True,
    }

    snapshots = [
        item
        for item in snapshots
        if not (
            isinstance(item, dict)
            and item.get("file")
            == relative_path
        )
    ]

    snapshots.append(entry)

    snapshots.sort(
        key=lambda item: item.get(
            "fetched_at",
            "",
        )
    )

    manifest["snapshots"] = snapshots
    manifest["source"] = "FI"
    manifest["dataset"] = (
        "aggregate_short_positions"
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_snapshot(
    records: list[dict],
    source_date: date,
) -> Path:
    """Skriver en kanonisk datum-snapshot."""

    SNAPSHOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_path(
        source_date
    )

    if path.exists():
        return path

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

    register_snapshot(
        path,
        records,
    )

    return path


def recover_missing_dates(
    start: date | None = None,
    end: date | None = None,
    delay: float = DEFAULT_DELAY,
) -> tuple[int, int]:
    """
    Försöker återställa saknade FI-dagar.

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

    search_start = min(missing)
    search_end = max(missing)

    candidates: list[Candidate] = []

    print(
        "FI backfill: söker historiska "
        "FI-filer."
    )

    indexes = get_commoncrawl_indexes(
        session
    )

    for index_name in indexes[-12:]:
        found = query_commoncrawl(
            session,
            index_name,
            search_start,
            search_end,
        )

        if found:
            candidates.extend(found)

        time.sleep(delay)

    wayback = query_wayback(
        session,
        search_start,
        search_end,
    )

    candidates.extend(
        wayback
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

    recovered = 0
    unresolved = 0

    for source_date in missing:
        path = output_path(
            source_date
        )

        if path.exists():
            continue

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
                records = parse_xlsx(
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
                records,
                source_date,
            )

            print(
                "FI backfill: återställde "
                f"{source_date}: "
                f"{len(records)} observationer "
                f"-> {path}"
            )

            recovered += 1
            recovered_this_date = True

            break

        if not recovered_this_date:
            print(
                "FI backfill: kunde inte "
                f"återställa {source_date}."
            )

            unresolved += 1

        time.sleep(delay)

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
        help=(
            "Första datum."
        ),
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
        help=(
            "Paus mellan externa anrop."
        ),
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
