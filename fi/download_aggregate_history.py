from __future__ import annotations
import io
import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable
import pandas as pd
import requests
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
START_DATE = date(2022, 5, 25)
END_DATE = date.today()
RAW_DIR = Path("data/raw/fi/aggregate/historical")
PROCESSED_DIR = Path("data/processed/fi/aggregate")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Blankdiss/1.0 "
            "(historical FI aggregate research; "
            "https://github.com/appsyndroms/Blankdiss)"
        )
    }
)
TIMEOUT = 60
# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    url: str
    snapshot_date: date
    source: str
    archive_timestamp: str | None = None
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text
def normalize_header(value: object) -> str:
    text = normalize_text(value).lower()
    replacements = {
        "å": "a",
        "ä": "a",
        "ö": "o",
        "é": "e",
        "á": "a",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")
def extract_date_from_filename(url: str) -> date | None:
    """
    Finds:
        aggregerade-blankningspositioner-2022-06-08.xlsx
    """
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
def in_range(value: date) -> bool:
    return START_DATE <= value <= END_DATE
def is_fi_aggregate_url(url: str) -> bool:
    lowered = url.lower()
    return (
        "fi.se/" in lowered
        and "aggregerade-blankningspositioner-" in lowered
        and lowered.endswith(".xlsx")
    )
# ---------------------------------------------------------------------------
# Common Crawl
# ---------------------------------------------------------------------------
def get_commoncrawl_indexes() -> list[str]:
    """
    Returns Common Crawl index names.
    We keep all indexes whose crawl period can overlap our requested
    historical range instead of arbitrarily taking only the newest 20.
    """
    url = "https://index.commoncrawl.org/collinfo.json"
    response = SESSION.get(
        url,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    collections = response.json()
    indexes = []
    for collection in collections:
        name = collection.get("id", "")
        if not name.startswith("CC-MAIN-"):
            continue
        match = re.match(
            r"CC-MAIN-(\d{4})-",
            name,
        )
        if not match:
            continue
        year = int(match.group(1))
        if year >= START_DATE.year:
            indexes.append(name)
    return indexes
def query_commoncrawl(index_name: str) -> list[Candidate]:
    """
    Searches Common Crawl for FI historical aggregate XLSX files.
    The wildcard is intentional because FI's contentassets GUID is
    not known in advance.
    """
    pattern = (
        "fi.se/contentassets/*/"
        "aggregerade-blankningspositioner-*.xlsx"
    )
    endpoint = (
        f"https://index.commoncrawl.org/"
        f"{index_name}-index"
    )
    params = {
        "url": pattern,
        "output": "json",
        "filter": "status:200",
        "collapse": "urlkey",
    }
    try:
        response = SESSION.get(
            endpoint,
            params=params,
            timeout=TIMEOUT,
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
    except requests.RequestException as exc:
        print(
            f"[Common Crawl] Kunde inte läsa {index_name}: "
            f"{exc}"
        )
        return []
    candidates: list[Candidate] = []
    for line in response.text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        original = item.get("url", "")
        if not is_fi_aggregate_url(original):
            continue
        snapshot_date = extract_date_from_filename(
            original
        )
        if snapshot_date is None:
            continue
        if not in_range(snapshot_date):
            continue
        candidates.append(
            Candidate(
                url=original,
                snapshot_date=snapshot_date,
                source="commoncrawl",
                archive_timestamp=item.get("timestamp"),
            )
        )
    return candidates
# ---------------------------------------------------------------------------
# Wayback Machine
# ---------------------------------------------------------------------------
def query_wayback(url_pattern: str) -> list[Candidate]:
    """
    Searches the Wayback CDX endpoint.
    We query the wildcard path because the FI contentassets GUID is
    not known.
    """
    endpoint = "https://web.archive.org/cdx/search/cdx"
    params = {
        "url": url_pattern,
        "output": "json",
        "filter": "statuscode:200",
        "fl": "timestamp,original,statuscode",
        "collapse": "urlkey",
    }
    try:
        response = SESSION.get(
            endpoint,
            params=params,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(
            f"[Wayback] Kunde inte söka: {exc}"
        )
        return []
    try:
        rows = response.json()
    except json.JSONDecodeError:
        return []
    if not rows:
        return []
    candidates: list[Candidate] = []
    for row in rows:
        if not isinstance(row, list):
            continue
        if len(row) < 2:
            continue
        timestamp = row[0]
        original = row[1]
        if not is_fi_aggregate_url(original):
            continue
        snapshot_date = extract_date_from_filename(
            original
        )
        if snapshot_date is None:
            continue
        if not in_range(snapshot_date):
            continue
        candidates.append(
            Candidate(
                url=original,
                snapshot_date=snapshot_date,
                source="wayback",
                archive_timestamp=timestamp,
            )
        )
    return candidates
# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_direct(
    candidate: Candidate,
) -> bytes | None:
    """
    Attempts a direct FI download.
    """
    try:
        response = SESSION.get(
            candidate.url,
            timeout=TIMEOUT,
        )
        if response.status_code != 200:
            return None
        if len(response.content) < 1000:
            return None
        first_bytes = response.content[:1000].lower()
        if (
            b"<html" in first_bytes
            or b"<!doctype" in first_bytes
        ):
            return None
        return response.content
    except requests.RequestException:
        return None
def download_wayback(
    candidate: Candidate,
) -> bytes | None:
    """
    Downloads the archived original file from Wayback.
    id_ asks Wayback for the original binary rather than
    the replay wrapper.
    """
    if not candidate.archive_timestamp:
        return None
    archived_url = (
        "https://web.archive.org/web/"
        f"{candidate.archive_timestamp}id_/"
        f"{candidate.url}"
    )
    try:
        response = SESSION.get(
            archived_url,
            timeout=TIMEOUT,
        )
        if response.status_code != 200:
            return None
        if len(response.content) < 1000:
            return None
        return response.content
    except requests.RequestException:
        return None
# ---------------------------------------------------------------------------
# Excel parsing
# ---------------------------------------------------------------------------
def find_column(
    columns: Iterable[object],
    *wanted: str,
) -> str | None:
    normalized = {
        normalize_header(column): str(column)
        for column in columns
    }
    for wanted_name in wanted:
        wanted_normalized = normalize_header(
            wanted_name
        )
        for normalized_name, original in normalized.items():
            if (
                normalized_name == wanted_normalized
                or wanted_normalized in normalized_name
                or normalized_name in wanted_normalized
            ):
                return original
    return None
def parse_position(
    value: object,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
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
        text.replace("%", "")
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
def parse_date(
    value: object,
) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = normalize_text(value)
    if not text:
        return None
    parsed = pd.to_datetime(
        text,
        errors="coerce",
        dayfirst=True,
    )
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()
def parse_xlsx(
    content: bytes,
    snapshot_date: date,
    source: str,
    source_url: str,
) -> list[dict]:
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
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        raise ValueError(
            "Excel-filen innehåller inga tabeller."
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
        raise ValueError(
            f"Hittade inte emittentkolumn. "
            f"Kolumner: {list(frame.columns)}"
        )
    if position_col is None:
        raise ValueError(
            f"Hittade inte aggregatkolumn. "
            f"Kolumner: {list(frame.columns)}"
        )
    records = []
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
            parse_date(
                row.get(latest_date_col)
            )
            if latest_date_col
            else None
        )
        records.append(
            {
                "snapshot_date": snapshot_date.isoformat(),
                "issuer": issuer,
                "lei": lei or None,
                "position": round(position, 6),
                "latest_position_date": latest_position_date,
                "source": source,
                "source_url": source_url,
            }
        )
    return records
# ---------------------------------------------------------------------------
# Existing JSONL compatibility
# ---------------------------------------------------------------------------
def normalize_existing_record(
    record: dict,
    fallback_snapshot_date: date | None = None,
) -> dict | None:
    """
    Normalizes both the old and new FI aggregate JSONL schemas.
    Old schema:
        source_date
        position_date
        issuer
        lei
        short_interest_pct
    New schema:
        snapshot_date
        issuer
        lei
        position
        latest_position_date
        source
        source_url
    """
    issuer = normalize_text(
        record.get("issuer")
    )
    if not issuer:
        return None
    snapshot_date = (
        record.get("snapshot_date")
        or record.get("source_date")
        or record.get("position_date")
    )
    if snapshot_date is None and fallback_snapshot_date:
        snapshot_date = fallback_snapshot_date.isoformat()
    snapshot_date = parse_date(snapshot_date)
    if snapshot_date is None:
        return None
    position = record.get("position")
    if position is None:
        position = record.get(
            "short_interest_pct"
        )
    position = parse_position(position)
    if position is None:
        return None
    lei = normalize_text(
        record.get("lei")
    )
    latest_position_date = (
        record.get("latest_position_date")
        or record.get("position_date")
    )
    latest_position_date = parse_date(
        latest_position_date
    )
    source = record.get("source")
    if not source:
        source = "existing-fi"
    source_url = record.get(
        "source_url"
    )
    normalized = {
        "snapshot_date": snapshot_date,
        "issuer": issuer,
        "lei": lei or None,
        "position": round(position, 6),
        "latest_position_date": latest_position_date,
        "source": source,
        "source_url": source_url,
    }
    return normalized
def read_existing_snapshot(
    path: Path,
) -> list[dict]:
    fallback_date = extract_date_from_filename(
        path.name
    )
    records = []
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            normalized = normalize_existing_record(
                raw,
                fallback_snapshot_date=fallback_date,
            )
            if normalized is not None:
                records.append(normalized)
    return records
# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_snapshot(
    records: list[dict],
    snapshot_date: date,
) -> dict:
    issuers = [
        record["issuer"]
        for record in records
    ]
    positions = [
        record["position"]
        for record in records
    ]
    duplicate_issuers = sorted(
        {
            issuer
            for issuer in issuers
            if issuers.count(issuer) > 1
        }
    )
    negative_positions = [
        record
        for record in records
        if record["position"] < 0
    ]
    invalid_positions = [
        record
        for record in records
        if record["position"] > 100
    ]
    return {
        "snapshot_date": snapshot_date.isoformat(),
        "rows": len(records),
        "unique_issuers": len(set(issuers)),
        "min_position": (
            min(positions)
            if positions
            else None
        ),
        "max_position": (
            max(positions)
            if positions
            else None
        ),
        "duplicate_issuers": duplicate_issuers,
        "negative_positions": len(
            negative_positions
        ),
        "positions_over_100": len(
            invalid_positions
        ),
        "valid": (
            bool(records)
            and not duplicate_issuers
            and not negative_positions
            and not invalid_positions
        ),
    }
# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def write_snapshot(
    records: list[dict],
    snapshot_date: date,
) -> Path:
    path = RAW_DIR / (
        f"fi_aggregate_"
        f"{snapshot_date.isoformat()}.jsonl"
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
def rebuild_combined_dataset() -> Path:
    """
    Rebuilds the combined dataset from all per-snapshot JSONL files.
    This function deliberately accepts both the historical schema and
    the current schema so existing FI snapshots remain usable.
    """
    output = (
        PROCESSED_DIR
        / "fi_aggregate_history.jsonl"
    )
    files = sorted(
        RAW_DIR.glob(
            "fi_aggregate_*.jsonl"
        )
    )
    seen = set()
    with output.open(
        "w",
        encoding="utf-8",
    ) as destination:
        for path in files:
            fallback_date = (
                extract_date_from_filename(
                    path.name
                )
            )
            with path.open(
                "r",
                encoding="utf-8",
            ) as source:
                for line in source:
                    if not line.strip():
                        continue
                    try:
                        raw_record = json.loads(
                            line
                        )
                    except json.JSONDecodeError:
                        continue
                    record = normalize_existing_record(
                        raw_record,
                        fallback_snapshot_date=fallback_date,
                    )
                    if record is None:
                        continue
                    key = (
                        record["snapshot_date"],
                        record["issuer"],
                        record["lei"],
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    destination.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    return output
# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def get_existing_snapshot_dates() -> list[str]:
    dates = []
    for path in RAW_DIR.glob(
        "fi_aggregate_*.jsonl"
    ):
        snapshot_date = (
            extract_date_from_filename(
                path.name
            )
        )
        if snapshot_date is None:
            continue
        if not in_range(snapshot_date):
            continue
        dates.append(
            snapshot_date.isoformat()
        )
    return sorted(set(dates))
def write_metadata(
    candidates: list[Candidate],
    validations: list[dict],
) -> None:
    existing_dates = (
        get_existing_snapshot_dates()
    )
    downloaded_dates = sorted(
        {
            validation["snapshot_date"]
            for validation in validations
            if validation["valid"]
        }
    )
    all_valid_dates = sorted(
        set(existing_dates)
        | set(downloaded_dates)
    )
    metadata = {
        "dataset": (
            "FI aggregate short positions"
        ),
        "definition": (
            "Aggregate short positions above "
            "FI reporting threshold of 0.1 percent."
        ),
        "requested_start": (
            START_DATE.isoformat()
        ),
        "requested_end": (
            END_DATE.isoformat()
        ),
        "available_snapshots": len(
            all_valid_dates
        ),
        "available_dates": all_valid_dates,
        "newly_downloaded_snapshots": len(
            downloaded_dates
        ),
        "newly_downloaded_dates": (
            downloaded_dates
        ),
        "existing_snapshots_reused": len(
            set(existing_dates)
        ),
        "existing_dates_reused": (
            existing_dates
        ),
        "candidate_count": len(
            candidates
        ),
        "sources": sorted(
            {
                candidate.source
                for candidate in candidates
            }
        ),
        "validation": validations,
        "generated_at": (
            datetime.utcnow().isoformat()
            + "Z"
        ),
    }
    path = (
        PROCESSED_DIR
        / "fi_aggregate_history_metadata.json"
    )
    path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    print("=" * 70)
    print(
        "FI AGGREGATE HISTORY DISCOVERY"
    )
    print("=" * 70)
    # ------------------------------------------------------------------
    # 1. Wayback
    # ------------------------------------------------------------------
    print()
    print("Söker Wayback...")
    wayback_pattern = (
        "fi.se/contentassets/*/"
        "aggregerade-blankningspositioner-*.xlsx"
    )
    wayback_candidates = query_wayback(
        wayback_pattern
    )
    candidates.extend(
        wayback_candidates
    )
    print(
        f"Wayback: "
        f"{len(wayback_candidates)} kandidater"
    )
    # ------------------------------------------------------------------
    # 2. Common Crawl
    # ------------------------------------------------------------------
    try:
        indexes = (
            get_commoncrawl_indexes()
        )
    except Exception as exc:
        print(
            "Kunde inte hämta "
            f"Common Crawl-index: {exc}"
        )
        indexes = []
    commoncrawl_count_before = (
        len(candidates)
    )
    for index_name in indexes:
        print(
            f"Söker Common Crawl: "
            f"{index_name}"
        )
        found = query_commoncrawl(
            index_name
        )
        candidates.extend(found)
        time.sleep(0.5)
    print(
        "Common Crawl:",
        len(candidates)
        - commoncrawl_count_before,
        "kandidater",
    )
    # ------------------------------------------------------------------
    # 3. Known direct FI examples
    # ------------------------------------------------------------------
    known_urls = [
        (
            "https://www.fi.se/contentassets/"
            "79e6c3558bd9473fb70a418f51df48d0/"
            "aggregerade-blankningspositioner-"
            "2022-06-01.xlsx"
        ),
        (
            "https://www.fi.se/contentassets/"
            "79e6c3558bd9473fb70a418f51df48d0/"
            "aggregerade-blankningspositioner-"
            "2022-06-08.xlsx"
        ),
    ]
    for url in known_urls:
        snapshot_date = (
            extract_date_from_filename(
                url
            )
        )
        if (
            snapshot_date is not None
            and in_range(snapshot_date)
        ):
            candidates.append(
                Candidate(
                    url=url,
                    snapshot_date=snapshot_date,
                    source="known-fi",
                )
            )
    # ------------------------------------------------------------------
    # Deduplicate
    # ------------------------------------------------------------------
    unique: dict[
        tuple[str, str],
        Candidate,
    ] = {}
    for candidate in candidates:
        key = (
            candidate.snapshot_date.isoformat(),
            candidate.url,
        )
        unique[key] = candidate
    result = sorted(
        unique.values(),
        key=lambda candidate: (
            candidate.snapshot_date,
            candidate.source,
            candidate.url,
        ),
    )
    return result
# ---------------------------------------------------------------------------
# Download candidates
# ---------------------------------------------------------------------------
def download_candidates(
    candidates: list[Candidate],
) -> list[dict]:
    validations = []
    by_date: dict[
        date,
        list[Candidate],
    ] = {}
    for candidate in candidates:
        by_date.setdefault(
            candidate.snapshot_date,
            [],
        ).append(candidate)
    for snapshot_date in sorted(
        by_date
    ):
        print()
        print("-" * 70)
        print(
            f"Snapshot "
            f"{snapshot_date.isoformat()}"
        )
        existing = (
            RAW_DIR
            / f"fi_aggregate_"
            f"{snapshot_date.isoformat()}.jsonl"
        )
        if existing.exists():
            existing_records = (
                read_existing_snapshot(
                    existing
                )
            )
            validation = (
                validate_snapshot(
                    existing_records,
                    snapshot_date,
                )
            )
            print(
                "Finns redan:",
                existing,
            )
            print(
                f"  Rader: "
                f"{validation['rows']}"
            )
            print(
                f"  Unika emittenter: "
                f"{validation['unique_issuers']}"
            )
            if validation["valid"]:
                validations.append(
                    validation
                )
                print(
                    "  Återanvänds: "
                    "befintlig snapshot är giltig."
                )
            else:
                print(
                    "  FEL: befintlig snapshot "
                    "klarade inte integritetskontrollen."
                )
            continue
        candidates_for_date = (
            by_date[snapshot_date]
        )
        successful = False
        candidates_for_date.sort(
            key=lambda candidate: {
                "known-fi": 0,
                "commoncrawl": 1,
                "wayback": 2,
            }.get(
                candidate.source,
                9,
            )
        )
        for candidate in candidates_for_date:
            print(
                f"Försöker "
                f"{candidate.source}: "
                f"{candidate.url}"
            )
            content = None
            if candidate.source in (
                "known-fi",
                "commoncrawl",
            ):
                content = download_direct(
                    candidate
                )
            if (
                content is None
                and (
                    candidate.source == "wayback"
                    or candidate.archive_timestamp
                )
            ):
                content = download_wayback(
                    candidate
                )
            if content is None:
                print("  MISS")
                continue
            print(
                f"  Hämtad: "
                f"{len(content):,} bytes"
            )
            try:
                records = parse_xlsx(
                    content=content,
                    snapshot_date=snapshot_date,
                    source=candidate.source,
                    source_url=candidate.url,
                )
                validation = (
                    validate_snapshot(
                        records,
                        snapshot_date,
                    )
                )
                print(
                    f"  Rader: "
                    f"{validation['rows']}"
                )
                print(
                    f"  Unika emittenter: "
                    f"{validation['unique_issuers']}"
                )
                print(
                    f"  Max blankning: "
                    f"{validation['max_position']}"
                )
                if not validation["valid"]:
                    print(
                        "  FEL: snapshot klarade "
                        "inte integritetskontrollen."
                    )
                    print(
                        json.dumps(
                            validation,
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                    continue
                path = write_snapshot(
                    records,
                    snapshot_date,
                )
                print(
                    "  OK:",
                    path,
                )
                validations.append(
                    validation
                )
                successful = True
                break
            except Exception as exc:
                print(
                    "  FEL vid parsing:",
                    exc,
                )
        if not successful:
            print(
                "  INGEN KÄLLA FUNNEN"
            )
    return validations
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    candidates = (
        discover_candidates()
    )
    print()
    print("=" * 70)
    print("KANDIDATER")
    print("=" * 70)
    for candidate in candidates:
        print(
            candidate.snapshot_date.isoformat(),
            candidate.source,
            candidate.url,
        )
    validations = (
        download_candidates(
            candidates
        )
    )
    combined = (
        rebuild_combined_dataset()
    )
    write_metadata(
        candidates,
        validations,
    )
    print()
    print("=" * 70)
    print("KLART")
    print("=" * 70)
    print(
        "Giltiga snapshots:",
        len(validations),
    )
    print(
        "Kombinerad fil:",
        combined,
    )
    print(
        "Metadata:",
        PROCESSED_DIR
        / "fi_aggregate_history_metadata.json",
    )
if __name__ == "__main__":
    main()
