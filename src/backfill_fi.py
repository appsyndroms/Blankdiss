"""
Historisk FI-backfill för Blankdiss.
Mål:
    FI
      ↓
    Aggregerade positioner
      ↓
    rå Excel
      ↓
    normaliserad JSONL
      ↓
    manifest/cache
Vi hämtar inte samma källa igen om den redan finns med samma
SHA-256 i manifestet.
Primärt dataset:
    FI:s "Aggregerade positioner"
Detta är viktigt eftersom FI anger att:
    Aktuella positioner
    Historiska positioner
bara innehåller positioner över 0,5 %, medan:
    Aggregerade positioner
innehåller summan av blankningspositioner över 0,1 %.
Källan kan innehålla antingen en aktuell snapshot eller historiska
datum. Om endast aktuell data finns tillgänglig skapar programmet
inte påhittad historik. Den sparar det som faktiskt finns.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import urljoin
import pandas as pd
import requests
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = (
    RAW_DIR / "fi_manifest.json"
)
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
class BackfillError(Exception):
    """Kontrollerat fel i FI-backfill."""
def normalize_text(value) -> str:
    if value is None:
        return ""
    return (
        str(value)
        .replace("\xa0", " ")
        .strip()
    )
def normalize_percent(value) -> float | None:
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
        value_float = float(text)
    except ValueError:
        return None
    if value_float < 0:
        return None
    if value_float > 100:
        return None
    return value_float
def normalize_date(value) -> str | None:
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
    text = normalize_text(value)
    if not text:
        return None
    parsed = pd.to_datetime(
        text,
        errors="coerce",
        dayfirst=False,
    )
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
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(
        data
    ).hexdigest()
def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {
            "source": "FI",
            "dataset": "aggregate_short_positions",
            "last_checked": None,
            "files": {},
        }
    try:
        with MANIFEST_PATH.open(
            "r",
            encoding="utf-8",
        ) as handle:
            return json.load(handle)
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise BackfillError(
            "Kunde inte läsa "
            "fi_manifest.json."
        ) from exc
def save_manifest(
    manifest: dict,
) -> None:
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
def fetch_html() -> str:
    try:
        response = requests.get(
            FI_URL,
            headers=HEADERS,
            timeout=30,
        )
    except requests.RequestException as exc:
        raise BackfillError(
            "HTTP-fel vid hämtning av FI-sidan: "
            f"{type(exc).__name__}"
        ) from exc
    if response.status_code != 200:
        raise BackfillError(
            "FI-sidan svarade med HTTP "
            f"{response.status_code}."
        )
    if not response.text.strip():
        raise BackfillError(
            "FI returnerade tom HTML."
        )
    return response.text
def find_aggregate_url(
    html: str,
) -> str | None:
    """
    Försöker hitta länken till FI:s
    "Aggregerade positioner".
    Vi letar först efter länkar vars text innehåller
    "Aggregerade positioner".
    Därefter försöker vi även hitta en href vars
    filnamn innehåller "aggregerade".
    """
    anchor_pattern = re.compile(
        r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>"
        r"(.*?)"
        r"</a>",
        re.IGNORECASE | re.DOTALL,
    )
    for href, text in anchor_pattern.findall(
        html
    ):
        clean_text = re.sub(
            r"<[^>]+>",
            " ",
            text,
        )
        clean_text = (
            " ".join(
                clean_text.split()
            )
            .lower()
        )
        if (
            "aggregerade positioner"
            in clean_text
        ):
            return urljoin(
                FI_URL,
                href,
            )
    href_pattern = re.compile(
        r"href=[\"']([^\"']+)[\"']",
        re.IGNORECASE,
    )
    for href in href_pattern.findall(
        html
    ):
        lowered = href.lower()
        if (
            "aggregerade" in lowered
            and (
                lowered.endswith(".xlsx")
                or lowered.endswith(".xls")
                or "download" in lowered
            )
        ):
            return urljoin(
                FI_URL,
                href,
            )
    return None
def download_excel(
    url: str,
) -> tuple[bytes, str]:
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise BackfillError(
            "HTTP-fel vid hämtning av "
            f"FI Excel: {type(exc).__name__}"
        ) from exc
    if response.status_code != 200:
        raise BackfillError(
            "FI Excel svarade med HTTP "
            f"{response.status_code}."
        )
    data = response.content
    if not data:
        raise BackfillError(
            "FI Excel-filen var tom."
        )
    content_type = (
        response.headers.get(
            "Content-Type",
            "",
        )
        .lower()
    )
    extension = ".xlsx"
    if (
        "spreadsheetml"
        not in content_type
        and not url.lower().endswith(
            ".xlsx"
        )
    ):
        if url.lower().endswith(
            ".xls"
        ):
            extension = ".xls"
    return data, extension
def resolve_column(
    columns,
    patterns: list[str],
) -> str | None:
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
def normalize_dataframe(
    table: pd.DataFrame,
    default_snapshot_date: str,
) -> list[dict]:
    """
    Försöker normalisera ett FI-ark.
    Vi accepterar flera möjliga kolumnnamn eftersom FI kan ändra
    rubriker utan att datasetets semantik förändras.
    """
    if table.empty:
        return []
    issuer_column = resolve_column(
        table.columns,
        [
            "emittentens namn",
            "bolagsnamn",
            "emittent",
            "issuer",
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
    date_column = resolve_column(
        table.columns,
        [
            "positionsdatum",
            "datum",
            "date",
            "snapshot",
        ],
    )
    short_column = resolve_column(
        table.columns,
        [
            "summa blankning %",
            "summa blankning",
            "aggregerad",
            "aggregate",
        ],
    )
    if not issuer_column:
        return []
    if not short_column:
        return []
    records: list[dict] = []
    for _, row in table.iterrows():
        issuer = normalize_text(
            row.get(
                issuer_column
            )
        )
        if not issuer:
            continue
        lei = ""
        if lei_column:
            lei = normalize_text(
                row.get(
                    lei_column
                )
            )
        snapshot_date = (
            default_snapshot_date
        )
        if date_column:
            candidate_date = (
                normalize_date(
                    row.get(
                        date_column
                    )
                )
            )
            if candidate_date:
                snapshot_date = (
                    candidate_date
                )
        short_interest_pct = (
            normalize_percent(
                row.get(
                    short_column
                )
            )
        )
        if short_interest_pct is None:
            continue
        records.append(
            {
                "snapshot_date": (
                    snapshot_date
                ),
                "position_date": (
                    snapshot_date
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
def parse_excel(
    data: bytes,
) -> list[dict]:
    try:
        workbook = pd.ExcelFile(
            BytesIO(data)
        )
    except Exception as exc:
        raise BackfillError(
            "Kunde inte öppna FI:s Excel-fil."
        ) from exc
    default_snapshot_date = (
        date.today().isoformat()
    )
    records: list[dict] = []
    for sheet_name in workbook.sheet_names:
        try:
            table = pd.read_excel(
                workbook,
                sheet_name=sheet_name,
            )
        except Exception as exc:
            raise BackfillError(
                "Kunde inte läsa FI-arket "
                f"'{sheet_name}'."
            ) from exc
        sheet_records = normalize_dataframe(
            table,
            default_snapshot_date,
        )
        records.extend(
            sheet_records
        )
    return records
def write_records_by_date(
    records: list[dict],
    start_date: date,
    end_date: date,
) -> list[str]:
    written_dates: list[str] = []
    grouped: dict[
        str,
        list[dict],
    ] = {}
    for record in records:
        snapshot = record[
            "snapshot_date"
        ]
        try:
            snapshot_date = (
                date.fromisoformat(
                    snapshot
                )
            )
        except ValueError:
            continue
        if not (
            start_date
            <= snapshot_date
            <= end_date
        ):
            continue
        grouped.setdefault(
            snapshot,
            [],
        ).append(
            record
        )
    for snapshot_date in sorted(
        grouped
    ):
        path = (
            RAW_DIR
            / (
                "fi_aggregate_"
                f"{snapshot_date}.jsonl"
            )
        )
        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for record in grouped[
                snapshot_date
            ]:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        written_dates.append(
            snapshot_date
        )
    return written_dates
def fallback_current_html(
    snapshot_date: str,
) -> list[dict]:
    html = fetch_html()
    try:
        tables = pd.read_html(
            StringIO(html),
            decimal=",",
            thousands=" ",
        )
    except Exception as exc:
        raise BackfillError(
            "Kunde inte tolka FI:s "
            "HTML-tabell."
        ) from exc
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
        date_column = resolve_column(
            table.columns,
            [
                "positionsdatum senaste position",
                "positionsdatum",
            ],
        )
        short_column = resolve_column(
            table.columns,
            [
                "summa blankning %",
                "summa blankning",
            ],
        )
        if not (
            issuer_column
            and lei_column
            and date_column
            and short_column
        ):
            continue
        records: list[dict] = []
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
                    date_column
                )
            )
            short_interest_pct = (
                normalize_percent(
                    row.get(
                        short_column
                    )
                )
            )
            if not issuer:
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
        if records:
            return records
    raise BackfillError(
        "Kunde inte hitta FI:s "
        "aggregerade HTML-tabell."
    )
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill av FI:s aggregerade "
            "blankningsdata."
        )
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help=(
            "Antal kalenderdagar bakåt. "
            "Standard: 90."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Hämta om källfilen även om "
            "manifestet redan innehåller "
            "samma URL."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Visa full traceback."
        ),
    )
    args = parser.parse_args()
    if args.days < 1:
        parser.error(
            "--days måste vara minst 1."
        )
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    end_date = date.today()
    start_date = (
        end_date
        - timedelta(
            days=args.days - 1
        )
    )
    print(
        "FI-backfill: "
        f"{start_date} → {end_date}"
    )
    manifest = load_manifest()
    manifest[
        "last_checked"
    ] = datetime.now(
        timezone.utc
    ).isoformat()
    try:
        html = fetch_html()
        aggregate_url = (
            find_aggregate_url(
                html
            )
        )
        if aggregate_url:
            print(
                "FI-backfill: "
                "aggregerad Excel-källa hittad."
            )
            source_key = (
                aggregate_url
            )
            existing = (
                manifest
                .get("files", {})
                .get(source_key)
            )
            data: bytes | None = None
            extension = ".xlsx"
            if (
                existing
                and not args.force
                and existing.get(
                    "local_file"
                )
            ):
                local_file = (
                    RAW_DIR
                    / existing[
                        "local_file"
                    ]
                )
                if local_file.exists():
                    data = (
                        local_file.read_bytes()
                    )
                    print(
                        "FI-backfill: "
                        "återanvänder lokal "
                        "källfil."
                    )
            if data is None:
                data, extension = (
                    download_excel(
                        aggregate_url
                    )
                )
                digest = sha256_bytes(
                    data
                )
                local_file = (
                    RAW_DIR
                    / (
                        "fi_aggregate_source_"
                        f"{digest[:16]}"
                        f"{extension}"
                    )
                )
                local_file.write_bytes(
                    data
                )
                manifest.setdefault(
                    "files",
                    {},
                )[
                    source_key
                ] = {
                    "url": aggregate_url,
                    "sha256": digest,
                    "local_file": (
                        local_file.name
                    ),
                    "downloaded_at": (
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    ),
                }
                print(
                    "FI-backfill: "
                    f"källfil sparad → "
                    f"{local_file}"
                )
            records = parse_excel(
                data
            )
            if records:
                written_dates = (
                    write_records_by_date(
                        records,
                        start_date,
                        end_date,
                    )
                )
                print(
                    "FI-backfill: "
                    f"{len(records)} "
                    "normaliserade "
                    "observationer."
                )
                print(
                    "FI-backfill: "
                    f"{len(written_dates)} "
                    "snapshot-datum "
                    "sparade."
                )
            else:
                print(
                    "FI-backfill: "
                    "Excel-filen innehöll "
                    "ingen igenkännbar "
                    "aggregerad historik."
                )
        else:
            print(
                "FI-backfill: "
                "ingen Excel-länk hittades."
            )
            print(
                "FI-backfill: "
                "faller tillbaka till "
                "FI:s aktuella HTML-data."
            )
            records = fallback_current_html(
                end_date.isoformat()
            )
            path = (
                RAW_DIR
                / (
                    "fi_aggregate_"
                    f"{end_date.isoformat()}.jsonl"
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
            print(
                "FI-backfill: "
                f"{len(records)} observationer "
                f"→ {path}"
            )
        save_manifest(
            manifest
        )
        print(
            "FI-backfill: manifest sparat → "
            f"{MANIFEST_PATH}"
        )
        print(
            "FI-backfill: klart."
        )
    except BackfillError as exc:
        save_manifest(
            manifest
        )
        print(
            f"FI-backfill: FEL - {exc}",
            file=sys.stderr,
        )
        if args.debug:
            raise
        sys.exit(1)
    except Exception as exc:
        save_manifest(
            manifest
        )
        print(
            "FI-backfill: OVÄNTAT FEL - "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        if args.debug:
            raise
        sys.exit(1)
if __name__ == "__main__":
    main()
