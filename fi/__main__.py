"""
FI-data för Blankdiss.

Ansvarar för:

- aktuell FI-data
- historisk FI-backfill
- normalisering
- manifest/cache
- rådata till data/raw/fi/aggregate/

Körs med:

    python -m fi

Exempel:

    python -m fi --days 90
    python -m fi --days 90 --force
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

RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "fi"
    / "aggregate"
)

SOURCE_DIR = RAW_DIR / "source"
SNAPSHOT_DIR = RAW_DIR / "snapshots"

MANIFEST_PATH = (
    RAW_DIR
    / "manifest.json"
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


class FIError(Exception):
    """Kontrollerat fel i FI-flödet."""


def normalize_text(value) -> str:
    if value is None:
        return ""

    return (
        str(value)
        .replace("\xa0", " ")
        .strip()
    )


def normalize_percent(value) -> float | None:
    """
    Normaliserar svensk procentnotation.

    Exempel:

        7,1     -> 7.1
        0,49    -> 0.49
        7.1 %   -> 7.1

    Viktigt:
    pandas måste läsa svensk decimalnotation korrekt innan
    denna funktion körs.
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
        result = float(text)
    except ValueError:
        return None

    if result < 0 or result > 100:
        return None

    return result


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


def fetch_html() -> str:
    try:
        response = requests.get(
            FI_URL,
            headers=HEADERS,
            timeout=30,
        )
    except requests.RequestException as exc:
        raise FIError(
            "HTTP-fel vid hämtning från FI: "
            f"{type(exc).__name__}"
        ) from exc

    if response.status_code != 200:
        raise FIError(
            "FI svarade med HTTP "
            f"{response.status_code}."
        )

    if not response.text.strip():
        raise FIError(
            "FI returnerade ett tomt HTML-svar."
        )

    return response.text


def find_aggregate_url(
    html: str,
) -> str | None:

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


def download_source(
    url: str,
) -> tuple[bytes, str]:

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise FIError(
            "HTTP-fel vid hämtning av FI Excel: "
            f"{type(exc).__name__}"
        ) from exc

    if response.status_code != 200:
        raise FIError(
            "FI Excel svarade med HTTP "
            f"{response.status_code}."
        )

    data = response.content

    if not data:
        raise FIError(
            "FI Excel-filen var tom."
        )

    if url.lower().endswith(".xls"):
        extension = ".xls"
    else:
        extension = ".xlsx"

    return data, extension


def sha256_bytes(
    data: bytes,
) -> str:

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
        return json.loads(
            MANIFEST_PATH.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise FIError(
            "Kunde inte läsa FI-manifestet."
        ) from exc


def save_manifest(
    manifest: dict,
) -> None:

    RAW_DIR.mkdir(
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


def find_current_table(
    tables: list[pd.DataFrame],
) -> pd.DataFrame:

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

    raise FIError(
        "Kunde inte hitta FI:s blankningstabell."
    )


def normalize_current_table(
    table: pd.DataFrame,
    snapshot_date: str,
) -> list[dict]:

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
                position_date_column
            )
        )

        short_interest_pct = normalize_percent(
            row.get(
                short_interest_column
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
                "snapshot_date": snapshot_date,
                "position_date": position_date,
                "lei": lei,
                "issuer": issuer,
                "short_interest_pct": short_interest_pct,
                "source": FI_URL,
            }
        )

    return records


def fetch_current() -> list[dict]:

    html = fetch_html()

    try:
        tables = pd.read_html(
            StringIO(html),
            decimal=",",
            thousands=" ",
        )
    except Exception as exc:
        raise FIError(
            "Kunde inte tolka FI:s HTML-tabeller."
        ) from exc

    if not tables:
        raise FIError(
            "FI-sidan innehöll inga tabeller."
        )

    table = find_current_table(
        tables
    )

    return normalize_current_table(
        table,
        date.today().isoformat(),
    )


def write_snapshot(
    records: list[dict],
) -> Path:

    SNAPSHOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_date = date.today().isoformat()

    path = (
        SNAPSHOT_DIR
        / (
            "fi_aggregate_"
            f"{snapshot_date}.jsonl"
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


def normalize_dataframe(
    table: pd.DataFrame,
    default_snapshot_date: str,
) -> list[dict]:

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

            candidate_date = normalize_date(
                row.get(
                    date_column
                )
            )

            if candidate_date:
                snapshot_date = candidate_date

        short_interest_pct = normalize_percent(
            row.get(
                short_column
            )
        )

        if short_interest_pct is None:
            continue

        records.append(
            {
                "snapshot_date": snapshot_date,
                "position_date": snapshot_date,
                "lei": lei,
                "issuer": issuer,
                "short_interest_pct": short_interest_pct,
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
        raise FIError(
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
            raise FIError(
                "Kunde inte läsa FI-arket "
                f"'{sheet_name}'."
            ) from exc

        records.extend(
            normalize_dataframe(
                table,
                default_snapshot_date,
            )
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
            snapshot_date = date.fromisoformat(
                snapshot
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
        ).append(record)

    for snapshot_date in sorted(
        grouped
    ):

        path = (
            SNAPSHOT_DIR
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


def run_backfill(
    days: int,
    force: bool,
) -> None:

    end_date = date.today()

    start_date = (
        end_date
        - timedelta(
            days=days - 1
        )
    )

    print(
        "FI: backfill "
        f"{start_date} → {end_date}"
    )

    manifest = load_manifest()

    manifest[
        "last_checked"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    html = fetch_html()

    aggregate_url = find_aggregate_url(
        html
    )

    if not aggregate_url:

        print(
            "FI: ingen Excel-länk hittades."
        )

        records = fetch_current()

        path = write_snapshot(
            records
        )

        print(
            "FI: aktuell snapshot "
            f"→ {path}"
        )

        save_manifest(
            manifest
        )

        return

    existing = (
        manifest
        .get("files", {})
        .get(aggregate_url)
    )

    data: bytes | None = None

    if (
        existing
        and not force
        and existing.get(
            "local_file"
        )
    ):

        local_file = (
            SOURCE_DIR
            / existing[
                "local_file"
            ]
        )

        if local_file.exists():

            data = local_file.read_bytes()

            print(
                "FI: återanvänder "
                "lokal källfil."
            )

    if data is None:

        data, extension = download_source(
            aggregate_url
        )

        digest = sha256_bytes(
            data
        )

        SOURCE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        local_file = (
            SOURCE_DIR
            / (
                "aggregate_"
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
        )[aggregate_url] = {
            "url": aggregate_url,
            "sha256": digest,
            "local_file": local_file.name,
            "downloaded_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        }

        print(
            "FI: källfil sparad → "
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
            "FI: "
            f"{len(records)} observationer."
        )

        print(
            "FI: "
            f"{len(written_dates)} snapshot-datum."
        )

    else:

        print(
            "FI: Excel-filen innehöll "
            "ingen igenkännbar historik."
        )

    save_manifest(
        manifest
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "FI-data för Blankdiss."
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
            "Hämta om FI-källan."
        ),
    )

    parser.add_argument(
        "--current-only",
        action="store_true",
        help=(
            "Hämta endast aktuell FI-snapshot."
        ),
    )

    args = parser.parse_args()

    if args.days < 1:
        parser.error(
            "--days måste vara minst 1."
        )

    try:

        RAW_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        if args.current_only:

            records = fetch_current()

            path = write_snapshot(
                records
            )

            print(
                "FI: OK - "
                f"{len(records)} observationer "
                f"→ {path}"
            )

            return

        run_backfill(
            days=args.days,
            force=args.force,
        )

        records = fetch_current()

        path = write_snapshot(
            records
        )

        print(
            "FI: aktuell snapshot - "
            f"{len(records)} observationer "
            f"→ {path}"
        )

        print(
            "FI: klart."
        )

    except FIError as exc:

        print(
            f"FI: FEL - {exc}",
            file=sys.stderr,
        )

        sys.exit(1)

    except Exception as exc:

        print(
            "FI: OVÄNTAT FEL - "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        raise


if __name__ == "__main__":
    main()
