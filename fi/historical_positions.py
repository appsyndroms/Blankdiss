from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT
    / "data"
    / "raw"
    / "fi"
    / "positions"
    / "historical"
)

OUTPUT_PATH = OUTPUT_DIR / "fi_historical_positions.jsonl"
METADATA_PATH = (
    OUTPUT_DIR
    / "fi_historical_positions_metadata.json"
)

FI_URL = (
    "https://www.fi.se/"
    "BlankningsRegister/GetHistFile"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(compatible; Blankdiss/1.0; "
        "+https://github.com/appsyndroms/Blankdiss)"
    ),
    "Accept": (
        "application/vnd.oasis.opendocument.spreadsheet,"
        "application/vnd.oasis.opendocument.spreadsheet-template,"
        "application/octet-stream"
    ),
    "Accept-Language": "sv-SE,sv;q=0.9,en;q=0.8",
}

TIMEOUT = 120

EXPECTED_COLUMNS = {
    "holder": "Innehavare av positionen",
    "issuer": "Namn på emittent",
    "position": "Position i procent",
    "position_date": "Datum för positionen",
    "isin": "ISIN",
}


def clean_text(value: object) -> str | None:
    """Normaliserar text från FI-filen."""
    if value is None:
        return None

    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def find_header_row(
    dataframe: pd.DataFrame,
) -> int:
    """
    Hittar rubrikraden i FI:s ODS-fil.

    FI:s historik innehåller metadata/rubriker före själva
    tabellen, så vi kan inte anta att första raden är header.
    """
    required_terms = [
        "Innehavare",
        "emittent",
        "position",
        "Datum",
    ]

    max_rows = min(
        len(dataframe),
        40,
    )

    for row_index in range(max_rows):
        values = [
            clean_text(value)
            for value in dataframe.iloc[row_index].tolist()
        ]

        combined = " ".join(
            value.lower()
            for value in values
            if value is not None
        )

        if all(
            term.lower() in combined
            for term in required_terms
        ):
            return row_index

    raise RuntimeError(
        "Kunde inte hitta rubrikraden i FI:s historiska "
        "positionsregister."
    )


def find_column(
    columns: list[object],
    expected: str,
) -> object:
    """Hittar en kolumn med robust textmatchning."""
    expected_clean = expected.strip().lower()

    for column in columns:
        text = clean_text(column)

        if text is None:
            continue

        if text.strip().lower() == expected_clean:
            return column

    for column in columns:
        text = clean_text(column)

        if text is None:
            continue

        if expected_clean in text.lower():
            return column

    raise RuntimeError(
        f"Kunde inte hitta kolumnen {expected!r}. "
        f"Tillgängliga kolumner: {list(columns)!r}"
    )


def parse_date(value: object) -> str | None:
    """
    Tolkar FI-datum med explicita format.

    Vi använder inte pd.to_datetime(..., dayfirst=True)
    som generell fallback eftersom FI-filen kan innehålla
    datum som pandas redan har tolkat innan vi får värdet.

    Framtida datum stoppas medvetet. Vi vill hellre att
    importen misslyckas än att Blankdiss bygger in felaktig
    historik.
    """
    if value is None:
        return None

    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        parsed_date = value.date()

    elif isinstance(value, datetime):
        parsed_date = value.date()

    elif isinstance(value, date):
        parsed_date = value

    else:
        text = clean_text(value)

        if text is None:
            return None

        # Ta bort eventuell tid efter datumet.
        text = text.split(" ", 1)[0]

        parsed_date = None

        explicit_formats = (
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d.%m.%Y",
            "%d-%m-%Y",
            "%d/%m/%Y",
        )

        for fmt in explicit_formats:
            try:
                parsed_date = datetime.strptime(
                    text,
                    fmt,
                ).date()
                break
            except ValueError:
                continue

        if parsed_date is None:
            raise RuntimeError(
                "Okänt datumformat i FI-historiken: "
                f"{text!r}"
            )

    today = date.today()

    if parsed_date > today:
        raise RuntimeError(
            "FI-historiken innehåller ett framtida "
            f"positionsdatum: {parsed_date.isoformat()}. "
            "Importen stoppas för att inte bygga in "
            "felaktiga datum i historiken."
        )

    return parsed_date.isoformat()


def parse_position(value: object) -> float | None:
    """Tolkar position i procent."""
    if value is None:
        return None

    if pd.isna(value):
        return None

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return None

        text = (
            text
            .replace("%", "")
            .replace(",", ".")
            .replace(" ", "")
        )

        try:
            return float(text)
        except ValueError:
            return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Normaliserar FI:s historiska positionsregister."""

    header_row = find_header_row(dataframe)

    dataframe = dataframe.iloc[
        header_row + 1 :
    ].copy()

    dataframe.columns = [
        clean_text(column) or ""
        for column in dataframe.columns
    ]

    holder_column = find_column(
        list(dataframe.columns),
        EXPECTED_COLUMNS["holder"],
    )

    issuer_column = find_column(
        list(dataframe.columns),
        EXPECTED_COLUMNS["issuer"],
    )

    position_column = find_column(
        list(dataframe.columns),
        EXPECTED_COLUMNS["position"],
    )

    date_column = find_column(
        list(dataframe.columns),
        EXPECTED_COLUMNS["position_date"],
    )

    isin_column = find_column(
        list(dataframe.columns),
        EXPECTED_COLUMNS["isin"],
    )

    rows: list[dict[str, object]] = []

    below_05_rows = 0
    numeric_position_rows = 0

    for row_number, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):
        holder = clean_text(
            row[holder_column],
        )

        issuer = clean_text(
            row[issuer_column],
        )

        isin = clean_text(
            row[isin_column],
        )

        raw_position = row[position_column]

        position = parse_position(
            raw_position,
        )

        raw_date = row[date_column]

        try:
            position_date = parse_date(
                raw_date,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Felaktigt datum i FI:s historiska "
                f"positionsregister på rad {row_number}: "
                f"råvärde={raw_date!r}, "
                f"typ={type(raw_date).__name__}. "
                f"Originalfel: {exc}"
            ) from exc

        # Tomma/slutliga rader i ODS-filen.
        if (
            holder is None
            and issuer is None
            and isin is None
            and position is None
            and position_date is None
        ):
            continue

        if position is not None:
            numeric_position_rows += 1

            if position < 0.5:
                below_05_rows += 1

        rows.append(
            {
                "holder": holder,
                "issuer": issuer,
                "position": position,
                "position_date": position_date,
                "isin": isin,
            }
        )

    metadata = {
        "header_row": header_row,
        "raw_rows": len(dataframe),
        "normalized_rows": len(rows),
        "numeric_position_rows": numeric_position_rows,
        "below_0_5_rows": below_05_rows,
        "unique_holders": len(
            {
                row["holder"]
                for row in rows
                if row["holder"] is not None
            }
        ),
        "unique_issuers": len(
            {
                row["issuer"]
                for row in rows
                if row["issuer"] is not None
            }
        ),
        "first_position_date": min(
            (
                row["position_date"]
                for row in rows
                if row["position_date"] is not None
            ),
            default=None,
        ),
        "last_position_date": max(
            (
                row["position_date"]
                for row in rows
                if row["position_date"] is not None
            ),
            default=None,
        ),
        "columns": EXPECTED_COLUMNS,
    }

    return rows, metadata


def download_file() -> bytes:
    """Hämtar FI:s historiska positionsregister."""
    import requests

    response = requests.get(
        FI_URL,
        headers=HEADERS,
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    if not response.content:
        raise RuntimeError(
            "FI:s historiska positionsregister "
            "returnerade en tom fil."
        )

    return response.content


def read_ods(
    content: bytes,
) -> pd.DataFrame:
    """Läser FI:s ODS-fil."""
    from io import BytesIO

    try:
        return pd.read_excel(
            BytesIO(content),
            engine="odf",
            header=None,
        )
    except Exception as exc:
        raise RuntimeError(
            "Kunde inte läsa FI:s historiska "
            "positionsregister som ODS."
        ) from exc


def write_jsonl(
    rows: list[dict[str, object]],
) -> None:
    """Skriver normaliserad historik som JSONL."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def write_metadata(
    metadata: dict[str, object],
    raw_content: bytes,
) -> None:
    """Skriver metadata och checksumma."""
    sha256 = hashlib.sha256(
        raw_content
    ).hexdigest()

    metadata = {
        **metadata,
        "source_url": FI_URL,
        "raw_sha256": sha256,
        "output_file": str(
            OUTPUT_PATH.relative_to(ROOT)
        ),
        "generated_at": datetime.now().isoformat(
            timespec="seconds",
        ),
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

        file.write("\n")


def main() -> None:
    """Hämtar, validerar och sparar FI:s historik."""
    print(
        "Hämtar FI:s historiska "
        "positionsregister..."
    )

    raw_content = download_file()

    print(
        "Hämtad fil:",
        f"{len(raw_content):,}",
        "bytes",
    )

    dataframe = read_ods(
        raw_content,
    )

    print(
        "ODS-dimension:",
        dataframe.shape,
    )

    rows, metadata = normalize_dataframe(
        dataframe,
    )

    if not rows:
        raise RuntimeError(
            "FI:s historiska positionsregister "
            "gav inga normaliserade rader."
        )

    write_jsonl(
        rows,
    )

    write_metadata(
        metadata,
        raw_content,
    )

    print()
    print(
        "FI:s historiska positionsregister "
        "sparat."
    )

    print(
        "Råa rader:",
        metadata["raw_rows"],
    )

    print(
        "Normaliserade rader:",
        metadata["normalized_rows"],
    )

    print(
        "Numeriska positioner:",
        metadata["numeric_position_rows"],
    )

    print(
        "Positioner under 0,5 %:",
        metadata["below_0_5_rows"],
    )

    print(
        "Unika innehavare:",
        metadata["unique_holders"],
    )

    print(
        "Unika emittenter:",
        metadata["unique_issuers"],
    )

    print(
        "Första positionsdatum:",
        metadata["first_position_date"],
    )

    print(
        "Sista positionsdatum:",
        metadata["last_position_date"],
    )

    print(
        "JSONL:",
        OUTPUT_PATH,
    )

    print(
        "Metadata:",
        METADATA_PATH,
    )


if __name__ == "__main__":
    main()
