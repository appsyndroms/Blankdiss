"""Hämta och normalisera FI:s historiska blankningspositioner.

Källan innehåller historiska individuella blankningspositioner
och ska hållas separerad från FI:s aggregerade >0,1 %-data.

FI:s historiska positionsregister publicerar betydande
positioner, dvs. positioner över 0,5 %.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
import requests

from fi.config import HEADERS


HISTORICAL_URL = (
    "https://www.fi.se/BlankningsRegister/"
    "GetHistFile"
)

REQUEST_TIMEOUT = 60

OUTPUT_DIR = Path(
    "data/raw/fi/positions/historical"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "fi_historical_positions.jsonl"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "fi_historical_positions_metadata.json"
)

SUB05_RE = re.compile(
    r"^\s*<\s*0[,.]5\s*$",
    re.IGNORECASE,
)


def download_history() -> bytes:
    """Hämtar FI:s historiska positionsfil."""
    response = requests.get(
        HISTORICAL_URL,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    if not response.content:
        raise RuntimeError(
            "FI returnerade en tom historikfil."
        )

    return response.content


def clean_text(
    value: object,
) -> str | None:
    """Normaliserar text."""
    if value is None:
        return None

    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.lower() == "nan":
        return None

    return text


def normalize_column_name(
    value: object,
) -> str:
    """Normaliserar kolumnnamn."""
    text = clean_text(value)

    if text is None:
        return ""

    text = text.replace(
        "\n",
        " ",
    )

    text = text.replace(
        "\r",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip().lower()


def find_column(
    columns: list[object],
    candidates: list[str],
) -> object | None:
    """Hittar en kolumn genom namn eller delsträng."""
    normalized = [
        (
            column,
            normalize_column_name(column),
        )
        for column in columns
    ]

    normalized_candidates = [
        normalize_column_name(candidate)
        for candidate in candidates
    ]

    for candidate in normalized_candidates:
        for column, name in normalized:
            if name == candidate:
                return column

    for candidate in normalized_candidates:
        for column, name in normalized:
            if candidate in name:
                return column

    return None


def parse_position(
    value: object,
) -> float | None:
    """Tolkar en numerisk positionsprocent."""
    text = clean_text(value)

    if text is None:
        return None

    if SUB05_RE.match(text):
        return None

    text = (
        text
        .replace(" ", "")
        .replace("%", "")
        .replace(",", ".")
    )

    try:
        return float(text)
    except ValueError:
        return None


def is_below_threshold(
    value: object,
) -> bool:
    """Returnerar True om FI uttrycker positionen som <0,5."""
    text = clean_text(value)

    if text is None:
        return False

    return bool(
        SUB05_RE.match(text)
    )


def parse_date(
    value: object,
) -> str | None:
    """Tolkar datum till ISO-format."""
    if value is None:
        return None

    if pd.isna(value):
        return None

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.date().isoformat()

    parsed = pd.to_datetime(
        value,
        errors="coerce",
        dayfirst=True,
    )

    if pd.isna(parsed):
        return None

    return parsed.date().isoformat()


def read_history(
    data: bytes,
) -> pd.DataFrame:
    """Läser FI:s historiska ODS-fil."""
    try:
        dataframe = pd.read_excel(
            io.BytesIO(data),
            sheet_name=0,
            engine="odf",
        )
    except ImportError as exc:
        raise RuntimeError(
            "ODS-stöd saknas. Installera odfpy."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Kunde inte läsa FI:s historiska "
            f"ODS-fil: {exc}"
        ) from exc

    if dataframe.empty:
        raise RuntimeError(
            "FI:s historiska fil innehöll "
            "inga rader."
        )

    return dataframe


def find_required_columns(
    dataframe: pd.DataFrame,
) -> dict[str, object]:
    """Identifierar FI:s relevanta kolumner."""
    columns = list(
        dataframe.columns
    )

    holder = find_column(
        columns,
        [
            "Innehavare",
            "Innehavare namn",
            "Position holder",
            "Holder",
        ],
    )

    issuer = find_column(
        columns,
        [
            "Emittentens namn",
            "Emittent",
            "Issuer",
            "Name of issuer",
        ],
    )

    position = find_column(
        columns,
        [
            "Position i procent",
            "Position %",
            "Position procent",
            "Net short position %",
            "Net short position in per cent",
            "Position",
        ],
    )

    position_date = find_column(
        columns,
        [
            "Positionsdatum",
            "Position date",
            "Date of position",
        ],
    )

    isin = find_column(
        columns,
        [
            "ISIN",
            "ISIN-kod",
        ],
    )

    if holder is None:
        raise RuntimeError(
            "Kunde inte hitta kolumnen "
            "för innehavare. "
            f"Kolumner: {columns}"
        )

    if issuer is None:
        raise RuntimeError(
            "Kunde inte hitta kolumnen "
            "för emittent. "
            f"Kolumner: {columns}"
        )

    if position is None:
        raise RuntimeError(
            "Kunde inte hitta kolumnen "
            "för position. "
            f"Kolumner: {columns}"
        )

    if position_date is None:
        raise RuntimeError(
            "Kunde inte hitta kolumnen "
            "för positionsdatum. "
            f"Kolumner: {columns}"
        )

    return {
        "holder": holder,
        "issuer": issuer,
        "position": position,
        "position_date": position_date,
        "isin": isin,
    }


def normalize_records(
    dataframe: pd.DataFrame,
    columns: dict[str, object],
) -> list[dict]:
    """Normaliserar FI:s historiska positioner."""
    records: list[dict] = []

    holder_column = columns["holder"]
    issuer_column = columns["issuer"]
    position_column = columns["position"]
    date_column = columns["position_date"]
    isin_column = columns["isin"]

    for _, row in dataframe.iterrows():
        holder = clean_text(
            row.get(holder_column)
        )

        issuer = clean_text(
            row.get(issuer_column)
        )

        raw_position = clean_text(
            row.get(position_column)
        )

        position_date = parse_date(
            row.get(date_column)
        )

        if holder is None:
            continue

        if issuer is None:
            continue

        if position_date is None:
            continue

        if raw_position is None:
            continue

        position_below_05 = (
            is_below_threshold(
                raw_position
            )
        )

        position_pct = parse_position(
            raw_position
        )

        # Vi behåller även <0,5-poster.
        # De får position_pct=None eftersom
        # FI inte anger ett exakt värde.
        if (
            position_pct is None
            and not position_below_05
        ):
            continue

        if isin_column is not None:
            isin = clean_text(
                row.get(isin_column)
            )
        else:
            isin = None

        records.append(
            {
                "position_date": position_date,
                "holder": holder,
                "issuer": issuer,
                "isin": isin,
                "position_pct": position_pct,
                "position_below_0_5": (
                    position_below_05
                ),
                "raw_position": raw_position,
            }
        )

    return records


def write_jsonl(
    records: list[dict],
) -> None:
    """Skriver normaliserad historik som JSONL."""
    import json

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def write_metadata(
    dataframe: pd.DataFrame,
    records: list[dict],
    columns: dict[str, object],
) -> None:
    """Skriver metadata om importen."""
    import json

    dates = [
        record["position_date"]
        for record in records
        if record.get("position_date")
    ]

    numeric_positions = [
        record
        for record in records
        if record.get("position_pct")
        is not None
    ]

    below_threshold = [
        record
        for record in records
        if record.get(
            "position_below_0_5"
        )
    ]

    metadata = {
        "source": HISTORICAL_URL,
        "raw_rows": len(dataframe),
        "normalized_rows": len(records),
        "numeric_position_rows": len(
            numeric_positions
        ),
        "below_0_5_rows": len(
            below_threshold
        ),
        "first_position_date": (
            min(dates)
            if dates
            else None
        ),
        "last_position_date": (
            max(dates)
            if dates
            else None
        ),
        "columns": {
            key: (
                str(value)
                if value is not None
                else None
            )
            for key, value in columns.items()
        },
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """Hämtar och sparar FI:s historiska positioner."""
    print(
        "FI historical positions: "
        "hämtar historik..."
    )

    data = download_history()

    print(
        "FI historical positions: "
        f"{len(data):,} bytes hämtade."
    )

    dataframe = read_history(
        data
    )

    print(
        "FI historical positions: "
        f"{len(dataframe):,} råa rader."
    )

    columns = find_required_columns(
        dataframe
    )

    print(
        "FI historical positions: "
        "kolumner identifierade."
    )

    records = normalize_records(
        dataframe,
        columns,
    )

    if not records:
        raise RuntimeError(
            "Ingen historisk position kunde "
            "normaliseras."
        )

    write_jsonl(
        records
    )

    write_metadata(
        dataframe,
        records,
        columns,
    )

    dates = sorted(
        {
            record["position_date"]
            for record in records
            if record.get("position_date")
        }
    )

    below_threshold = sum(
        record["position_below_0_5"]
        for record in records
    )

    numeric = sum(
        record["position_pct"] is not None
        for record in records
    )

    print()
    print(
        "=========================================="
    )
    print(
        "FI HISTORISKA POSITIONER KLART"
    )
    print(
        "=========================================="
    )

    print(
        f"Råa rader:          {len(dataframe):,}"
    )

    print(
        f"Normaliserade:      {len(records):,}"
    )

    print(
        f"Numeriska:          {numeric:,}"
    )

    print(
        f"<0,5:               {below_threshold:,}"
    )

    print(
        "Första datum:       "
        f"{dates[0] if dates else 'saknas'}"
    )

    print(
        "Sista datum:        "
        f"{dates[-1] if dates else 'saknas'}"
    )

    print(
        f"JSONL:              {OUTPUT_PATH}"
    )

    print(
        f"Metadata:           {METADATA_PATH}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
