from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
import requests

from fi.config import HEADERS


HISTORICAL_URL = "https://www.fi.se/BlankningsRegister/GetHistFile"

REQUEST_TIMEOUT = 60

OUTPUT_DIR = Path("data/diagnostics/fi")
ODS_PATH = OUTPUT_DIR / "historical-positions.ods"
REPORT_PATH = OUTPUT_DIR / "historical-positions-report.txt"

SUB05_RE = re.compile(
    r"^\s*<\s*0[,.]5\s*$",
    re.IGNORECASE,
)


def download_history() -> bytes:
    response = requests.get(
        HISTORICAL_URL,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    return response.content


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def find_column(
    columns: list[str],
    *terms: str,
) -> str | None:
    for column in columns:
        lowered = column.lower()

        if all(term.lower() in lowered for term in terms):
            return column

    return None


def parse_position(value: object) -> float | None:
    text = clean_text(value)

    if not text:
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


def is_sub05(value: object) -> bool:
    return bool(
        SUB05_RE.match(
            clean_text(value)
        )
    )


def choose_columns(
    df: pd.DataFrame,
) -> tuple[str, str, str, str]:
    columns = [
        str(column)
        for column in df.columns
    ]

    holder = find_column(
        columns,
        "innehavare",
    )

    issuer = find_column(
        columns,
        "emittent",
    )

    position = find_column(
        columns,
        "position",
        "procent",
    )

    date = find_column(
        columns,
        "positionsdatum",
    )

    if not holder or not issuer or not position or not date:
        raise RuntimeError(
            "Kunde inte identifiera FI-kolumnerna. "
            f"Kolumner: {columns}"
        )

    return (
        holder,
        issuer,
        position,
        date,
    )


def analyse(
    df: pd.DataFrame,
) -> list[str]:
    (
        holder_col,
        issuer_col,
        position_col,
        date_col,
    ) = choose_columns(df)

    rows = len(df)

    holders = df[holder_col].map(
        clean_text
    )

    issuers = df[issuer_col].map(
        clean_text
    )

    raw_positions = df[position_col].map(
        clean_text
    )

    dates = pd.to_datetime(
        df[date_col],
        errors="coerce",
    )

    numeric_positions = raw_positions.map(
        parse_position
    )

    sub05_mask = raw_positions.map(
        is_sub05
    )

    keys = pd.DataFrame(
        {
            "holder": holders,
            "issuer": issuers,
        }
    )

    valid_keys = keys[
        (holders != "")
        & (issuers != "")
    ]

    pair_counts = (
        valid_keys
        .value_counts()
        .rename("count")
        .reset_index()
    )

    repeated_pairs = pair_counts[
        pair_counts["count"] > 1
    ]

    lines: list[str] = []

    lines.append(
        "FI HISTORISKA POSITIONER"
    )
    lines.append("=" * 80)

    lines.append(
        f"Källa: {HISTORICAL_URL}"
    )

    lines.append(
        f"Rader: {rows:,}".replace(",", " ")
    )

    lines.append(
        f"Kolumner: {len(df.columns)}"
    )

    lines.append("")

    lines.append(
        "Kolumner som identifierades:"
    )

    lines.append(
        f"  innehavare: {holder_col}"
    )

    lines.append(
        f"  emittent: {issuer_col}"
    )

    lines.append(
        f"  position: {position_col}"
    )

    lines.append(
        f"  datum: {date_col}"
    )

    lines.append("")

    lines.append("POSITIONER")
    lines.append("-" * 80)

    lines.append(
        f"<0,5-rader: {int(sub05_mask.sum()):,}"
        .replace(",", " ")
    )

    lines.append(
        "Exakta numeriska positioner: "
        f"{int(numeric_positions.notna().sum()):,}"
        .replace(",", " ")
    )

    unknown_mask = (
        ~sub05_mask
        & numeric_positions.isna()
    )

    lines.append(
        "Övriga/okända positionsvärden: "
        f"{int(unknown_mask.sum()):,}"
        .replace(",", " ")
    )

    lines.append("")

    lines.append("DATUM")
    lines.append("-" * 80)

    unique_dates = (
        dates
        .dropna()
        .dt.date
        .nunique()
    )

    lines.append(
        f"Unika datum: {unique_dates:,}"
        .replace(",", " ")
    )

    if dates.notna().any():
        lines.append(
            f"Första datum: {dates.min().date()}"
        )

        lines.append(
            f"Senaste datum: {dates.max().date()}"
        )

    lines.append("")

    lines.append(
        "INNEHAVARE + EMITTENT"
    )
    lines.append("-" * 80)

    lines.append(
        f"Unika kombinationer: "
        f"{len(pair_counts):,}"
        .replace(",", " ")
    )

    lines.append(
        f"Kombinationer som förekommer "
        f"flera gånger: {len(repeated_pairs):,}"
        .replace(",", " ")
    )

    lines.append("")

    if not repeated_pairs.empty:
        lines.append(
            "10 vanligaste upprepade kombinationerna:"
        )

        for row in repeated_pairs.head(10).itertuples(
            index=False
        ):
            lines.append(
                f"  {row.holder} | "
                f"{row.issuer} | "
                f"{int(row.count)} rader"
            )

        lines.append("")

    lines.append(
        "KONKRETA TIDSSERIER"
    )
    lines.append("-" * 80)

    repeated_keys = repeated_pairs.head(5)[
        ["holder", "issuer"]
    ]

    for key in repeated_keys.itertuples(
        index=False
    ):
        mask = (
            (holders == key.holder)
            & (issuers == key.issuer)
        )

        sample = pd.DataFrame(
            {
                "date": dates[mask],
                "position": raw_positions[mask],
            }
        )

        sample = (
            sample
            .dropna(subset=["date"])
            .sort_values("date")
        )

        lines.append(
            f"{key.holder} | {key.issuer}"
        )

        for row in sample.head(12).itertuples(
            index=False
        ):
            date_text = row.date.strftime(
                "%Y-%m-%d"
            )

            lines.append(
                f"  {date_text} | "
                f"{row.position}"
            )

        if len(sample) > 12:
            lines.append(
                f"  ... "
                f"{len(sample) - 12} ytterligare rader"
            )

        lines.append("")

    lines.append("TOLKNING")
    lines.append("-" * 80)

    lines.append(
        "Den här analysen behandlar <0,5 "
        "som ett kategoriskt värde, inte som "
        "0,5 eller ett exakt numeriskt värde."
    )

    lines.append(
        "Syftet är att dokumentera vad "
        "GetHistFile faktiskt innehåller "
        "innan vi går vidare till FI:s "
        "aggregeringsmotor."
    )

    return lines


def main() -> int:
    print(
        "Hämtar FI:s historiska blankningsfil..."
    )

    content = download_history()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ODS_PATH.write_bytes(
        content
    )

    df = pd.read_excel(
        io.BytesIO(content),
        sheet_name=0,
        engine="odf",
    )

    lines = analyse(df)

    report = (
        "\n".join(lines)
        + "\n"
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print(
        report,
        end="",
    )

    print(
        f"ODS sparad: {ODS_PATH}"
    )

    print(
        f"Rapport sparad: {REPORT_PATH}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
