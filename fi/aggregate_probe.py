"""Probar om FI:s aggregat-endpoint stöder historiska datum."""
from __future__ import annotations
import argparse
import hashlib
from io import BytesIO
import pandas as pd
import requests
from .config import (
    FI_AGGREGATE_TIMEOUT,
    FI_AGGREGATE_URL,
    HEADERS,
)
from .errors import FIError
DEFAULT_DATE = "2022-06-09"
PARAMETER_NAMES = [
    "date",
    "datum",
    "source_date",
    "position_date",
    "positionsdatum",
]
def read_position_dates(
    data: bytes,
) -> list[str]:
    """Läser positionsdatum från FI:s ODS-svar."""
    try:
        table = pd.read_excel(
            BytesIO(data),
            engine="odf",
            skiprows=6,
            header=None,
        )
    except Exception as exc:
        raise FIError(
            "Kunde inte läsa FI:s ODS-svar: "
            f"{exc}"
        ) from exc
    if table.empty or len(table.columns) < 4:
        raise FIError(
            "FI:s ODS-svar hade oväntat format. "
            f"shape={table.shape}"
        )
    dates = (
        pd.to_datetime(
            table.iloc[:, 3],
            errors="coerce",
        )
        .dropna()
        .dt.date
        .astype(str)
        .tolist()
    )
    return sorted(set(dates))
def fetch(
    session: requests.Session,
    params: dict[str, str] | None = None,
) -> tuple[bytes, list[str]]:
    """Hämtar FI:s aggregatfil."""
    response = session.get(
        FI_AGGREGATE_URL,
        headers=HEADERS,
        params=params,
        timeout=FI_AGGREGATE_TIMEOUT,
        allow_redirects=True,
    )
    print(
        "FI probe: "
        f"HTTP={response.status_code}, "
        "Content-Type="
        f"{response.headers.get('Content-Type', '')!r}, "
        f"bytes={len(response.content)}, "
        f"url={response.url}"
    )
    if response.status_code != 200:
        raise FIError(
            "FI returnerade HTTP "
            f"{response.status_code}."
        )
    return (
        response.content,
        read_position_dates(
            response.content
        ),
    )
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probar om FI:s aggregat-endpoint "
            "kan returnera historiska datum."
        )
    )
    parser.add_argument(
        "--date",
        default=DEFAULT_DATE,
        help=(
            "Historiskt datum som ska provas, "
            "YYYY-MM-DD."
        ),
    )
    return parser.parse_args()
def main() -> int:
    args = parse_args()
    session = requests.Session()
    print(
        "FI probe: grundanrop"
    )
    baseline_data, baseline_dates = fetch(
        session
    )
    baseline_hash = hashlib.sha256(
        baseline_data
    ).hexdigest()
    print(
        "FI probe: grundsvar SHA-256="
        f"{baseline_hash}"
    )
    print(
        "FI probe: positionsdatum från "
        f"{min(baseline_dates)} "
        "till "
        f"{max(baseline_dates)}"
    )
    print()
    print(
        "FI probe: testar historiskt datum "
        f"{args.date}"
    )
    found = False
    for parameter_name in PARAMETER_NAMES:
        print()
        print(
            "FI probe: parameter="
            f"{parameter_name}"
        )
        data, dates = fetch(
            session,
            {
                parameter_name: args.date,
            },
        )
        digest = hashlib.sha256(
            data
        ).hexdigest()
        same_as_baseline = (
            digest == baseline_hash
        )
        contains_target = (
            args.date in dates
        )
        print(
            "FI probe: "
            f"sha_match_baseline="
            f"{same_as_baseline}, "
            "contains_target_date="
            f"{contains_target}, "
            "date_range="
            f"{min(dates) if dates else 'saknas'}"
            ".."
            f"{max(dates) if dates else 'saknas'}"
        )
        if contains_target and not same_as_baseline:
            found = True
            print(
                "FI probe: "
                "HISTORIK FUNNEN via "
                f"{parameter_name}="
                f"{args.date}"
            )
    print()
    if found:
        print(
            "FI probe: historisk endpoint "
            "identifierad."
        )
    else:
        print(
            "FI probe: inget av de testade "
            "datumparametrarna gav historisk data."
        )
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
