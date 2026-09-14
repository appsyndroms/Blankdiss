"""CLI för FI-data i Blankdiss."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from .current import fetch_current
from .errors import FIError
from .historical import fetch_historical
from .storage import write_snapshot


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Hämta och normalisera "
            "Finansinspektionens blankningsdata."
        )
    )

    parser.add_argument(
        "--days",
        type=int,
        default=0,
        help=(
            "Antal dagar historik som ska "
            "hämtas före idag."
        ),
    )

    parser.add_argument(
        "--current-only",
        action="store_true",
        help=(
            "Hämta endast aktuell data. "
            "Används främst för felsökning."
        ),
    )

    return parser.parse_args()


def run_current() -> None:

    records = fetch_current()

    if not records:
        raise FIError(
            "FI returnerade inga aktuella observationer."
        )

    path = write_snapshot(records)

    print(
        "FI: aktuell snapshot - "
        f"{len(records)} observationer → {path}"
    )


def run_historical(
    days: int,
) -> None:

    if days <= 0:
        return

    end_date = date.today()
    start_date = (
        end_date
        - timedelta(days=days - 1)
    )

    print(
        "FI: backfill "
        f"{start_date} → {end_date}"
    )

    records = fetch_historical(
        start_date,
        end_date,
    )

    if not records:
        raise FIError(
            "FI: historisk backfill "
            "returnerade inga observationer."
        )

    path = write_snapshot(records)

    print(
        "FI: historisk snapshot - "
        f"{len(records)} observationer → {path}"
    )


def main() -> int:

    args = parse_args()

    try:

        if args.current_only:
            run_current()
            return 0

        if args.days > 0:
            run_historical(args.days)

        run_current()

        print("FI: klart.")
        return 0

    except FIError as exc:

        print(
            f"FI: FEL: {exc}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
