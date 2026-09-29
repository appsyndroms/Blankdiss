"""
CLI för FI-data i Blankdiss.

Hämtar FI:s aggregerade blankningsdata
och sparar den som tidsstämplade snapshots.

Körs med:

    python -m fi

För felsökning:

    python -m fi --current-only
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta

from .backfill import recover_missing_dates, snapshot_dates
from .current import fetch_current
from .errors import FIError
from .normalize import now_stockholm
from .storage import write_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hämta och normalisera "
            "Finansinspektionens "
            "aggregerade blankningsdata."
        )
    )

    parser.add_argument(
        "--current-only",
        action="store_true",
        help=(
            "Hämta aktuell FI-data utan "
            "historisk backfill."
        ),
    )

    return parser.parse_args()


def run_backfill() -> None:
    """
    Försöker återställa saknade FI-dagar
    innan aktuell snapshot hämtas.
    """

    existing = snapshot_dates()

    if existing:
        start = (
            min(existing)
            + timedelta(days=1)
        )
    else:
        start = None

    end = (
        now_stockholm().date()
        - timedelta(days=1)
    )

    if start is not None and start > end:
        print(
            "FI backfill: ingen historisk "
            "lucka att kontrollera."
        )
        return

    print()
    print(
        "=========================================="
    )
    print(
        "FI BACKFILL"
    )
    print(
        "=========================================="
    )

    recovered, unresolved = (
        recover_missing_dates(
            start=start,
            end=end,
        )
    )

    print()
    print(
        "FI backfill resultat:"
    )
    print(
        f"  återställda dagar: {recovered}"
    )
    print(
        f"  kvarvarande luckor: {unresolved}"
    )
    print()


def run_current() -> None:
    records = fetch_current()

    if not records:
        raise FIError(
            "FI returnerade inga "
            "aktuella observationer."
        )

    path = write_snapshot(
        records
    )

    source_dates = sorted(
        {
            record["source_date"]
            for record in records
            if record.get("source_date")
        }
    )

    print(
        "FI: aggregerad snapshot sparad."
    )

    print(
        f"FI: observationer = {len(records)}"
    )

    print(
        "FI: source_date = "
        f"{', '.join(source_dates)}"
    )

    print(
        f"FI: snapshot = {path}"
    )


def main() -> int:
    args = parse_args()

    try:
        if not args.current_only:
            run_backfill()

        run_current()

        if args.current_only:
            print(
                "FI: current-only klart."
            )
        else:
            print(
                "FI: klart."
            )

        return 0

    except FIError as exc:
        print(
            f"FI: FEL: {exc}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
