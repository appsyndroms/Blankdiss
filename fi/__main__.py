"""
CLI för FI-data i Blankdiss.

Hämtar aktuell aggregerad blankningsdata från
Finansinspektionens blankningsregister och sparar
den som en tidsstämplad snapshot.

Körs med:

    python -m fi

För felsökning:

    python -m fi --current-only
"""

from __future__ import annotations

import argparse
import sys

from .current import fetch_current
from .errors import FIError
from .storage import write_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hämta och normalisera "
            "Finansinspektionens blankningsdata."
        )
    )

    parser.add_argument(
        "--current-only",
        action="store_true",
        help=(
            "Hämta endast aktuell data. "
            "Flaggan finns för tydlighet och felsökning."
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


def main() -> int:
    args = parse_args()

    try:
        run_current()

        if args.current_only:
            print("FI: current-only klart.")
        else:
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
