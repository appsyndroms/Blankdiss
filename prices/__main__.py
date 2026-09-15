"""
Orkestrerar hämtning av prisdata för Blankdiss.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

from .fetch import fetch_prices, write_jsonl
from .mapping import (
    build_instrument_map,
    get_yahoo_symbols,
    load_instrument_map,
    read_all_fi_data,
    save_instrument_map,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hämta historiska priser för "
            "mappade Blankdiss-instrument."
        )
    )

    parser.add_argument(
        "--start",
        default=None,
        help=(
            "Första datum för prisdata. "
            "Standard: 180 dagar bakåt."
        ),
    )

    parser.add_argument(
        "--end",
        default=None,
        help=(
            "Sista datum för prisdata. "
            "Yahoo använder slutdatum exklusivt."
        ),
    )

    args = parser.parse_args()

    if args.start is None:
        start = (
            date.today()
            - timedelta(days=180)
        ).isoformat()
    else:
        start = args.start

    print(
        "Prisjobb: startar."
    )

    print(
        f"Prisjobb: startdatum = {start}"
    )

    if args.end is not None:
        print(
            f"Prisjobb: slutdatum = {args.end}"
        )

    fi_records = read_all_fi_data()

    print(
        "Prisjobb: "
        f"{len(fi_records)} FI-observationer "
        "lästa."
    )

    existing_mapping = (
        load_instrument_map()
    )

    print(
        "Prisjobb: "
        f"{len(existing_mapping)} befintliga "
        "instrumentmappningar."
    )

    (
        mapping,
        new_mappings,
        unresolved,
    ) = build_instrument_map(
        existing=existing_mapping,
        fi_records=fi_records,
    )

    map_path = save_instrument_map(
        mapping
    )

    print(
        "Prisjobb: instrumentmappning "
        f"sparad → {map_path}"
    )

    instruments = get_yahoo_symbols(
        mapping
    )

    print(
        "Prisjobb: "
        f"{len(instruments)} instrument "
        "har Yahoo-symbol."
    )

    if not instruments:
        print(
            "Priser: inga instrument "
            "med Yahoo-symbol."
        )

        print(
            "Mappning: "
            f"{new_mappings} nya, "
            f"{unresolved} olösta"
        )

        return

    records = fetch_prices(
        instruments=instruments,
        start=start,
        end=args.end,
    )

    if records:
        path = write_jsonl(
            records,
            start=start,
            end=args.end,
        )

        symbols = {
            record["yahoo_symbol"]
            for record in records
        }

        print(
            "Priser: "
            f"{len(symbols)} instrument, "
            f"{len(records):,} nya observationer "
            f"→ {path}"
        )
    else:
        print(
            "Priser: inga nya observationer "
            "att skriva."
        )

    print(
        "Mappning: "
        f"{new_mappings} nya, "
        f"{unresolved} olösta, "
        f"{len(instruments)} totalt"
    )

    print(
        "Prisjobb: klart."
    )


if __name__ == "__main__":
    main()
