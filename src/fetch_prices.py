"""
Orkestrerar hämtning av prisdata för Blankdiss.

Datakedja:

    FI-data
      ↓
    yahoo_mapping.py
      ↓
    instrument_map.json
      ↓
    price_data.py
      ↓
    prices_YYYY-MM-DD.jsonl
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from .price_data import (
    fetch_prices,
    write_jsonl,
)
from .yahoo_mapping import (
    build_instrument_map,
    get_yahoo_symbols,
    load_instrument_map,
    read_all_fi_data,
    save_instrument_map,
)


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"


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

    print("Prisjobb: startar.")
    print(
        f"Prisjobb: startdatum = {start}"
    )

    # ---------------------------------------------------------
    # 1. Läs FI-data
    # ---------------------------------------------------------

    fi_records = read_all_fi_data()

    print(
        "Prisjobb: "
        f"{len(fi_records)} FI-observationer "
        "lästa."
    )

    # ---------------------------------------------------------
    # 2. Läs befintlig mapping
    # ---------------------------------------------------------

    existing_mapping = (
        load_instrument_map()
    )

    print(
        "Prisjobb: "
        f"{len(existing_mapping)} befintliga "
        "instrumentmappningar."
    )

    # ---------------------------------------------------------
    # 3. Komplettera mapping
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 4. Hämta mappade instrument
    # ---------------------------------------------------------

    instruments = get_yahoo_symbols(
        mapping
    )

    print(
        "Prisjobb: "
        f"{len(instruments)} instrument "
        "har Yahoo-symbol."
    )

    if not instruments:

        path = write_jsonl([])

        print(
            "Priser: 0 instrument "
            f"→ {path}"
        )

        print(
            "Mappning: "
            f"{new_mappings} nya, "
            f"{unresolved} olösta"
        )

        return

    # ---------------------------------------------------------
    # 5. Hämta prisdata
    # ---------------------------------------------------------

    records = fetch_prices(
        instruments=instruments,
        start=start,
        end=args.end,
    )

    # ---------------------------------------------------------
    # 6. Spara prisdata
    # ---------------------------------------------------------

    path = write_jsonl(
        records
    )

    symbols = {
        record["yahoo_symbol"]
        for record in records
    }

    print(
        "Priser: "
        f"{len(symbols)} instrument, "
        f"{len(records)} observationer "
        f"→ {path}"
    )

    print(
        "Mappning: "
        f"{new_mappings} nya, "
        f"{unresolved} olösta, "
        f"{len(instruments)} totalt"
    )

    print("Prisjobb: klart.")


if __name__ == "__main__":
    main()
