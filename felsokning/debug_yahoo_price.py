from __future__ import annotations

from pathlib import Path

from prices.fetch import (
    _build_fetch_intervals,
    _download_batch,
    _existing_price_bounds,
)


SYMBOL = "SINCH.ST"
START = "2022-01-01"
END = "2026-10-01"

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print("=" * 70)
    print("SINCH / YAHOO FELSÖKNING")
    print("=" * 70)

    instrument = {
        "isin": "SE0016101844",
        "issuer": "Sinch AB (publ)",
        "lei": "549300UXY7QM6IDCGI12",
        "ticker": None,
        "yahoo_symbol": SYMBOL,
        "mapping_source": "known_name",
    }

    print()
    print("INSTRUMENT")
    print("-" * 70)
    for key, value in instrument.items():
        print(f"{key}: {value}")

    # ---------------------------------------------------------
    # 1. Befintlig lokal prisstatus
    # ---------------------------------------------------------

    print()
    print("1. LOKAL PRISHISTORIK")
    print("-" * 70)

    bounds = _existing_price_bounds(
        Path("data/raw/prices")
    )

    symbol_bounds = bounds.get(SYMBOL)

    if symbol_bounds is None:
        print("Ingen lokal historik hittades.")
    else:
        print(
            f"first: {symbol_bounds['first'].isoformat()}"
        )
        print(
            f"last : {symbol_bounds['last'].isoformat()}"
        )

    # ---------------------------------------------------------
    # 2. Befintlig intervallberäkning
    # ---------------------------------------------------------

    print()
    print("2. FETCH-INTERVALL")
    print("-" * 70)

    from datetime import date

    requested_start = date.fromisoformat(START)
    effective_end = date.fromisoformat(END)

    intervals = _build_fetch_intervals(
        instrument=instrument,
        requested_start=requested_start,
        effective_end=effective_end,
        existing_bounds=bounds,
    )

    if not intervals:
        print("Inga intervall skapades.")
    else:
        for fetch_start, fetch_end, reason in intervals:
            print(
                f"{reason}: "
                f"{fetch_start.isoformat()} -> "
                f"{fetch_end.isoformat()}"
            )

    # ---------------------------------------------------------
    # 3. Använd exakt befintlig Yahoo-funktion
    # ---------------------------------------------------------

    print()
    print("3. BEFINTLIG YAHOO-HÄMTNING")
    print("-" * 70)

    for fetch_start, fetch_end, reason in intervals:
        print()
        print(
            f"Kör _download_batch(): "
            f"{reason} "
            f"{fetch_start.isoformat()} -> "
            f"{fetch_end.isoformat()}"
        )

        records = _download_batch(
            [instrument],
            fetch_start,
            fetch_end,
        )

        print()
        print(f"Returnerade records: {len(records):,}")

        if not records:
            print("INGEN DATA RETURNERADES.")
            continue

        dates = [
            record["date"]
            for record in records
            if record.get("date")
        ]

        print(
            f"Första record: {min(dates)}"
        )
        print(
            f"Sista record : {max(dates)}"
        )

        print()
        print("Första 5 records:")

        for record in records[:5]:
            print(record)

        print()
        print("Sista 5 records:")

        for record in records[-5:]:
            print(record)

    # ---------------------------------------------------------
    # Slutsats
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("SLUTSATS")
    print("=" * 70)

    print()
    print(
        "Om backfill skapas men _download_batch() "
        "returnerar 0 records är problemet i Yahoo/yfinance "
        "eller hur svaret tolkas."
    )

    print()
    print(
        "Om records returneras fungerar Yahoo-hämtningen "
        "och felet ligger efter _download_batch()."
    )


if __name__ == "__main__":
    main()
