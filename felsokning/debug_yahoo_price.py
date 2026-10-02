from __future__ import annotations

from datetime import date
from pathlib import Path

from prices.fetch import (
    _build_fetch_intervals,
    _download_batch,
    _existing_price_bounds,
)


SYMBOL = "SINCH.ST"
START = "2022-01-01"
END = "2026-10-01"

# Separat testintervall för att tvinga fram ett riktigt Yahoo-anrop.
# Detta påverkas inte av befintlig lokal prisdata.
YAHOO_TEST_START = "2026-09-01"
YAHOO_TEST_END = "2026-10-01"

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def print_records(
    records: list[dict],
) -> None:
    print()
    print(f"Returnerade records: {len(records):,}")

    if not records:
        print("INGEN DATA RETURNERADES.")
        return

    dates = [
        record["date"]
        for record in records
        if record.get("date")
    ]

    if dates:
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

    symbol_bounds = bounds.get(
        SYMBOL
    )

    if symbol_bounds is None:
        print("Ingen lokal historik hittades.")
    else:
        print(
            f"first: "
            f"{symbol_bounds['first'].isoformat()}"
        )
        print(
            f"last : "
            f"{symbol_bounds['last'].isoformat()}"
        )

    # ---------------------------------------------------------
    # 2. Befintlig produktionslogik
    # ---------------------------------------------------------

    print()
    print("2. FETCH-INTERVALL")
    print("-" * 70)

    requested_start = date.fromisoformat(
        START
    )

    effective_end = date.fromisoformat(
        END
    )

    intervals = _build_fetch_intervals(
        instrument=instrument,
        requested_start=requested_start,
        effective_end=effective_end,
        existing_bounds=bounds,
    )

    if not intervals:
        print(
            "Inga intervall skapades."
        )
        print(
            "Lokal historik täcker det "
            "begärda intervallet."
        )

    else:
        for (
            fetch_start,
            fetch_end,
            reason,
        ) in intervals:
            print(
                f"{reason}: "
                f"{fetch_start.isoformat()} -> "
                f"{fetch_end.isoformat()}"
            )

    # ---------------------------------------------------------
    # 3. Yahoo-hämtning enligt produktionslogiken
    # ---------------------------------------------------------

    print()
    print("3. YAHOO-HÄMTNING VIA FETCH-INTERVALL")
    print("-" * 70)

    if not intervals:
        print(
            "Hoppar över eftersom "
            "produktionslogiken inte skapade "
            "något intervall."
        )

    for (
        fetch_start,
        fetch_end,
        reason,
    ) in intervals:

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

        print_records(
            records
        )

    # ---------------------------------------------------------
    # 4. Tvingat direkt Yahoo-test
    #
    # Detta kringgår ENDAST den lokala intervallkontrollen.
    #
    # Själva Yahoo-hämtningen är exakt samma
    # _download_batch() som produktionen använder.
    # ---------------------------------------------------------

    print()
    print("4. DIREKT YAHOO-TEST")
    print("-" * 70)

    yahoo_test_start = date.fromisoformat(
        YAHOO_TEST_START
    )

    yahoo_test_end = date.fromisoformat(
        YAHOO_TEST_END
    )

    print(
        "Detta test ignorerar befintlig lokal "
        "prisdata och anropar direkt "
        "_download_batch()."
    )

    print()
    print(
        f"Testintervall: "
        f"{yahoo_test_start.isoformat()} -> "
        f"{yahoo_test_end.isoformat()}"
    )

    print()
    print(
        "Kör _download_batch()..."
    )

    direct_records = _download_batch(
        [instrument],
        yahoo_test_start,
        yahoo_test_end,
    )

    print_records(
        direct_records
    )

    # ---------------------------------------------------------
    # 5. Slutsats
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("SLUTSATS")
    print("=" * 70)

    print()

    if direct_records:
        print(
            "YAHOO TEST OK:"
        )
        print(
            "_download_batch() returnerade "
            f"{len(direct_records):,} records."
        )
        print(
            "Yahoo/yfinance och "
            "Close-tolkningen fungerar "
            "för testintervallet."
        )

    else:
        print(
            "YAHOO TEST MISSLYCKADES:"
        )
        print(
            "_download_batch() returnerade "
            "0 records för det direkta "
            "testintervallet."
        )
        print(
            "Problemet är därmed isolerat "
            "till Yahoo/yfinance eller "
            "hur Yahoo-svaret tolkas."
        )


if __name__ == "__main__":
    main()
