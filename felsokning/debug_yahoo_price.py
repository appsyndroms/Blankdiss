from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

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

PRICE_DIR = Path("data/raw/prices")


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


def _expected_weekdays(
    start: date,
    end: date,
) -> set[date]:
    expected: set[date] = set()

    current = start

    while current <= end:
        if current.weekday() < 5:
            expected.add(current)

        current += timedelta(days=1)

    return expected


def inspect_local_price_data() -> None:
    print()
    print("3. DETALJERAD KONTROLL AV LOKAL PRISDATA")
    print("-" * 70)

    files = sorted(
        PRICE_DIR.glob("prices_*.jsonl")
    )

    print(
        f"Prisfiler hittades: {len(files)}"
    )

    if not files:
        print("INGA PRISFILER HITTADES.")
        return

    frames: list[pd.DataFrame] = []

    matching_files = 0

    for path in files:
        try:
            frame = pd.read_json(
                path,
                lines=True,
            )

        except (
            ValueError,
            OSError,
        ) as exc:
            print(
                f"FEL vid läsning av {path.name}: "
                f"{exc}"
            )
            continue

        if "yahoo_symbol" not in frame.columns:
            continue

        matching = frame.loc[
            frame["yahoo_symbol"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq(SYMBOL)
        ].copy()

        if matching.empty:
            continue

        matching_files += 1
        matching["_source_file"] = path.name

        frames.append(
            matching
        )

    print(
        f"Filer som innehåller {SYMBOL}: "
        f"{matching_files}"
    )

    if not frames:
        print(
            f"INGA LOKALA RADER FÖR {SYMBOL}."
        )
        return

    frame = pd.concat(
        frames,
        ignore_index=True,
    )

    print(
        f"Lokala rader totalt: "
        f"{len(frame):,}"
    )

    # ---------------------------------------------------------
    # Datum
    # ---------------------------------------------------------

    if "date" not in frame.columns:
        print("FEL: kolumnen 'date' saknas.")
        return

    frame["_date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    ).dt.date

    invalid_dates = int(
        frame["_date"].isna().sum()
    )

    print(
        f"Ogiltiga datum: "
        f"{invalid_dates:,}"
    )

    valid_dates = frame.loc[
        frame["_date"].notna()
    ]

    if valid_dates.empty:
        print("INGA GILTIGA DATUM.")
        return

    first_date = valid_dates[
        "_date"
    ].min()

    last_date = valid_dates[
        "_date"
    ].max()

    print(
        f"Faktiskt första datum: "
        f"{first_date.isoformat()}"
    )

    print(
        f"Faktiskt sista datum : "
        f"{last_date.isoformat()}"
    )

    # ---------------------------------------------------------
    # Unika datum och dubbletter
    # ---------------------------------------------------------

    unique_dates = set(
        valid_dates["_date"]
    )

    duplicate_rows = (
        len(valid_dates)
        - len(unique_dates)
    )

    print()
    print(
        f"Unika datum: "
        f"{len(unique_dates):,}"
    )

    print(
        f"Extra rader utöver unika datum: "
        f"{duplicate_rows:,}"
    )

    duplicate_date_counts = (
        valid_dates["_date"]
        .value_counts()
    )

    duplicate_dates = (
        duplicate_date_counts[
            duplicate_date_counts > 1
        ]
    )

    print(
        f"Datum med flera rader: "
        f"{len(duplicate_dates):,}"
    )

    if not duplicate_dates.empty:
        print()
        print("Exempel på dubblettdatum:")

        for duplicate_date, count in (
            duplicate_dates
            .sort_index()
            .head(10)
            .items()
        ):
            print(
                f"  {duplicate_date.isoformat()}: "
                f"{count} rader"
            )

    # ---------------------------------------------------------
    # Vardagsluckor
    # ---------------------------------------------------------

    print()
    print("KONTROLL AV VARDAGSLUCKOR")
    print("-" * 70)

    expected = _expected_weekdays(
        first_date,
        last_date,
    )

    missing_dates = sorted(
        expected - unique_dates
    )

    print(
        f"Förväntade vardagar mellan "
        f"{first_date.isoformat()} och "
        f"{last_date.isoformat()}: "
        f"{len(expected):,}"
    )

    print(
        f"Observerade datum: "
        f"{len(unique_dates):,}"
    )

    print(
        f"Potentiella vardagsluckor: "
        f"{len(missing_dates):,}"
    )

    if missing_dates:
        print()
        print(
            "Första 20 potentiella luckor:"
        )

        for missing_date in missing_dates[:20]:
            print(
                f"  {missing_date.isoformat()}"
            )

    # ---------------------------------------------------------
    # Close
    # ---------------------------------------------------------

    print()
    print("KONTROLL AV CLOSE")
    print("-" * 70)

    if "close" not in frame.columns:
        print(
            "FEL: kolumnen 'close' saknas."
        )

    else:
        close_numeric = pd.to_numeric(
            frame["close"],
            errors="coerce",
        )

        invalid_close = (
            close_numeric.isna()
        )

        nonfinite_close = (
            close_numeric
            .notna()
            .map(
                lambda value: False
            )
        )

        # Kontrollera NaN/inf explicit.
        finite_mask = close_numeric.apply(
            lambda value: (
                False
                if pd.isna(value)
                else math.isfinite(
                    float(value)
                )
            )
        )

        invalid_close_count = int(
            (~finite_mask).sum()
        )

        print(
            f"Ogiltiga/icke-finit close: "
            f"{invalid_close_count:,}"
        )

        if invalid_close_count:
            print(
                "Exempel på rader med "
                "ogiltigt close:"
            )

            invalid_rows = frame.loc[
                ~finite_mask
            ]

            columns = [
                column
                for column in [
                    "date",
                    "isin",
                    "issuer",
                    "yahoo_symbol",
                    "close",
                    "_source_file",
                ]
                if column in invalid_rows.columns
            ]

            for row in (
                invalid_rows[columns]
                .head(10)
                .to_dict("records")
            ):
                print(row)

        valid_close = close_numeric.loc[
            finite_mask
        ]

        if not valid_close.empty:
            print(
                f"Min close: "
                f"{valid_close.min()}"
            )

            print(
                f"Max close: "
                f"{valid_close.max()}"
            )

    # ---------------------------------------------------------
    # Identitet på prisraderna
    # ---------------------------------------------------------

    print()
    print("KONTROLL AV IDENTITET PÅ PRISRADER")
    print("-" * 70)

    for column in [
        "isin",
        "lei",
        "issuer",
        "ticker",
        "yahoo_symbol",
        "mapping_source",
    ]:
        if column not in frame.columns:
            print(
                f"{column}: KOLUMN SAKNAS"
            )
            continue

        values = (
            frame[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        non_empty = values[
            values != ""
        ]

        print(
            f"{column}: "
            f"{non_empty.nunique():,} unika värden, "
            f"{(values == '').sum():,} tomma"
        )

        if (
            column != "issuer"
            and non_empty.nunique() <= 5
            and not non_empty.empty
        ):
            print(
                f"  värden: "
                f"{sorted(non_empty.unique())}"
            )

    # ---------------------------------------------------------
    # Filfördelning
    # ---------------------------------------------------------

    print()
    print("FÖRDELNING PER PRISFIL")
    print("-" * 70)

    per_file = (
        frame.groupby(
            "_source_file"
        )
        .agg(
            rows=("_date", "size"),
            first=("_date", "min"),
            last=("_date", "max"),
            unique_dates=(
                "_date",
                "nunique",
            ),
        )
        .sort_index()
    )

    for filename, row in per_file.iterrows():
        print(
            f"{filename}: "
            f"{int(row['rows']):,} rader, "
            f"{int(row['unique_dates']):,} unika datum, "
            f"{row['first']} -> {row['last']}"
        )

    # ---------------------------------------------------------
    # Sammanfattning
    # ---------------------------------------------------------

    print()
    print("LOKAL DATA – SAMMANFATTNING")
    print("-" * 70)

    problems = []

    if invalid_dates:
        problems.append(
            f"{invalid_dates} ogiltiga datum"
        )

    if duplicate_dates.any():
        problems.append(
            f"{len(duplicate_dates)} datum med dubbletter"
        )

    if missing_dates:
        problems.append(
            f"{len(missing_dates)} potentiella vardagsluckor"
        )

    if "close" in frame.columns:
        close_numeric = pd.to_numeric(
            frame["close"],
            errors="coerce",
        )

        finite_mask = close_numeric.apply(
            lambda value: (
                False
                if pd.isna(value)
                else math.isfinite(
                    float(value)
                )
            )
        )

        invalid_close_count = int(
            (~finite_mask).sum()
        )

        if invalid_close_count:
            problems.append(
                f"{invalid_close_count} ogiltiga close"
            )

    if problems:
        print(
            "PROBLEM HITTADES:"
        )

        for problem in problems:
            print(
                f"  - {problem}"
            )

    else:
        print(
            "Inga uppenbara strukturella problem "
            "hittades i den lokala Sinch-datan."
        )


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
        PRICE_DIR
    )

    symbol_bounds = bounds.get(
        SYMBOL
    )

    if symbol_bounds is None:
        print(
            "Ingen lokal historik hittades."
        )

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
    # 2. Produktionslogik
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
    # 3. Detaljerad kontroll av lokal data
    # ---------------------------------------------------------

    inspect_local_price_data()

    # ---------------------------------------------------------
    # 4. Yahoo-hämtning enligt produktionslogiken
    # ---------------------------------------------------------

    print()
    print("4. YAHOO-HÄMTNING VIA FETCH-INTERVALL")
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
    # 5. Tvingat direkt Yahoo-test
    # ---------------------------------------------------------

    print()
    print("5. DIREKT YAHOO-TEST")
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
    # 6. Slutsats
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
