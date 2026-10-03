from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, timedelta
from itertools import combinations
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

YAHOO_TEST_START = "2026-09-01"
YAHOO_TEST_END = "2026-10-01"

PRICE_DIR = Path(
    "data/raw/prices"
)


def print_records(
    title: str,
    records: list[dict],
) -> None:
    print()
    print(title)
    print("-" * 70)

    for record in records:
        print(record)


def _expected_weekdays(
    start_date: date,
    end_date: date,
) -> list[date]:
    dates: list[date] = []

    current = start_date

    while current <= end_date:
        if current.weekday() < 5:
            dates.append(current)

        current += timedelta(days=1)

    return dates


def _load_local_price_data() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for path in sorted(
        PRICE_DIR.glob(
            "prices_*.jsonl"
        )
    ):
        try:
            frame = pd.read_json(
                path,
                lines=True,
            )

        except (
            ValueError,
            OSError,
        ):
            continue

        if (
            "yahoo_symbol"
            not in frame.columns
        ):
            continue

        frame = frame.loc[
            frame[
                "yahoo_symbol"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq(SYMBOL)
        ].copy()

        if frame.empty:
            continue

        frame["_source_file"] = (
            path.name
        )

        frame["_date"] = pd.to_datetime(
            frame["date"],
            errors="coerce",
        ).dt.date

        frame["_close_numeric"] = (
            pd.to_numeric(
                frame["close"],
                errors="coerce",
            )
        )

        frames.append(
            frame
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def _print_value_distribution(
    frame: pd.DataFrame,
    column: str,
) -> None:
    print()
    print(
        f"{column.upper()}:"
    )

    if column not in frame.columns:
        print("  kolumn saknas")
        return

    values = (
        frame[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    counts = (
        values.value_counts(
            dropna=False
        )
    )

    for value, count in counts.items():
        print(
            f"  {value or '<tom>'}: "
            f"{count:,}"
        )


def _normalise_comparison_value(
    value,
):
    if pd.isna(value):
        return None

    if isinstance(
        value,
        float,
    ):
        if math.isnan(value):
            return None

        return value

    return value


def _compare_overlapping_files(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[int, int, int]:
    left_by_date = {
        row["_date"]: row
        for _, row in left.iterrows()
    }

    right_by_date = {
        row["_date"]: row
        for _, row in right.iterrows()
    }

    common_dates = sorted(
        set(
            left_by_date
        ).intersection(
            right_by_date
        )
    )

    identical = 0
    different = 0

    for current_date in common_dates:
        left_row = left_by_date[
            current_date
        ]

        right_row = right_by_date[
            current_date
        ]

        columns = sorted(
            set(left_row.index).intersection(
                right_row.index
            )
        )

        same = True

        for column in columns:
            left_value = (
                _normalise_comparison_value(
                    left_row[column]
                )
            )

            right_value = (
                _normalise_comparison_value(
                    right_row[column]
                )
            )

            if (
                left_value
                != right_value
            ):
                same = False
                break

        if same:
            identical += 1
        else:
            different += 1

    return (
        len(common_dates),
        identical,
        different,
    )


def test_instrument_coverage_logic() -> None:
    """
    Verifierar att samma Yahoo-symbol kan tillhöra flera instrument.

    Testet använder endast befintliga lokala bounds och
    _build_fetch_intervals(). Det anropar inte Yahoo och skriver
    inget till disk.
    """
    print()
    print(
        "TEST: INSTRUMENTTÄCKNING / ISIN"
    )
    print("-" * 70)

    old_instrument = {
        "isin": "SE0007439112",
        "issuer": "Sinch AB (publ)",
        "lei": "549300UXY7QM6IDCGI12",
        "ticker": None,
        "yahoo_symbol": SYMBOL,
        "mapping_source": "test",
    }

    new_instrument = {
        "isin": "SE0016101844",
        "issuer": "Sinch AB (publ)",
        "lei": "549300UXY7QM6IDCGI12",
        "ticker": None,
        "yahoo_symbol": SYMBOL,
        "mapping_source": "test",
    }

    bounds = _existing_price_bounds(
        PRICE_DIR
    )

    old_key = (
        f"isin:{old_instrument['isin']}"
    )

    new_key = (
        f"isin:{new_instrument['isin']}"
    )

    print(
        f"Gammal instrumentnyckel: "
        f"{old_key}"
    )

    print(
        f"Ny instrumentnyckel    : "
        f"{new_key}"
    )

    print()
    print(
        "Lokal täckning:"
    )

    old_bounds = bounds.get(
        old_key
    )

    new_bounds = bounds.get(
        new_key
    )

    if old_bounds is None:
        print(
            "  Gammal ISIN: "
            "ingen lokal historik"
        )

    else:
        print(
            "  Gammal ISIN: "
            f"{old_bounds['first']} -> "
            f"{old_bounds['last']}"
        )

    if new_bounds is None:
        print(
            "  Ny ISIN    : "
            "ingen lokal historik"
        )

    else:
        print(
            "  Ny ISIN    : "
            f"{new_bounds['first']} -> "
            f"{new_bounds['last']}"
        )

    requested_start = date.fromisoformat(
        START
    )

    effective_end = date.fromisoformat(
        END
    )

    old_intervals = (
        _build_fetch_intervals(
            instrument=old_instrument,
            requested_start=requested_start,
            effective_end=effective_end,
            existing_bounds=bounds,
        )
    )

    new_intervals = (
        _build_fetch_intervals(
            instrument=new_instrument,
            requested_start=requested_start,
            effective_end=effective_end,
            existing_bounds=bounds,
        )
    )

    print()
    print(
        "Fetch-intervall:"
    )

    if old_intervals:
        for (
            fetch_start,
            fetch_end,
            reason,
        ) in old_intervals:
            print(
                "  Gammal ISIN: "
                f"{reason} "
                f"{fetch_start} -> "
                f"{fetch_end}"
            )

    else:
        print(
            "  Gammal ISIN: "
            "inget intervall"
        )

    if new_intervals:
        for (
            fetch_start,
            fetch_end,
            reason,
        ) in new_intervals:
            print(
                "  Ny ISIN    : "
                f"{reason} "
                f"{fetch_start} -> "
                f"{fetch_end}"
            )

    else:
        print(
            "  Ny ISIN    : "
            "inget intervall"
        )

    print()
    print(
        "Förväntat:"
    )

    print(
        "  Gammal ISIN -> "
        "lokal historik finns"
    )

    print(
        "  Ny ISIN     -> "
        "separat instrument, "
        "ska inte ärva gammal ISIN:s täckning"
    )

    passed = (
        old_bounds is not None
        and new_bounds is None
        and not old_intervals
        and bool(new_intervals)
    )

    print()

    if passed:
        print(
            "RESULTAT: PASS"
        )

        print(
            "Instrumenttäckningen är separerad "
            "per ISIN trots samma Yahoo-symbol."
        )

    else:
        print(
            "RESULTAT: FAIL"
        )

        print(
            "Instrumenttäckningen fungerar inte "
            "som förväntat."
        )


def inspect_local_price_data(
    frame: pd.DataFrame,
) -> None:
    print()
    print(
        "4. DETALJERAD KONTROLL AV LOKAL PRISDATA"
    )
    print("-" * 70)

    if frame.empty:
        print(
            "Ingen lokal prisdata hittades."
        )

        return

    print(
        "Filer som innehåller "
        f"{SYMBOL}: "
        f"{frame['_source_file'].nunique()}"
    )

    print(
        "Lokala rader totalt: "
        f"{len(frame):,}"
    )

    invalid_dates = (
        frame["_date"].isna().sum()
    )

    print(
        "Ogiltiga datum: "
        f"{invalid_dates:,}"
    )

    valid_dates = frame.loc[
        frame["_date"].notna(),
        "_date",
    ]

    if not valid_dates.empty:
        print(
            "Faktiskt första datum: "
            f"{valid_dates.min()}"
        )

        print(
            "Faktiskt sista datum : "
            f"{valid_dates.max()}"
        )

    unique_dates = (
        frame["_date"]
        .dropna()
        .nunique()
    )

    duplicate_rows = (
        len(frame)
        - unique_dates
    )

    print()
    print(
        f"Unika datum: "
        f"{unique_dates:,}"
    )

    print(
        "Extra rader utöver unika datum: "
        f"{duplicate_rows:,}"
    )

    duplicate_dates = (
        frame.groupby(
            "_date"
        )
        .size()
    )

    duplicate_dates = (
        duplicate_dates[
            duplicate_dates > 1
        ]
    )

    print(
        "Datum med flera rader: "
        f"{len(duplicate_dates):,}"
    )

    print()
    print(
        "KONTROLL AV ISIN-HISTORIK"
    )

    if "isin" not in frame.columns:
        print(
            "isin-kolumn saknas."
        )

    else:
        isin_values = (
            frame["isin"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        unique_isins = sorted(
            value
            for value in isin_values.unique()
            if value
        )

        empty_isins = (
            isin_values.eq("").sum()
        )

        print(
            "isin: "
            f"{len(unique_isins)} "
            "unika värden, "
            f"{empty_isins:,} tomma"
        )

        print(
            "  värden: "
            f"{unique_isins}"
        )

        print(
            "Datumintervall per ISIN:"
        )

        for isin in unique_isins:
            isin_frame = frame.loc[
                isin_values.eq(isin)
            ]

            dates = isin_frame[
                "_date"
            ].dropna()

            print(
                f"  {isin}: "
                f"{len(isin_frame):,} rader, "
                f"{dates.nunique():,} unika datum, "
                f"{dates.min()} -> "
                f"{dates.max()}"
            )

    print()
    print(
        "KONTROLL AV DATUM + ISIN"
    )

    if "isin" not in frame.columns:
        print(
            "isin-kolumn saknas."
        )

    else:
        duplicate_date_isin = (
            frame.groupby(
                [
                    "_date",
                    "isin",
                ],
                dropna=False,
            )
            .size()
        )

        duplicate_date_isin = (
            duplicate_date_isin[
                duplicate_date_isin > 1
            ]
        )

        print(
            "Dubbletter på datum + ISIN: "
            f"{len(duplicate_date_isin):,}"
        )

        if not duplicate_date_isin.empty:
            print(
                "Första 20:"
            )

            for key, count in (
                duplicate_date_isin
                .head(20)
                .items()
            ):
                print(
                    f"  {key}: "
                    f"{count} rader"
                )

    print()
    print(
        "KONTROLL AV SAMMA DATUM MED FLERA ISIN"
    )

    if "isin" not in frame.columns:
        print(
            "isin-kolumn saknas."
        )

    else:
        date_isin_counts = (
            frame.groupby(
                "_date"
            )["isin"]
            .nunique(
                dropna=True
            )
        )

        multiple_isins = (
            date_isin_counts[
                date_isin_counts > 1
            ]
        )

        print(
            "Datum med flera ISIN: "
            f"{len(multiple_isins):,}"
        )

        if not multiple_isins.empty:
            print(
                "Första 20:"
            )

            for current_date, count in (
                multiple_isins
                .head(20)
                .items()
            ):
                print(
                    f"  {current_date}: "
                    f"{count} ISIN"
                )

    print()
    print(
        "KONTROLL AV DUBLETTER OCH CLOSE"
    )

    close_per_date = (
        frame.groupby(
            "_date"
        )["_close_numeric"]
        .nunique(
            dropna=False
        )
    )

    conflicting_close_dates = (
        close_per_date[
            close_per_date > 1
        ]
    )

    print(
        "Datum med flera olika close-värden: "
        f"{len(conflicting_close_dates):,}"
    )

    if conflicting_close_dates.empty:
        print(
            "Alla dubbletter har samma "
            "close-värde per datum."
        )

    else:
        print(
            "Första 20:"
        )

        for current_date in (
            conflicting_close_dates
            .head(20)
            .index
        ):
            values = sorted(
                frame.loc[
                    frame["_date"].eq(
                        current_date
                    ),
                    "_close_numeric",
                ]
                .dropna()
                .unique()
                .tolist()
            )

            print(
                f"  {current_date}: "
                f"{values}"
            )

    print()
    print(
        "KONTROLL AV ÖVERLAPPANDE PRISFILER"
    )

    file_frames = {
        name: group.copy()
        for name, group in frame.groupby(
            "_source_file"
        )
    }

    for name in sorted(
        file_frames
    ):
        file_frame = file_frames[
            name
        ]

        dates = (
            file_frame["_date"]
            .dropna()
        )

        print(
            f"{name}: "
            f"{len(file_frame):,} rows, "
            f"{dates.min()} -> "
            f"{dates.max()}"
        )

    overlap_results = []

    for left_name, right_name in combinations(
        sorted(file_frames),
        2,
    ):
        left = file_frames[
            left_name
        ]

        right = file_frames[
            right_name
        ]

        left_dates = set(
            left["_date"]
            .dropna()
        )

        right_dates = set(
            right["_date"]
            .dropna()
        )

        common_dates = (
            left_dates.intersection(
                right_dates
            )
        )

        if not common_dates:
            continue

        overlap_results.append(
            (
                left_name,
                right_name,
                min(common_dates),
                max(common_dates),
                len(common_dates),
            )
        )

    print()

    print(
        "Overlapping file pairs: "
        f"{len(overlap_results)}:"
    )

    for (
        left_name,
        right_name,
        first,
        last,
        count,
    ) in overlap_results:
        print(
            f"- {left_name.removesuffix('.jsonl')} "
            f"↔ "
            f"{right_name.removesuffix('.jsonl')}: "
            f"{first} -> {last}, "
            f"{count:,} common dates"
        )

    print()
    print(
        "KONTROLL AV HELA RADER I FILÖVERLAPP"
    )

    total_common = 0
    total_identical = 0
    total_different = 0

    for (
        left_name,
        right_name,
        _,
        _,
        _,
    ) in overlap_results:
        common, identical, different = (
            _compare_overlapping_files(
                file_frames[left_name],
                file_frames[right_name],
            )
        )

        total_common += common
        total_identical += identical
        total_different += different

        print(
            f"{left_name.removesuffix('.jsonl')} "
            f"↔ "
            f"{right_name.removesuffix('.jsonl')}:"
        )

        print(
            f"  Gemensamma datum: "
            f"{common:,}"
        )

        print(
            f"  Helt identiska rader: "
            f"{identical:,}"
        )

        print(
            f"  Datum med skillnader: "
            f"{different:,}"
        )

    print()
    print(
        "TOTAL JÄMFÖRELSE"
    )

    print(
        "Gemensamma datum: "
        f"{total_common:,}"
    )

    print(
        "Helt identiska rader: "
        f"{total_identical:,}"
    )

    print(
        "Datum med skillnader: "
        f"{total_different:,}"
    )

    if total_different == 0:
        print(
            "Alla överlappande prisrader "
            "är identiska."
        )

    print()
    print(
        "VIRTUELL DEDUPLICERING"
    )

    dedupe_columns = [
        "_date",
        "yahoo_symbol",
    ]

    dedupe_frame = (
        frame.drop_duplicates(
            subset=dedupe_columns,
            keep="first",
        )
    )

    print(
        "Rader före: "
        f"{len(frame):,}"
    )

    print(
        "Rader efter datum + Yahoo-symbol: "
        f"{len(dedupe_frame):,}"
    )

    print(
        "Rader som skulle tas bort: "
        f"{len(frame) - len(dedupe_frame):,}"
    )

    if "isin" in frame.columns:
        print()
        print(
            "VIRTUELL DEDUPLICERING PÅ DATUM + ISIN"
        )

        isin_dedupe = (
            frame.drop_duplicates(
                subset=[
                    "_date",
                    "isin",
                ],
                keep="first",
            )
        )

        print(
            "Rader före: "
            f"{len(frame):,}"
        )

        print(
            "Rader efter datum + ISIN: "
            f"{len(isin_dedupe):,}"
        )

        print(
            "Rader som skulle tas bort: "
            f"{len(frame) - len(isin_dedupe):,}"
        )

    print()
    print(
        "KONTROLL AV VARDAGSLUCKOR"
    )

    if valid_dates.empty:
        print(
            "Ingen giltig lokal datumdata."
        )

    else:
        first_date = valid_dates.min()
        last_date = valid_dates.max()

        expected = set(
            _expected_weekdays(
                first_date,
                last_date,
            )
        )

        observed = set(
            valid_dates.unique()
        )

        missing = sorted(
            expected - observed
        )

        print(
            "Förväntade vardagar: "
            f"{len(expected):,}"
        )

        print(
            "Observerade: "
            f"{len(observed):,}"
        )

        print(
            "Potentiella luckor: "
            f"{len(missing):,}"
        )

        if missing:
            print(
                "Första 50:"
            )

            for current_date in missing[
                :50
            ]:
                print(
                    f"  {current_date}"
                )

    print()
    print(
        "KONTROLL AV CLOSE"
    )

    invalid_close = (
        frame["_close_numeric"]
        .isna()
        .sum()
    )

    nonfinite_close = (
        ~frame[
            "_close_numeric"
        ].map(
            lambda value: (
                math.isfinite(value)
                if pd.notna(value)
                else False
            )
        )
    ).sum()

    print(
        "Ogiltiga/non-finite: "
        f"{invalid_close + nonfinite_close:,}"
    )

    valid_close = frame.loc[
        frame["_close_numeric"].notna(),
        "_close_numeric",
    ]

    valid_close = valid_close[
        valid_close.map(
            math.isfinite
        )
    ]

    if not valid_close.empty:
        print(
            "Min "
            f"{valid_close.min()}"
        )

        print(
            "Max "
            f"{valid_close.max()}"
        )

    print()
    print(
        "IDENTITY"
    )

    _print_value_distribution(
        frame,
        "isin",
    )

    _print_value_distribution(
        frame,
        "lei",
    )

    _print_value_distribution(
        frame,
        "issuer",
    )

    _print_value_distribution(
        frame,
        "ticker",
    )

    _print_value_distribution(
        frame,
        "yahoo_symbol",
    )

    _print_value_distribution(
        frame,
        "mapping_source",
    )

    print()
    print(
        "FÖRDELNING PER PRISFIL"
    )

    distribution = (
        frame["_source_file"]
        .value_counts()
        .sort_index()
    )

    for filename, count in (
        distribution.items()
    ):
        print(
            f"  {filename}: "
            f"{count:,}"
        )

    print()
    print(
        "SUMMARY"
    )

    problems: list[str] = []

    if len(duplicate_dates) > 0:
        problems.append(
            f"{len(duplicate_dates):,} "
            "datum med dubbletter"
        )

    if (
        not valid_dates.empty
        and len(missing) > 0
    ):
        problems.append(
            f"{len(missing):,} "
            "potentiella vardagsluckor"
        )

    if total_different > 0:
        problems.append(
            f"{total_different:,} "
            "överlappande datum med olika data"
        )

    if (
        invalid_close
        + nonfinite_close
        > 0
    ):
        problems.append(
            f"{invalid_close + nonfinite_close:,} "
            "ogiltiga close-värden"
        )

    if problems:
        print(
            "PROBLEM HITTADES:"
        )

        for problem in problems:
            print(
                f"- {problem}"
            )

    else:
        print(
            "Inga problem hittades."
        )


def main() -> None:
    print("=" * 70)
    print(
        "SINCH / YAHOO FELSÖKNING"
    )
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
    print(
        "INSTRUMENT"
    )

    for key, value in (
        instrument.items()
    ):
        print(
            f"{key}: {value}"
        )

    print()
    print(
        "1. LOKAL PRISHISTORIK"
    )

    bounds = _existing_price_bounds(
        PRICE_DIR
    )

    instrument_key = (
        f"isin:{instrument['isin']}"
    )

    instrument_bounds = bounds.get(
        instrument_key
    )

    if instrument_bounds is None:
        print(
            "Ingen lokal historik hittades "
            "för instrumentets ISIN."
        )

    else:
        print(
            f"first: "
            f"{instrument_bounds['first'].isoformat()}"
        )

        print(
            f"last : "
            f"{instrument_bounds['last'].isoformat()}"
        )

    print()
    print(
        "2. TEST AV INSTRUMENTTÄCKNING"
    )

    test_instrument_coverage_logic()

    print()
    print(
        "3. FETCH-INTERVALL"
    )

    intervals = _build_fetch_intervals(
        instrument=instrument,
        requested_start=date.fromisoformat(
            START
        ),
        effective_end=date.fromisoformat(
            END
        ),
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
                f"{fetch_start} -> "
                f"{fetch_end}"
            )

    print()
    print(
        "4. DETALJERAD KONTROLL AV LOKAL PRISDATA"
    )

    frame = _load_local_price_data()

    inspect_local_price_data(
        frame
    )

    print()
    print(
        "5. YAHOO-HÄMTNING VIA FETCH-INTERVALL"
    )

    if not intervals:
        print(
            "Hoppar över eftersom "
            "produktionslogiken inte "
            "skapade något intervall."
        )

    else:
        fetched_records = []

        for (
            fetch_start,
            fetch_end,
            reason,
        ) in intervals:
            print()
            print(
                f"Testar {reason}: "
                f"{fetch_start} -> "
                f"{fetch_end}"
            )

            records = _download_batch(
                [instrument],
                fetch_start,
                fetch_end,
            )

            fetched_records.extend(
                records
            )

        print_records(
            "HÄMTADE RECORDS",
            fetched_records[
                :20
            ],
        )

        print(
            "Totalt hämtade records: "
            f"{len(fetched_records):,}"
        )

    print()
    print(
        "6. DIREKT YAHOO-TEST"
    )

    yahoo_records = _download_batch(
        [instrument],
        date.fromisoformat(
            YAHOO_TEST_START
        ),
        date.fromisoformat(
            YAHOO_TEST_END
        ),
    )

    print_records(
        "FÖRSTA 20 YAHOO-RECORDS",
        yahoo_records[
            :20
        ],
    )

    print(
        "Totalt antal Yahoo-records: "
        f"{len(yahoo_records):,}"
    )

    if yahoo_records:
        dates = [
            record["date"]
            for record in yahoo_records
        ]

        print(
            "Yahoo första datum: "
            f"{min(dates)}"
        )

        print(
            "Yahoo sista datum: "
            f"{max(dates)}"
        )

        print()
        print(
            "SENASTE YAHOO-RECORD"
        )

        latest = max(
            yahoo_records,
            key=lambda record: record[
                "date"
            ],
        )

        for key, value in latest.items():
            print(
                f"{key}: {value}"
            )

    print()
    print("=" * 70)
    print(
        "SLUTSATS"
    )
    print("=" * 70)

    if yahoo_records:
        print(
            "Yahoo/yfinance fungerar för "
            f"{SYMBOL}."
        )

        print(
            "Close-data kan hämtas och "
            "parsas korrekt."
        )

    else:
        print(
            "Yahoo-testet returnerade "
            "ingen användbar data."
        )


if __name__ == "__main__":
    main()
