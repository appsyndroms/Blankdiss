from __future__ import annotations

import math
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

# Separat testintervall för att tvinga fram ett riktigt Yahoo-anrop.
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


def _load_local_price_data() -> tuple[
    pd.DataFrame,
    list[Path],
]:
    """
    Läser endast prisrader för SYMBOL.
    Funktionen skriver aldrig till disk.
    """

    files = sorted(
        PRICE_DIR.glob("prices_*.jsonl")
    )

    frames: list[pd.DataFrame] = []
    matching_files: list[Path] = []

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

        matching["_source_file"] = path.name

        frames.append(matching)
        matching_files.append(path)

    if not frames:
        return (
            pd.DataFrame(),
            matching_files,
        )

    frame = pd.concat(
        frames,
        ignore_index=True,
    )

    if "date" in frame.columns:
        frame["_date"] = pd.to_datetime(
            frame["date"],
            errors="coerce",
        ).dt.date

    if "close" in frame.columns:
        frame["_close_numeric"] = pd.to_numeric(
            frame["close"],
            errors="coerce",
        )

    return (
        frame,
        matching_files,
    )


def _print_value_distribution(
    frame: pd.DataFrame,
    column: str,
) -> None:
    if column not in frame.columns:
        print(
            f"{column}: KOLUMN SAKNAS"
        )
        return

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
        non_empty.nunique() <= 10
        and not non_empty.empty
    ):
        print(
            f"  värden: "
            f"{sorted(non_empty.unique())}"
        )


def _normalise_comparison_value(
    value,
):
    """
    Normaliserar ett värde inför jämförelse mellan prisfiler.

    Datum och numeriska close-värden normaliseras så att exempelvis
    pandas Timestamp och Python date inte skapar falska skillnader.
    """

    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

        return value

    return str(value).strip()


def _compare_overlapping_files(
    frame: pd.DataFrame,
    file_a: str,
    file_b: str,
    overlap_start: date,
    overlap_end: date,
) -> tuple[
    int,
    int,
    dict[str, int],
    list[tuple[date, dict[str, tuple[object, object]]]],
]:
    """
    Jämför hela prisrader mellan två överlappande filer.

    Jämförelsen görs på datum.

    Returnerar:

        antal gemensamma datum
        antal identiska datum
        antal skillnader per kolumn
        exempel på datum med skillnader
    """

    columns = [
        column
        for column in frame.columns
        if column not in {
            "_source_file",
            "_date",
            "_close_numeric",
        }
    ]

    data = frame.loc[
        frame["_source_file"].isin(
            [
                file_a,
                file_b,
            ]
        )
        & frame["_date"].between(
            overlap_start,
            overlap_end,
        )
    ].copy()

    first = data.loc[
        data["_source_file"] == file_a
    ].copy()

    second = data.loc[
        data["_source_file"] == file_b
    ].copy()

    first = (
        first
        .drop_duplicates(
            subset=["_date"],
            keep="first",
        )
        .set_index("_date")
    )

    second = (
        second
        .drop_duplicates(
            subset=["_date"],
            keep="first",
        )
        .set_index("_date")
    )

    common_dates = sorted(
        set(first.index)
        & set(second.index)
    )

    identical_dates = 0
    differences_per_column: dict[
        str,
        int,
    ] = {}

    examples: list[
        tuple[
            date,
            dict[
                str,
                tuple[object, object],
            ],
        ]
    ] = []

    for current_date in common_dates:
        row_a = first.loc[
            current_date
        ]

        row_b = second.loc[
            current_date
        ]

        differences: dict[
            str,
            tuple[object, object],
        ] = {}

        for column in columns:
            value_a = (
                _normalise_comparison_value(
                    row_a.get(column)
                )
            )

            value_b = (
                _normalise_comparison_value(
                    row_b.get(column)
                )
            )

            if value_a != value_b:
                differences[
                    column
                ] = (
                    value_a,
                    value_b,
                )

                differences_per_column[
                    column
                ] = (
                    differences_per_column.get(
                        column,
                        0,
                    )
                    + 1
                )

        if not differences:
            identical_dates += 1

        elif len(examples) < 20:
            examples.append(
                (
                    current_date,
                    differences,
                )
            )

    return (
        len(common_dates),
        identical_dates,
        differences_per_column,
        examples,
    )


def inspect_local_price_data(
    frame: pd.DataFrame,
) -> None:
    print()
    print("3. DETALJERAD KONTROLL AV LOKAL PRISDATA")
    print("-" * 70)

    if frame.empty:
        print(
            f"INGA LOKALA RADER FÖR {SYMBOL}."
        )
        return

    matching_files = (
        frame["_source_file"]
        .nunique()
    )

    print(
        f"Filer som innehåller {SYMBOL}: "
        f"{matching_files}"
    )

    print(
        f"Lokala rader totalt: "
        f"{len(frame):,}"
    )

    # ---------------------------------------------------------
    # Datum
    # ---------------------------------------------------------

    if "_date" not in frame.columns:
        print(
            "FEL: kolumnen 'date' saknas."
        )
        return

    invalid_dates = int(
        frame["_date"].isna().sum()
    )

    print(
        f"Ogiltiga datum: "
        f"{invalid_dates:,}"
    )

    valid_dates = frame.loc[
        frame["_date"].notna()
    ].copy()

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

    unique_dates = set(
        valid_dates["_date"]
    )

    duplicate_rows = (
        len(valid_dates)
        - len(unique_dates)
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

    print()
    print(
        f"Unika datum: "
        f"{len(unique_dates):,}"
    )

    print(
        f"Extra rader utöver unika datum: "
        f"{duplicate_rows:,}"
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
    # ISIN
    # ---------------------------------------------------------

    print()
    print("KONTROLL AV ISIN-HISTORIK")
    print("-" * 70)

    _print_value_distribution(
        frame,
        "isin",
    )

    if "isin" in frame.columns:
        isin_frame = frame.copy()

        isin_frame["_isin"] = (
            isin_frame["isin"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        isin_frame = isin_frame.loc[
            (isin_frame["_isin"] != "")
            & isin_frame["_date"].notna()
        ]

        if not isin_frame.empty:
            print()
            print(
                "Datumintervall per ISIN:"
            )

            isin_ranges = (
                isin_frame
                .groupby("_isin")
                .agg(
                    rows=("_date", "size"),
                    unique_dates=(
                        "_date",
                        "nunique",
                    ),
                    first=("_date", "min"),
                    last=("_date", "max"),
                )
                .sort_index()
            )

            for isin, row in isin_ranges.iterrows():
                print(
                    f"  {isin}: "
                    f"{int(row['rows']):,} rader, "
                    f"{int(row['unique_dates']):,} unika datum, "
                    f"{row['first']} -> "
                    f"{row['last']}"
                )

    # ---------------------------------------------------------
    # Datum + ISIN
    # ---------------------------------------------------------

    print()
    print("KONTROLL AV DATUM + ISIN")
    print("-" * 70)

    if "isin" in frame.columns:
        identity_frame = frame.loc[
            frame["_date"].notna()
        ].copy()

        identity_frame["_isin"] = (
            identity_frame["isin"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        date_isin_counts = (
            identity_frame
            .groupby(
                ["_date", "_isin"]
            )
            .size()
            .reset_index(
                name="rows"
            )
        )

        duplicate_date_isin = (
            date_isin_counts.loc[
                date_isin_counts["rows"] > 1
            ]
        )

        print(
            "Dubbletter på "
            "datum + ISIN: "
            f"{len(duplicate_date_isin):,}"
        )

        if not duplicate_date_isin.empty:
            print(
                "Första 20:"
            )

            for _, row in (
                duplicate_date_isin
                .sort_values(
                    ["_date", "_isin"]
                )
                .head(20)
                .iterrows()
            ):
                print(
                    f"  {row['_date'].isoformat()} "
                    f"{row['_isin']}: "
                    f"{row['rows']} rader"
                )

    # ---------------------------------------------------------
    # Samma datum, flera ISIN
    # ---------------------------------------------------------

    print()
    print("KONTROLL AV SAMMA DATUM MED FLERA ISIN")
    print("-" * 70)

    if "isin" in frame.columns:
        date_isin = (
            frame.loc[
                frame["_date"].notna()
            ]
            .assign(
                _isin=lambda data: (
                    data["isin"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )
            )
            .groupby("_date")["_isin"]
            .nunique()
        )

        dates_with_multiple_isins = (
            date_isin[
                date_isin > 1
            ]
        )

        print(
            "Datum med flera ISIN: "
            f"{len(dates_with_multiple_isins):,}"
        )

        if not dates_with_multiple_isins.empty:
            print()
            print(
                "Första 20 datum med flera ISIN:"
            )

            for duplicate_date in (
                dates_with_multiple_isins
                .sort_index()
                .head(20)
                .index
            ):
                rows = frame.loc[
                    frame["_date"]
                    == duplicate_date
                ].copy()

                rows["_isin"] = (
                    rows["isin"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

                print()
                print(
                    f"  {duplicate_date.isoformat()}"
                )

                for row in (
                    rows[
                        [
                            "_isin",
                            "close",
                            "_source_file",
                        ]
                    ]
                    .drop_duplicates()
                    .sort_values(
                        [
                            "_isin",
                            "_source_file",
                        ]
                    )
                    .to_dict("records")
                ):
                    print(
                        f"    ISIN={row['_isin']} "
                        f"close={row['close']} "
                        f"file={row['_source_file']}"
                    )

    # ---------------------------------------------------------
    # Dubbletter: skiljer close sig?
    # ---------------------------------------------------------

    print()
    print("KONTROLL AV DUBLETTER OCH CLOSE")
    print("-" * 70)

    if (
        "_close_numeric" in frame.columns
        and "_date" in frame.columns
    ):
        valid_close_frame = frame.loc[
            frame["_date"].notna()
            & frame["_close_numeric"].notna()
        ].copy()

        close_per_date = (
            valid_close_frame
            .groupby("_date")[
                "_close_numeric"
            ]
            .nunique()
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

        if not conflicting_close_dates.empty:
            print()
            print(
                "Första 20 konflikter:"
            )

            for duplicate_date in (
                conflicting_close_dates
                .sort_index()
                .head(20)
                .index
            ):
                rows = (
                    valid_close_frame.loc[
                        valid_close_frame["_date"]
                        == duplicate_date
                    ]
                    .sort_values(
                        [
                            "_close_numeric",
                            "_source_file",
                        ]
                    )
                )

                print()
                print(
                    f"  {duplicate_date.isoformat()}"
                )

                for row in (
                    rows[
                        [
                            "isin",
                            "_close_numeric",
                            "_source_file",
                        ]
                    ]
                    .drop_duplicates()
                    .to_dict("records")
                ):
                    print(
                        f"    ISIN={row['isin']} "
                        f"close={row['_close_numeric']} "
                        f"file={row['_source_file']}"
                    )

        else:
            print(
                "Alla dubbletter har samma "
                "close-värde per datum."
            )

    # ---------------------------------------------------------
    # Filintervall och överlapp
    # ---------------------------------------------------------

    print()
    print("KONTROLL AV ÖVERLAPPANDE PRISFILER")
    print("-" * 70)

    file_ranges: dict[
        str,
        tuple[date, date, int],
    ] = {}

    for filename, group in (
        frame.loc[
            frame["_date"].notna()
        ]
        .groupby("_source_file")
    ):
        file_ranges[filename] = (
            group["_date"].min(),
            group["_date"].max(),
            len(group),
        )

    for filename, (
        file_first,
        file_last,
        rows,
    ) in sorted(
        file_ranges.items()
    ):
        print(
            f"{filename}: "
            f"{rows:,} rader, "
            f"{file_first} -> "
            f"{file_last}"
        )

    overlaps = []

    for (
        file_a,
        file_b,
    ) in combinations(
        sorted(file_ranges),
        2,
    ):
        first_a, last_a, rows_a = (
            file_ranges[file_a]
        )

        first_b, last_b, rows_b = (
            file_ranges[file_b]
        )

        overlap_start = max(
            first_a,
            first_b,
        )

        overlap_end = min(
            last_a,
            last_b,
        )

        if overlap_start <= overlap_end:
            overlap_rows = frame.loc[
                frame["_source_file"].isin(
                    [
                        file_a,
                        file_b,
                    ]
                )
                & frame["_date"].between(
                    overlap_start,
                    overlap_end,
                )
            ]

            overlap_dates = (
                overlap_rows[
                    "_date"
                ]
                .nunique()
            )

            overlaps.append(
                (
                    file_a,
                    file_b,
                    overlap_start,
                    overlap_end,
                    overlap_dates,
                )
            )

    print()
    print(
        f"Överlappande filpar: "
        f"{len(overlaps):,}"
    )

    for (
        file_a,
        file_b,
        overlap_start,
        overlap_end,
        overlap_dates,
    ) in overlaps:
        print(
            f"  {file_a}"
        )

        print(
            f"    <-> {file_b}"
        )

        print(
            f"    {overlap_start} -> "
            f"{overlap_end} "
            f"({overlap_dates:,} gemensamma datum)"
        )

    # ---------------------------------------------------------
    # Jämförelse av hela rader i överlapp
    # ---------------------------------------------------------

    print()
    print("KONTROLL AV HELA RADER I FILÖVERLAPP")
    print("-" * 70)

    total_common_dates = 0
    total_identical_dates = 0
    total_different_dates = 0

    all_difference_columns: dict[
        str,
        int,
    ] = {}

    for (
        file_a,
        file_b,
        overlap_start,
        overlap_end,
        overlap_dates,
    ) in overlaps:
        (
            common_dates,
            identical_dates,
            differences_per_column,
            examples,
        ) = _compare_overlapping_files(
            frame=frame,
            file_a=file_a,
            file_b=file_b,
            overlap_start=overlap_start,
            overlap_end=overlap_end,
        )

        different_dates = (
            common_dates
            - identical_dates
        )

        total_common_dates += (
            common_dates
        )

        total_identical_dates += (
            identical_dates
        )

        total_different_dates += (
            different_dates
        )

        for column, count in (
            differences_per_column.items()
        ):
            all_difference_columns[
                column
            ] = (
                all_difference_columns.get(
                    column,
                    0,
                )
                + count
            )

        print()
        print(
            f"{file_a}"
        )
        print(
            f"  <-> {file_b}"
        )
        print(
            f"  Gemensamma datum: "
            f"{common_dates:,}"
        )
        print(
            f"  Helt identiska rader: "
            f"{identical_dates:,}"
        )
        print(
            f"  Datum med skillnader: "
            f"{different_dates:,}"
        )

        if differences_per_column:
            print(
                "  Skillnader per kolumn:"
            )

            for column, count in sorted(
                differences_per_column.items()
            ):
                print(
                    f"    {column}: "
                    f"{count:,}"
                )

        if examples:
            print(
                "  Första exempel på skillnader:"
            )

            for (
                example_date,
                differences,
            ) in examples:
                print()
                print(
                    f"    {example_date.isoformat()}"
                )

                for column, (
                    value_a,
                    value_b,
                ) in differences.items():
                    print(
                        f"      {column}: "
                        f"{value_a!r} "
                        f"<-> "
                        f"{value_b!r}"
                    )

        else:
            print(
                "  Alla jämförda rader är "
                "identiska."
            )

    print()
    print("TOTAL JÄMFÖRELSE")
    print("-" * 70)
    print(
        f"Gemensamma datum: "
        f"{total_common_dates:,}"
    )
    print(
        f"Helt identiska rader: "
        f"{total_identical_dates:,}"
    )
    print(
        f"Datum med skillnader: "
        f"{total_different_dates:,}"
    )

    if all_difference_columns:
        print()
        print(
            "Totala skillnader per kolumn:"
        )

        for column, count in sorted(
            all_difference_columns.items()
        ):
            print(
                f"  {column}: "
                f"{count:,}"
            )
    else:
        print()
        print(
            "Alla överlappande prisrader är "
            "identiska."
        )

    # ---------------------------------------------------------
    # Virtuell deduplicering
    # ---------------------------------------------------------

    print()
    print("VIRTUELL DEDUPLICERING")
    print("-" * 70)

    print(
        "Ingen fil ändras."
    )

    print(
        "Här räknas endast hur datan skulle se ut "
        "om vi behåller en rad per datum + Yahoo-symbol."
    )

    dedupe_frame = frame.loc[
        frame["_date"].notna()
    ].copy()

    before_rows = len(
        dedupe_frame
    )

    after_rows = (
        dedupe_frame
        .drop_duplicates(
            subset=[
                "_date",
                "yahoo_symbol",
            ],
            keep="first",
        )
        .shape[0]
    )

    print(
        f"Rader före: "
        f"{before_rows:,}"
    )

    print(
        f"Rader efter datum + Yahoo-symbol: "
        f"{after_rows:,}"
    )

    print(
        f"Rader som skulle tas bort: "
        f"{before_rows - after_rows:,}"
    )

    # ---------------------------------------------------------
    # Virtuell deduplicering på datum + ISIN
    # ---------------------------------------------------------

    if "isin" in frame.columns:
        print()
        print(
            "VIRTUELL DEDUPLICERING PÅ "
            "DATUM + ISIN"
        )
        print("-" * 70)

        dedupe_instrument = (
            frame.loc[
                frame["_date"].notna()
            ]
            .assign(
                _isin=lambda data: (
                    data["isin"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )
            )
        )

        before_instrument = len(
            dedupe_instrument
        )

        after_instrument = (
            dedupe_instrument
            .drop_duplicates(
                subset=[
                    "_date",
                    "_isin",
                ],
                keep="first",
            )
            .shape[0]
        )

        print(
            f"Rader före: "
            f"{before_instrument:,}"
        )

        print(
            f"Rader efter datum + ISIN: "
            f"{after_instrument:,}"
        )

        print(
            f"Rader som skulle tas bort: "
            f"{before_instrument - after_instrument:,}"
        )

    # ---------------------------------------------------------
    # Handelsdagar / luckor
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

    if "_close_numeric" not in frame.columns:
        print(
            "FEL: kolumnen 'close' saknas."
        )

    else:
        finite_mask = (
            frame["_close_numeric"]
            .apply(
                lambda value: (
                    False
                    if pd.isna(value)
                    else math.isfinite(
                        float(value)
                    )
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

        valid_close = frame.loc[
            finite_mask,
            "_close_numeric",
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
    # Identitet
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
        _print_value_distribution(
            frame,
            column,
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

    if "isin" in frame.columns:
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

        if len(unique_isins) > 1:
            problems.append(
                f"{len(unique_isins)} olika ISIN "
                f"förekommer i prisdatan"
            )

    if missing_dates:
        problems.append(
            f"{len(missing_dates)} potentiella "
            f"vardagsluckor"
        )

    if (
        "_close_numeric" in frame.columns
    ):
        finite_mask = (
            frame["_close_numeric"]
            .apply(
                lambda value: (
                    False
                    if pd.isna(value)
                    else math.isfinite(
                        float(value)
                    )
                )
            )
        )

        invalid_close_count = int(
            (~finite_mask).sum()
        )

        if invalid_close_count:
            problems.append(
                f"{invalid_close_count} "
                f"ogiltiga close"
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
        print(
            f"{key}: {value}"
        )

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
    # Läs lokal data en gång
    # ---------------------------------------------------------

    frame, _ = _load_local_price_data()

    # ---------------------------------------------------------
    # 3. Detaljerad kontroll
    # ---------------------------------------------------------

    inspect_local_price_data(
        frame
    )

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
