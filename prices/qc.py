from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any


PRICE_DIR = Path(
    "data/raw/prices"
)

MAPPING_PATH = Path(
    "data/analysis/instrument_map.json"
)

QC_PATH = (
    PRICE_DIR
    / "price_qc.json"
)


def _load_mapping() -> dict[str, dict[str, Any]]:
    if not MAPPING_PATH.exists():
        return {}

    with MAPPING_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(
            handle
        )

    if isinstance(data, dict):
        return data

    return {}


def _find_price_file() -> Path:
    files = sorted(
        PRICE_DIR.glob(
            "prices_*.jsonl"
        )
    )

    if not files:
        raise FileNotFoundError(
            "Ingen prices_*.jsonl hittades "
            f"i {PRICE_DIR}"
        )

    return files[-1]


def _parse_date(
    value: Any,
) -> date | None:
    if not isinstance(
        value,
        str,
    ):
        return None

    try:
        return date.fromisoformat(
            value
        )
    except ValueError:
        return None


def _load_records(
    path: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, int],
]:
    records: list[
        dict[str, Any]
    ] = []

    invalid = {
        "missing_symbol": 0,
        "missing_date": 0,
        "bad_date": 0,
        "missing_close": 0,
        "nonfinite_close": 0,
    }

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(
                    line
                )
            except json.JSONDecodeError:
                invalid[
                    "missing_close"
                ] += 1
                continue

            symbol = record.get(
                "yahoo_symbol"
            )

            if (
                symbol is None
                or not str(symbol).strip()
            ):
                invalid[
                    "missing_symbol"
                ] += 1
                continue

            raw_date = record.get(
                "date"
            )

            if raw_date is None:
                invalid[
                    "missing_date"
                ] += 1
                continue

            parsed_date = _parse_date(
                raw_date
            )

            if parsed_date is None:
                invalid[
                    "bad_date"
                ] += 1
                continue

            close = record.get(
                "close"
            )

            if close is None:
                invalid[
                    "missing_close"
                ] += 1
                continue

            try:
                numeric_close = float(
                    close
                )
            except (
                TypeError,
                ValueError,
            ):
                invalid[
                    "nonfinite_close"
                ] += 1
                continue

            if not math.isfinite(
                numeric_close
            ):
                invalid[
                    "nonfinite_close"
                ] += 1
                continue

            clean_record = dict(
                record
            )

            clean_record[
                "date"
            ] = parsed_date.isoformat()

            clean_record[
                "close"
            ] = numeric_close

            records.append(
                clean_record
            )

    return records, invalid


def _check_duplicates(
    records: list[dict[str, Any]],
) -> int:
    seen: set[
        tuple[str, str]
    ] = set()

    duplicates = 0

    for record in records:
        key = (
            str(
                record[
                    "yahoo_symbol"
                ]
            ),
            str(
                record["date"]
            ),
        )

        if key in seen:
            duplicates += 1
        else:
            seen.add(key)

    return duplicates


def _observation_counts(
    records: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[
        str,
        int,
    ] = defaultdict(int)

    for record in records:
        counts[
            str(
                record[
                    "yahoo_symbol"
                ]
            )
        ] += 1

    return dict(counts)


def _large_moves(
    records: list[dict[str, Any]],
    threshold: float = 0.50,
) -> int:
    grouped: dict[
        str,
        list[tuple[date, float]],
    ] = defaultdict(list)

    for record in records:
        grouped[
            str(
                record[
                    "yahoo_symbol"
                ]
            )
        ].append(
            (
                date.fromisoformat(
                    record["date"]
                ),
                float(
                    record["close"]
                ),
            )
        )

    count = 0

    for values in grouped.values():
        values.sort(
            key=lambda item: item[0]
        )

        previous_close: float | None = None

        for _, close in values:
            if (
                previous_close is not None
                and previous_close != 0
            ):
                change = (
                    close
                    / previous_close
                    - 1.0
                )

                if abs(change) >= threshold:
                    count += 1

            previous_close = close

    return count


def _large_gaps(
    records: list[dict[str, Any]],
    threshold_days: int = 7,
) -> int:
    grouped: dict[
        str,
        list[date],
    ] = defaultdict(list)

    for record in records:
        grouped[
            str(
                record[
                    "yahoo_symbol"
                ]
            )
        ].append(
            date.fromisoformat(
                record["date"]
            )
        )

    count = 0

    for dates in grouped.values():
        dates.sort()

        previous: date | None = None

        for current in dates:
            if previous is not None:
                gap = (
                    current
                    - previous
                ).days

                if gap > threshold_days:
                    count += 1

            previous = current

    return count


def _mapping_coverage(
    records: list[dict[str, Any]],
    mapping: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    price_symbols = {
        str(
            record[
                "yahoo_symbol"
            ]
        )
        for record in records
    }

    mapping_symbols = set()

    unresolved = 0

    for entry in mapping.values():
        symbol = entry.get(
            "yahoo_symbol"
        )

        if symbol:
            mapping_symbols.add(
                str(symbol)
            )
        else:
            unresolved += 1

    mapped_without_prices = sorted(
        mapping_symbols
        - price_symbols
    )

    prices_without_mapping = sorted(
        price_symbols
        - mapping_symbols
    )

    return {
        "mapping_entries": len(
            mapping
        ),
        "mapped_symbols": len(
            mapping_symbols
        ),
        "unresolved": unresolved,
        "price_symbols": len(
            price_symbols
        ),
        "mapped_without_prices": (
            mapped_without_prices
        ),
        "prices_without_mapping": (
            prices_without_mapping
        ),
    }


def _group_dates_by_symbol(
    records: list[dict[str, Any]],
) -> dict[str, list[date]]:
    grouped: dict[
        str,
        list[date],
    ] = defaultdict(list)

    for record in records:
        symbol = str(
            record[
                "yahoo_symbol"
            ]
        )

        grouped[symbol].append(
            date.fromisoformat(
                record["date"]
            )
        )

    for dates in grouped.values():
        dates.sort()

    return grouped


def _expected_weekdays(
    start: date,
    end: date,
) -> int:
    if end < start:
        return 0

    count = 0
    current = start

    while current <= end:
        if current.weekday() < 5:
            count += 1

        current += timedelta(
            days=1
        )

    return count


def _coverage_by_symbol(
    records: list[dict[str, Any]],
    mapping: dict[str, dict[str, Any]],
    dataset_start: date | None,
    dataset_end: date | None,
) -> dict[str, Any]:
    """
    Analysera faktisk prisdatatäckning per Yahoo-symbol.

    Klassificering:
      - no_data
      - active
      - starts_late
      - ends_early
      - starts_late_and_ends_early
      - internal_gaps

    "Late" och "early" är endast diagnostiska signaler.
    De betyder inte automatiskt att datan är felaktig.

    Ett instrument kan exempelvis ha noterats efter
    datasetets start eller ha avnoterats före datasetets slut.
    """

    dates_by_symbol = _group_dates_by_symbol(
        records
    )

    mapping_symbols = {
        str(
            entry.get(
                "yahoo_symbol"
            )
        ).strip()
        for entry in mapping.values()
        if entry.get(
            "yahoo_symbol"
        )
        and str(
            entry.get(
                "yahoo_symbol"
            )
        ).strip()
    }

    all_symbols = (
        mapping_symbols
        | set(dates_by_symbol)
    )

    result: dict[
        str,
        dict[str, Any],
    ] = {}

    for symbol in sorted(
        all_symbols
    ):
        dates = dates_by_symbol.get(
            symbol,
            [],
        )

        if not dates:
            result[symbol] = {
                "first_date": None,
                "last_date": None,
                "observations": 0,
                "expected_weekdays": 0,
                "missing_weekdays": 0,
                "largest_gap_days": 0,
                "internal_gap_count": 0,
                "starts_late": False,
                "ends_early": False,
                "status": "no_data",
            }

            continue

        first = dates[0]
        last = dates[-1]

        expected = _expected_weekdays(
            first,
            last,
        )

        actual_dates = set(
            dates
        )

        missing_weekdays = 0

        current = first

        while current <= last:
            if (
                current.weekday() < 5
                and current
                not in actual_dates
            ):
                missing_weekdays += 1

            current += timedelta(
                days=1
            )

        largest_gap_days = 0
        internal_gap_count = 0

        previous = None

        for current in dates:
            if previous is not None:
                gap_days = (
                    current
                    - previous
                ).days

                if gap_days > largest_gap_days:
                    largest_gap_days = (
                        gap_days
                    )

                if gap_days > 7:
                    internal_gap_count += 1

            previous = current

        starts_late = (
            dataset_start is not None
            and first > dataset_start
        )

        ends_early = (
            dataset_end is not None
            and last < dataset_end
        )

        if (
            internal_gap_count > 0
        ):
            status = "internal_gaps"

        elif (
            starts_late
            and ends_early
        ):
            status = (
                "starts_late_and_ends_early"
            )

        elif starts_late:
            status = "starts_late"

        elif ends_early:
            status = "ends_early"

        else:
            status = "active"

        result[symbol] = {
            "first_date": first.isoformat(),
            "last_date": last.isoformat(),
            "observations": len(dates),
            "expected_weekdays": expected,
            "missing_weekdays": missing_weekdays,
            "largest_gap_days": largest_gap_days,
            "internal_gap_count": internal_gap_count,
            "starts_late": starts_late,
            "ends_early": ends_early,
            "status": status,
        }

    return result


def _coverage_summary(
    coverage_by_symbol: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    summary = {
        "total_symbols": len(
            coverage_by_symbol
        ),
        "no_data": 0,
        "active": 0,
        "starts_late": 0,
        "ends_early": 0,
        "starts_late_and_ends_early": 0,
        "internal_gaps": 0,
        "with_missing_weekdays": 0,
        "with_large_gaps": 0,
    }

    for item in coverage_by_symbol.values():
        status = item["status"]

        if status in summary:
            summary[status] += 1

        if item[
            "missing_weekdays"
        ] > 0:
            summary[
                "with_missing_weekdays"
            ] += 1

        if item[
            "largest_gap_days"
        ] > 7:
            summary[
                "with_large_gaps"
            ] += 1

    return summary


def main() -> None:
    print(
        "Price QC: startar."
    )

    mapping = _load_mapping()

    print(
        "Price QC: mapping entries = "
        f"{len(mapping)}"
    )

    price_file = _find_price_file()

    print(
        "Price QC: prisfil = "
        f"{price_file.name}"
    )

    records, invalid = (
        _load_records(
            price_file
        )
    )

    duplicates = _check_duplicates(
        records
    )

    counts = _observation_counts(
        records
    )

    equal_1080 = sum(
        1
        for count in counts.values()
        if count == 1080
    )

    below_1000 = sum(
        1
        for count in counts.values()
        if count < 1000
    )

    large_moves = _large_moves(
        records
    )

    large_gaps = _large_gaps(
        records
    )

    coverage = _mapping_coverage(
        records,
        mapping,
    )

    symbols = {
        str(
            record[
                "yahoo_symbol"
            ]
        )
        for record in records
    }

    isins = {
        str(
            record["isin"]
        ).strip()
        for record in records
        if record.get("isin")
    }

    dates = [
        date.fromisoformat(
            record["date"]
        )
        for record in records
    ]

    first_date = (
        min(dates)
        if dates
        else None
    )

    last_date = (
        max(dates)
        if dates
        else None
    )

    coverage_by_symbol = (
        _coverage_by_symbol(
            records,
            mapping,
            first_date,
            last_date,
        )
    )

    coverage_summary = (
        _coverage_summary(
            coverage_by_symbol
        )
    )

    total_invalid = sum(
        invalid.values()
    )

    integrity_pass = (
        duplicates == 0
    )

    print()
    print("Price QC")
    print(
        f"Rows: "
        f"{len(records) + total_invalid:,}"
        .replace(",", " ")
        + " | Valid: "
        f"{len(records):,}"
        .replace(",", " ")
        + " | Invalid: "
        f"{total_invalid:,}"
        .replace(",", " ")
    )

    print(
        f"Symbols: {len(symbols)} "
        f"| ISIN: {len(isins)}"
    )

    print(
        "Period: "
        f"{first_date} -> {last_date}"
    )

    print(
        "Mapping: "
        f"{coverage['mapped_symbols']} mapped "
        f"| {coverage['unresolved']} unresolved"
    )

    print(
        "Coverage: "
        f"{len(coverage['mapped_without_prices'])} "
        "mapped without prices "
        f"| {len(coverage['prices_without_mapping'])} "
        "prices without mapping"
    )

    print()
    print("Prisdatatäckning")

    print(
        "  symboler totalt: "
        f"{coverage_summary['total_symbols']}"
    )

    print(
        "  ingen prisdata: "
        f"{coverage_summary['no_data']}"
    )

    print(
        "  aktiva genom hela perioden: "
        f"{coverage_summary['active']}"
    )

    print(
        "  börjar sent: "
        f"{coverage_summary['starts_late']}"
    )

    print(
        "  slutar tidigt: "
        f"{coverage_summary['ends_early']}"
    )

    print(
        "  börjar sent + slutar tidigt: "
        f"{coverage_summary['starts_late_and_ends_early']}"
    )

    print(
        "  interna luckor >7 dagar: "
        f"{coverage_summary['internal_gaps']}"
    )

    print(
        "  med saknade vardagar: "
        f"{coverage_summary['with_missing_weekdays']}"
    )

    print(
        "  med lucka >7 dagar: "
        f"{coverage_summary['with_large_gaps']}"
    )

    no_data_symbols = sorted(
        symbol
        for symbol, item
        in coverage_by_symbol.items()
        if item["status"] == "no_data"
    )

    early_end_symbols = sorted(
        symbol
        for symbol, item
        in coverage_by_symbol.items()
        if item["status"]
        in {
            "ends_early",
            "starts_late_and_ends_early",
        }
    )

    internal_gap_symbols = sorted(
        symbol
        for symbol, item
        in coverage_by_symbol.items()
        if item["status"]
        == "internal_gaps"
    )

    print(
        "  utan prisdata: "
        + (
            ", ".join(
                no_data_symbols
            )
            if no_data_symbols
            else "-"
        )
    )

    print(
        "  historik slutar tidigt: "
        + (
            ", ".join(
                early_end_symbols
            )
            if early_end_symbols
            else "-"
        )
    )

    print(
        "  interna luckor: "
        + (
            ", ".join(
                internal_gap_symbols
            )
            if internal_gap_symbols
            else "-"
        )
    )

    print(
        "Invalid breakdown: "
        f"missing symbol={invalid['missing_symbol']} "
        f"| missing date={invalid['missing_date']} "
        f"| bad date={invalid['bad_date']} "
        f"| missing close={invalid['missing_close']} "
        f"| nonfinite close={invalid['nonfinite_close']}"
    )

    print(
        "Integrity: "
        f"{'PASS' if integrity_pass else 'FAIL'} "
        f"| Duplicates: {duplicates}"
    )

    print(
        "Observations: "
        f"1080 = {equal_1080} "
        f"| <1000 = {below_1000}"
    )

    print(
        "Large moves >=50%: "
        f"{large_moves} "
        f"| Gaps >7d: {large_gaps}"
    )

    report = {
        "price_file": price_file.name,
        "rows_total": (
            len(records)
            + total_invalid
        ),
        "rows_valid": len(records),
        "rows_invalid": total_invalid,
        "symbols": len(symbols),
        "isins": len(isins),
        "period": {
            "start": (
                first_date.isoformat()
                if first_date
                else None
            ),
            "end": (
                last_date.isoformat()
                if last_date
                else None
            ),
        },
        "mapping": coverage,
        "invalid": invalid,
        "integrity": {
            "pass": integrity_pass,
            "duplicates": duplicates,
        },
        "observations": {
            "exactly_1080": equal_1080,
            "below_1000": below_1000,
        },
        "large_moves_ge_50pct": (
            large_moves
        ),
        "gaps_gt_7d": large_gaps,
        "coverage_summary": coverage_summary,
        "coverage_by_symbol": (
            coverage_by_symbol
        ),
    }

    QC_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with QC_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "QC report: "
        f"{QC_PATH.resolve()}"
    )

    if not integrity_pass:
        raise RuntimeError(
            "Price QC failed: "
            f"{duplicates} duplicate "
            "symbol/date combinations."
        )


if __name__ == "__main__":
    main()
