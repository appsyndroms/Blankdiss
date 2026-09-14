from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date
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
        data = json.load(handle)

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
