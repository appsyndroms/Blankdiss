from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FI_RECONSTRUCTED = (
    ROOT / "data" / "processed" / "fi" / "aggregate" / "reconstructed.jsonl"
)
MAPPING_FILE = ROOT / "data" / "analysis" / "instrument_map.json"
PRICE_DIR = ROOT / "data" / "raw" / "prices"
QC_FILE = PRICE_DIR / "price_qc.json"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Ogiltig JSON på rad {line_no} i {path}: {exc}"
                ) from exc


def load_fi_isins() -> set[str]:
    isins: set[str] = set()

    for row in read_jsonl(FI_RECONSTRUCTED):
        isin = str(row.get("isin", "")).strip()
        if isin:
            isins.add(isin)

    return isins


def load_mapping() -> dict:
    with MAPPING_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_price_file() -> Path:
    files = sorted(PRICE_DIR.glob("prices_*.jsonl"))

    if not files:
        raise FileNotFoundError(
            f"Inga prisfiler hittades i {PRICE_DIR}"
        )

    return files[-1]


def finite_positive_number(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(number) and number > 0


def main() -> None:
    print("Price QC: startar.")

    if not FI_RECONSTRUCTED.exists():
        raise FileNotFoundError(FI_RECONSTRUCTED)

    if not MAPPING_FILE.exists():
        raise FileNotFoundError(MAPPING_FILE)

    if not PRICE_DIR.exists():
        raise FileNotFoundError(PRICE_DIR)

    fi_isins = load_fi_isins()
    mapping = load_mapping()

    if not isinstance(mapping, dict):
        raise ValueError("instrument_map.json måste innehålla ett JSON-objekt.")

    latest_price_file = find_latest_price_file()

    print(f"Price QC: FI ISIN = {len(fi_isins)}")
    print(f"Price QC: mapping entries = {len(mapping)}")
    print(f"Price QC: prisfil = {latest_price_file.name}")

    mapping_symbols: dict[str, str] = {}
    unresolved_mapping = []
    duplicate_yahoo_symbols: dict[str, list[str]] = defaultdict(list)

    for isin, entry in mapping.items():
        if not isinstance(entry, dict):
            unresolved_mapping.append(isin)
            continue

        yahoo_symbol = str(entry.get("yahoo_symbol", "")).strip()

        if yahoo_symbol:
            mapping_symbols[isin] = yahoo_symbol
            duplicate_yahoo_symbols[yahoo_symbol].append(isin)
        else:
            unresolved_mapping.append(isin)

    duplicate_symbols = {
        symbol: isins
        for symbol, isins in duplicate_yahoo_symbols.items()
        if len(isins) > 1
    }

    rows = 0
    invalid_rows = 0
    duplicate_rows = 0

    price_symbols: set[str] = set()
    price_isins: set[str] = set()
    observations_by_symbol = Counter()
    observations_by_isin = Counter()

    dates_by_symbol: dict[str, list[date]] = defaultdict(list)
    previous_close: dict[str, float] = {}

    large_price_moves = []
    duplicate_keys: set[tuple[str, str]] = set()

    min_date = None
    max_date = None

    for row in read_jsonl(latest_price_file):
        rows += 1

        yahoo_symbol = str(row.get("yahoo_symbol", "")).strip()
        isin = str(row.get("isin", "")).strip()
        date_text = str(row.get("date", "")).strip()
        close = row.get("close")

        valid = (
            bool(yahoo_symbol)
            and bool(isin)
            and bool(date_text)
            and finite_positive_number(close)
        )

        if not valid:
            invalid_rows += 1
            continue

        try:
            current_date = date.fromisoformat(date_text)
        except ValueError:
            invalid_rows += 1
            continue

        key = (yahoo_symbol, date_text)

        if key in duplicate_keys:
            duplicate_rows += 1
        else:
            duplicate_keys.add(key)

        price_symbols.add(yahoo_symbol)
        price_isins.add(isin)

        observations_by_symbol[yahoo_symbol] += 1
        observations_by_isin[isin] += 1

        dates_by_symbol[yahoo_symbol].append(current_date)

        if min_date is None or current_date < min_date:
            min_date = current_date

        if max_date is None or current_date > max_date:
            max_date = current_date

        close_value = float(close)

        previous = previous_close.get(yahoo_symbol)

        if previous is not None and previous > 0:
            relative_change = abs(close_value / previous - 1.0)

            if relative_change >= 0.50:
                large_price_moves.append(
                    {
                        "yahoo_symbol": yahoo_symbol,
                        "date": date_text,
                        "previous_close": previous,
                        "close": close_value,
                        "relative_change": relative_change,
                    }
                )

        previous_close[yahoo_symbol] = close_value

    mapped_symbols = set(mapping_symbols.values())

    mapped_without_prices = sorted(mapped_symbols - price_symbols)
    prices_without_mapping = sorted(price_symbols - mapped_symbols)

    missing_price_isins = sorted(set(mapping_symbols) - price_isins)

    gaps_over_7_days = []

    for symbol, dates in dates_by_symbol.items():
        dates = sorted(set(dates))

        for previous_date, current_date in zip(dates, dates[1:]):
            gap_days = (current_date - previous_date).days

            if gap_days > 7:
                gaps_over_7_days.append(
                    {
                        "yahoo_symbol": symbol,
                        "from": previous_date.isoformat(),
                        "to": current_date.isoformat(),
                        "gap_days": gap_days,
                    }
                )

    unresolved_examples = sorted(unresolved_mapping)[:20]

    report = {
        "dataset_summary": {
            "price_file": latest_price_file.name,
            "rows": rows,
            "invalid_rows": invalid_rows,
            "duplicate_rows": duplicate_rows,
            "unique_price_symbols": len(price_symbols),
            "unique_price_isins": len(price_isins),
            "first_price_date": min_date.isoformat() if min_date else None,
            "last_price_date": max_date.isoformat() if max_date else None,
        },
        "mapping_summary": {
            "mapping_entries": len(mapping),
            "mapped_instruments": len(mapping_symbols),
            "unresolved_instruments": len(unresolved_mapping),
            "duplicate_yahoo_symbols": len(duplicate_symbols),
            "mapped_symbols_without_prices": len(mapped_without_prices),
            "price_symbols_without_mapping": len(prices_without_mapping),
            "mapped_isins_without_prices": len(missing_price_isins),
            "unresolved_examples": unresolved_examples,
            "duplicate_yahoo_symbol_examples": {
                symbol: isins
                for symbol, isins in list(duplicate_symbols.items())[:20]
            },
            "mapped_symbols_without_prices_examples": mapped_without_prices[:20],
            "price_symbols_without_mapping_examples": prices_without_mapping[:20],
        },
        "price_quality": {
            "symbols_with_1080_observations": sum(
                1 for count in observations_by_symbol.values() if count == 1080
            ),
            "symbols_with_less_than_1000_observations": sum(
                1 for count in observations_by_symbol.values() if count < 1000
            ),
            "largest_observation_count": (
                max(observations_by_symbol.values())
                if observations_by_symbol
                else 0
            ),
            "smallest_observation_count": (
                min(observations_by_symbol.values())
                if observations_by_symbol
                else 0
            ),
            "gaps_over_7_days": len(gaps_over_7_days),
            "large_price_moves_ge_50pct": len(large_price_moves),
            "large_price_move_examples": large_price_moves[:20],
            "gap_examples": gaps_over_7_days[:20],
        },
        "integrity": {
            "invalid_rows": invalid_rows,
            "duplicate_rows": duplicate_rows,
            "passed": invalid_rows == 0 and duplicate_rows == 0,
        },
    }

    PRICE_DIR.mkdir(parents=True, exist_ok=True)

    with QC_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print("Price QC")
    print(
        f"Rows: {rows:,} | Symbols: {len(price_symbols)} | "
        f"ISIN: {len(price_isins)}"
    )
    print(
        f"Period: {min_date or '-'} -> {max_date or '-'}"
    )
    print(
        f"Mapping: {len(mapping_symbols)} mapped | "
        f"{len(unresolved_mapping)} unresolved"
    )
    print(
        f"Coverage: {len(mapped_without_prices)} mapped without prices | "
        f"{len(prices_without_mapping)} prices without mapping"
    )
    print(
        f"Integrity: {'PASS' if report['integrity']['passed'] else 'FAIL'} | "
        f"Invalid rows: {invalid_rows} | "
        f"Duplicates: {duplicate_rows}"
    )
    print(
        f"Observations: 1080 = "
        f"{report['price_quality']['symbols_with_1080_observations']} | "
        f"<1000 = "
        f"{report['price_quality']['symbols_with_less_than_1000_observations']}"
    )
    print(
        f"Large moves >=50%: {len(large_price_moves)} | "
        f"Gaps >7d: {len(gaps_over_7_days)}"
    )
    print(f"QC report: {QC_FILE}")

    if not report["integrity"]["passed"]:
        raise RuntimeError("Price QC misslyckades på strukturell integritet.")


if __name__ == "__main__":
    main()
