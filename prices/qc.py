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


def parse_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def extract_mapping_entries(mapping: dict) -> list[dict]:
    """
    Instrument_map.json kan vara uppbyggd på olika sätt beroende på
    tidigare version av mapping-koden.

    Vi försöker därför hitta instrumentposter robust utan att anta
    att toppnivån är direkt {isin: instrument}.
    """

    entries = []

    if not isinstance(mapping, dict):
        return entries

    for key, value in mapping.items():

        if not isinstance(value, dict):
            continue

        entry = dict(value)

        # Behåll nyckeln som möjlig identitet.
        entry.setdefault("_mapping_key", key)

        entries.append(entry)

    return entries


def main() -> None:
    print("Price QC: startar.")

    if not FI_RECONSTRUCTED.exists():
        raise FileNotFoundError(FI_RECONSTRUCTED)

    if not MAPPING_FILE.exists():
        raise FileNotFoundError(MAPPING_FILE)

    if not PRICE_DIR.exists():
        raise FileNotFoundError(PRICE_DIR)

    mapping = load_mapping()
    mapping_entries = extract_mapping_entries(mapping)

    latest_price_file = find_latest_price_file()

    print(f"Price QC: mapping entries = {len(mapping_entries)}")
    print(f"Price QC: prisfil = {latest_price_file.name}")

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    mapping_symbols: set[str] = set()
    unresolved_mapping = []
    duplicate_yahoo_symbols: dict[str, list[str]] = defaultdict(list)

    for entry in mapping_entries:

        yahoo_symbol = str(
            entry.get("yahoo_symbol", "")
        ).strip()

        if yahoo_symbol:
            mapping_symbols.add(yahoo_symbol)
            duplicate_yahoo_symbols[yahoo_symbol].append(
                str(entry.get("_mapping_key", ""))
            )
        else:
            unresolved_mapping.append(
                str(entry.get("_mapping_key", ""))
            )

    duplicate_symbols = {
        symbol: keys
        for symbol, keys in duplicate_yahoo_symbols.items()
        if len(keys) > 1
    }

    # ------------------------------------------------------------------
    # Price data
    # ------------------------------------------------------------------

    rows = 0

    invalid_rows = 0
    invalid_missing_symbol = 0
    invalid_missing_date = 0
    invalid_bad_date = 0
    invalid_missing_close = 0
    invalid_nonfinite_close = 0

    duplicate_rows = 0

    price_symbols: set[str] = set()
    price_isins: set[str] = set()

    observations_by_symbol = Counter()

    dates_by_symbol: dict[str, list[date]] = defaultdict(list)

    previous_close: dict[str, float] = {}

    large_price_moves = []
    duplicate_keys: set[tuple[str, str]] = set()

    min_date = None
    max_date = None

    for row in read_jsonl(latest_price_file):

        rows += 1

        yahoo_symbol = str(
            row.get("yahoo_symbol", "")
        ).strip()

        date_text = str(
            row.get("date", "")
        ).strip()

        isin = str(
            row.get("isin", "")
        ).strip()

        close_raw = row.get("close")

        row_invalid = False

        # --------------------------------------------------------------
        # Symbol
        # --------------------------------------------------------------

        if not yahoo_symbol:
            invalid_missing_symbol += 1
            row_invalid = True

        # --------------------------------------------------------------
        # Date
        # --------------------------------------------------------------

        current_date = None

        if not date_text:
            invalid_missing_date += 1
            row_invalid = True
        else:
            try:
                current_date = date.fromisoformat(date_text)
            except ValueError:
                invalid_bad_date += 1
                row_invalid = True

        # --------------------------------------------------------------
        # Close
        # --------------------------------------------------------------

        if close_raw is None or close_raw == "":
            invalid_missing_close += 1
            row_invalid = True

        close_value = parse_float(close_raw)

        if close_raw is not None and close_value is None:
            invalid_nonfinite_close += 1
            row_invalid = True

        if row_invalid:
            invalid_rows += 1
            continue

        # --------------------------------------------------------------
        # Valid row
        # --------------------------------------------------------------

        price_symbols.add(yahoo_symbol)

        if isin:
            price_isins.add(isin)

        observations_by_symbol[yahoo_symbol] += 1

        dates_by_symbol[yahoo_symbol].append(current_date)

        if min_date is None or current_date < min_date:
            min_date = current_date

        if max_date is None or current_date > max_date:
            max_date = current_date

        key = (yahoo_symbol, date_text)

        if key in duplicate_keys:
            duplicate_rows += 1
        else:
            duplicate_keys.add(key)

        previous = previous_close.get(yahoo_symbol)

        if previous is not None and previous > 0:

            relative_change = abs(
                close_value / previous - 1.0
            )

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

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------

    mapped_symbols_without_prices = sorted(
        mapping_symbols - price_symbols
    )

    price_symbols_without_mapping = sorted(
        price_symbols - mapping_symbols
    )

    # ------------------------------------------------------------------
    # Gaps
    # ------------------------------------------------------------------

    gaps_over_7_days = []

    for symbol, dates in dates_by_symbol.items():

        dates = sorted(set(dates))

        for previous_date, current_date in zip(
            dates,
            dates[1:],
        ):

            gap_days = (
                current_date - previous_date
            ).days

            if gap_days > 7:

                gaps_over_7_days.append(
                    {
                        "yahoo_symbol": symbol,
                        "from": previous_date.isoformat(),
                        "to": current_date.isoformat(),
                        "gap_days": gap_days,
                    }
                )

    # ------------------------------------------------------------------
    # Observation statistics
    # ------------------------------------------------------------------

    observation_counts = list(
        observations_by_symbol.values()
    )

    symbols_with_1080 = sum(
        1
        for count in observation_counts
        if count == 1080
    )

    symbols_with_less_than_1000 = sum(
        1
        for count in observation_counts
        if count < 1000
    )

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    # Missing/invalid price rows are reported, but they do NOT automatically
    # fail the entire dataset. We first need to understand whether these are
    # Yahoo gaps, delisted instruments, or an extraction problem.
    #
    # Duplicate symbol/date pairs ARE structural errors.

    integrity_passed = duplicate_rows == 0

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    report = {
        "dataset_summary": {
            "price_file": latest_price_file.name,
            "rows": rows,
            "valid_rows": rows - invalid_rows,
            "invalid_rows": invalid_rows,
            "duplicate_rows": duplicate_rows,
            "unique_price_symbols": len(price_symbols),
            "unique_price_isins": len(price_isins),
            "first_price_date": (
                min_date.isoformat()
                if min_date
                else None
            ),
            "last_price_date": (
                max_date.isoformat()
                if max_date
                else None
            ),
        },

        "invalid_row_breakdown": {
            "missing_symbol": invalid_missing_symbol,
            "missing_date": invalid_missing_date,
            "bad_date": invalid_bad_date,
            "missing_close": invalid_missing_close,
            "nonfinite_close": invalid_nonfinite_close,
        },

        "mapping_summary": {
            "mapping_entries": len(mapping_entries),
            "mapped_symbols": len(mapping_symbols),
            "unresolved_instruments": len(
                unresolved_mapping
            ),
            "duplicate_yahoo_symbols": len(
                duplicate_symbols
            ),
            "mapped_symbols_without_prices": len(
                mapped_symbols_without_prices
            ),
            "price_symbols_without_mapping": len(
                price_symbols_without_mapping
            ),
            "unresolved_examples": (
                unresolved_mapping[:30]
            ),
            "duplicate_yahoo_symbol_examples": {
                symbol: keys
                for symbol, keys in list(
                    duplicate_symbols.items()
                )[:20]
            },
            "mapped_symbols_without_prices_examples": (
                mapped_symbols_without_prices[:30]
            ),
            "price_symbols_without_mapping_examples": (
                price_symbols_without_mapping[:30]
            ),
        },

        "price_quality": {
            "symbols_with_1080_observations": (
                symbols_with_1080
            ),
            "symbols_with_less_than_1000_observations": (
                symbols_with_less_than_1000
            ),
            "largest_observation_count": (
                max(observation_counts)
                if observation_counts
                else 0
            ),
            "smallest_observation_count": (
                min(observation_counts)
                if observation_counts
                else 0
            ),
            "gaps_over_7_days": len(
                gaps_over_7_days
            ),
            "large_price_moves_ge_50pct": len(
                large_price_moves
            ),
            "large_price_move_examples": (
                large_price_moves[:20]
            ),
            "gap_examples": (
                gaps_over_7_days[:20]
            ),
        },

        "integrity": {
            "invalid_rows": invalid_rows,
            "duplicate_rows": duplicate_rows,
            "passed": integrity_passed,
        },
    }

    PRICE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with QC_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    print()
    print("Price QC")

    print(
        f"Rows: {rows:,} | "
        f"Valid: {rows - invalid_rows:,} | "
        f"Invalid: {invalid_rows:,}"
    )

    print(
        f"Symbols: {len(price_symbols)} | "
        f"ISIN: {len(price_isins)}"
    )

    print(
        f"Period: {min_date or '-'} -> "
        f"{max_date or '-'}"
    )

    print(
        f"Mapping: {len(mapping_symbols)} mapped | "
        f"{len(unresolved_mapping)} unresolved"
    )

    print(
        f"Coverage: "
        f"{len(mapped_symbols_without_prices)} mapped without prices | "
        f"{len(price_symbols_without_mapping)} "
        f"prices without mapping"
    )

    print(
        f"Invalid breakdown: "
        f"missing symbol={invalid_missing_symbol} | "
        f"missing date={invalid_missing_date} | "
        f"bad date={invalid_bad_date} | "
        f"missing close={invalid_missing_close} | "
        f"nonfinite close={invalid_nonfinite_close}"
    )

    print(
        f"Integrity: "
        f"{'PASS' if integrity_passed else 'FAIL'} | "
        f"Duplicates: {duplicate_rows}"
    )

    print(
        f"Observations: 1080 = "
        f"{symbols_with_1080} | "
        f"<1000 = "
        f"{symbols_with_less_than_1000}"
    )

    print(
        f"Large moves >=50%: "
        f"{len(large_price_moves)} | "
        f"Gaps >7d: "
        f"{len(gaps_over_7_days)}"
    )

    print(
        f"QC report: {QC_FILE}"
    )

    if not integrity_passed:
        raise RuntimeError(
            "Price QC misslyckades: "
            "duplicerade symbol/datum-kombinationer."
        )


if __name__ == "__main__":
    main()
