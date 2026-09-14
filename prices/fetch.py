from __future__ import annotations
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any
import yfinance as yf
OUTPUT_DIR = Path(
    "data/raw/prices"
)
def _valid_symbol(
    symbol: Any,
) -> bool:
    if symbol is None:
        return False
    value = str(symbol).strip()
    if not value:
        return False
    if value.upper() in {
        "NONE",
        "NULL",
        "NAN",
        "NAT",
    }:
        return False
    return True
def _normalise_date(
    value: str | date,
) -> date:
    """
    Convert a date or ISO date string to a date object.
    __main__.py passes command-line arguments as strings,
    while other callers may pass datetime.date objects.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(
            value
        )
    raise TypeError(
        "Date must be a date or ISO "
        f"date string, got {type(value).__name__}"
    )
def _extract_close(
    data,
    symbol: str,
):
    """
    Extract Close from yfinance output.
    Handles both:
      - MultiIndex columns
      - ordinary DataFrame columns
    """
    if data is None or data.empty:
        return None
    # MultiIndex:
    # ('Close', 'VOLV-B.ST')
    if hasattr(
        data.columns,
        "levels",
    ):
        try:
            level0 = (
                data.columns
                .get_level_values(0)
            )
            if "Close" in level0:
                close = data["Close"]
                if hasattr(
                    close,
                    "columns",
                ):
                    if (
                        symbol
                        in close.columns
                    ):
                        return close[symbol]
                    if (
                        len(close.columns)
                        == 1
                    ):
                        return close.iloc[
                            :,
                            0,
                        ]
                return close
        except Exception:
            pass
    # Ordinary DataFrame:
    if "Close" in data.columns:
        return data["Close"]
    return None
def fetch_prices(
    instruments: list[dict[str, Any]],
    start: str | date,
    end: str | date,
) -> list[dict[str, Any]]:
    """
    Fetch historical closing prices from Yahoo Finance.
    start/end may be supplied either as:
      - ISO date strings, e.g. "2022-05-25"
      - datetime.date objects
    Instruments without a valid Yahoo symbol are skipped.
    """
    start_date = _normalise_date(
        start
    )
    end_date = _normalise_date(
        end
    )
    if end_date <= start_date:
        raise ValueError(
            "End date must be later "
            "than start date: "
            f"{start_date} -> {end_date}"
        )
    valid_instruments: list[
        dict[str, Any]
    ] = []
    skipped = 0
    for instrument in instruments:
        symbol = instrument.get(
            "yahoo_symbol"
        )
        if not _valid_symbol(
            symbol
        ):
            skipped += 1
            print(
                "Pris: hoppar över "
                "ogiltig Yahoo-symbol - "
                f"{symbol}"
            )
            continue
        valid_instruments.append(
            instrument
        )
    if skipped:
        print(
            f"Pris: {skipped} instrument "
            "hoppades över eftersom "
            "Yahoo-symbol saknas."
        )
    symbols = sorted(
        {
            instrument[
                "yahoo_symbol"
            ]
            for instrument
            in valid_instruments
        }
    )
    if not symbols:
        print(
            "Pris: inga giltiga "
            "Yahoo-symboler att hämta."
        )
        return []
    print(
        "Priser: hämtar "
        f"{len(symbols)} instrument "
        "i en batch..."
    )
    data = yf.download(
        tickers=symbols,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        auto_adjust=False,
        actions=False,
        threads=True,
        timeout=20,
        progress=False,
        group_by="column",
    )
    if data is None or data.empty:
        print(
            "Pris: Yahoo returnerade "
            "ingen prisdata."
        )
        return []
    instrument_by_symbol = {
        instrument[
            "yahoo_symbol"
        ]: instrument
        for instrument
        in valid_instruments
    }
    records: list[
        dict[str, Any]
    ] = []
    for symbol in symbols:
        instrument = (
            instrument_by_symbol[
                symbol
            ]
        )
        close_series = (
            _extract_close(
                data,
                symbol,
            )
        )
        if close_series is None:
            print(
                f"Pris: saknas - {symbol}"
            )
            continue
        count = 0
        for timestamp, value in (
            close_series.items()
        ):
            try:
                close = float(value)
            except (
                TypeError,
                ValueError,
            ):
                continue
            records.append(
                {
                    "date": (
                        timestamp.strftime(
                            "%Y-%m-%d"
                        )
                    ),
                    "isin": instrument.get(
                        "isin"
                    ),
                    "lei": instrument.get(
                        "lei"
                    ),
                    "issuer": instrument.get(
                        "issuer"
                    ),
                    "ticker": instrument.get(
                        "ticker"
                    ),
                    "yahoo_symbol": symbol,
                    "mapping_source": (
                        instrument.get(
                            "mapping_source"
                        )
                    ),
                    "close": close,
                }
            )
            count += 1
        print(
            f"Pris: {symbol} - "
            f"{count} observationer"
        )
    return records
def write_jsonl(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Write price records as JSONL.
    Kept as a public compatibility function
    because prices.__main__ imports it.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            handle.write("\n")
def save_prices(
    records: list[dict[str, Any]],
    *,
    start: str | date,
    end: str | date,
) -> Path:
    """
    Convenience wrapper for writing
    the standard price filename.
    """
    start_date = _normalise_date(
        start
    )
    end_date = _normalise_date(
        end
    )
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    output = (
        OUTPUT_DIR
        / (
            "prices_"
            f"{start_date.isoformat()}"
            "_"
            f"{end_date.isoformat()}"
            ".jsonl"
        )
    )
    write_jsonl(
        records,
        output,
    )
    return output
