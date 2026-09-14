from __future__ import annotations
from datetime import date
from pathlib import Path
from typing import Any
import yfinance as yf
OUTPUT_DIR = Path("data/raw/prices")
def _valid_symbol(symbol: Any) -> bool:
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
def _extract_close(
    data,
    symbol: str,
):
    """
    Handle both the normal yfinance MultiIndex result and simpler
    DataFrame layouts.
    """
    if data is None or data.empty:
        return None
    # MultiIndex columns:
    #
    # ('Close', 'VOLV-B.ST')
    #
    if hasattr(data.columns, "levels"):
        try:
            if "Close" in data.columns.get_level_values(0):
                close = data["Close"]
                if hasattr(close, "columns"):
                    if symbol in close.columns:
                        return close[symbol]
                    if len(close.columns) == 1:
                        return close.iloc[:, 0]
                return close
        except Exception:
            pass
    # Standard single-level DataFrame.
    if "Close" in data.columns:
        return data["Close"]
    return None
def fetch_prices(
    *,
    instruments: list[dict[str, Any]],
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    valid_instruments = []
    skipped = 0
    for instrument in instruments:
        symbol = instrument.get("yahoo_symbol")
        if not _valid_symbol(symbol):
            skipped += 1
            print(
                "Pris: hoppar över ogiltig Yahoo-symbol - "
                f"{symbol}"
            )
            continue
        valid_instruments.append(instrument)
    if skipped:
        print(
            f"Pris: {skipped} instrument hoppades över "
            "eftersom Yahoo-symbol saknas."
        )
    symbols = sorted(
        {
            instrument["yahoo_symbol"]
            for instrument in valid_instruments
        }
    )
    if not symbols:
        print(
            "Pris: inga giltiga Yahoo-symboler att hämta."
        )
        return []
    print(
        f"Priser: hämtar {len(symbols)} instrument i en batch..."
    )
    data = yf.download(
        tickers=symbols,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=False,
        actions=False,
        threads=True,
        timeout=20,
        progress=False,
        group_by="column",
    )
    if data is None or data.empty:
        print(
            "Pris: Yahoo returnerade ingen prisdata."
        )
        return []
    instrument_by_symbol = {
        instrument["yahoo_symbol"]: instrument
        for instrument in valid_instruments
    }
    records: list[dict[str, Any]] = []
    for symbol in symbols:
        instrument = instrument_by_symbol[symbol]
        close_series = _extract_close(
            data,
            symbol,
        )
        if close_series is None:
            print(
                f"Pris: saknas - {symbol}"
            )
            continue
        count = 0
        for timestamp, value in close_series.items():
            try:
                close = float(value)
            except (TypeError, ValueError):
                continue
            records.append(
                {
                    "date": timestamp.strftime("%Y-%m-%d"),
                    "isin": instrument.get("isin"),
                    "lei": instrument.get("lei"),
                    "issuer": instrument.get("issuer"),
                    "ticker": instrument.get("ticker"),
                    "yahoo_symbol": symbol,
                    "mapping_source": instrument.get(
                        "mapping_source"
                    ),
                    "close": close,
                }
            )
            count += 1
        print(
            f"Pris: {symbol} - {count} observationer"
        )
    return records
def save_prices(
    records: list[dict[str, Any]],
    *,
    start: date,
    end: date,
) -> Path:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    output = (
        OUTPUT_DIR
        / f"prices_{start.isoformat()}_{end.isoformat()}.jsonl"
    )
    with output.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                __import__("json").dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            handle.write("\n")
    return output
