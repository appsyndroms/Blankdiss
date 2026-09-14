"""
Hämtar och sparar historiska prisdata från Yahoo Finance.
"""
from __future__ import annotations
import json
from pathlib import Path
import yfinance as yf
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "prices"
)
PRICE_TIMEOUT_SECONDS = 20
def extract_close_series(
    data,
    symbol: str,
):
    """
    Hämtar Close från yfinance.
    Hanterar både vanliga kolumner och MultiIndex.
    """
    if data is None:
        return None
    if data.empty:
        return None
    if "Close" not in data.columns:
        return None
    close = data["Close"]
    if hasattr(
        close,
        "columns",
    ):
        if symbol in close.columns:
            close = close[
                symbol
            ]
        elif len(
            close.columns
        ) == 1:
            close = close.iloc[
                :,
                0
            ]
        else:
            return None
    return close
def fetch_prices(
    instruments: list[dict],
    start: str,
    end: str | None = None,
) -> list[dict]:
    if not instruments:
        return []
    symbols = [
        instrument[
            "yahoo_symbol"
        ]
        for instrument in instruments
    ]
    symbols = list(
        dict.fromkeys(
            symbols
        )
    )
    print(
        "Priser: hämtar "
        f"{len(symbols)} instrument "
        "i en batch..."
    )
    try:
        data = yf.download(
            symbols,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            actions=False,
            threads=True,
            timeout=PRICE_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        print(
            "Priser: FEL vid "
            "batchhämtning - "
            f"{type(exc).__name__}: {exc}"
        )
        return []
    if data is None or data.empty:
        print(
            "Priser: Yahoo returnerade "
            "ingen prisdata."
        )
        return []
    instrument_by_symbol = {
        instrument[
            "yahoo_symbol"
        ]: instrument
        for instrument in instruments
    }
    records: list[
        dict
    ] = []
    for symbol in symbols:
        instrument = (
            instrument_by_symbol[
                symbol
            ]
        )
        close = extract_close_series(
            data,
            symbol,
        )
        if close is None:
            print(
                "Pris: saknas - "
                f"{symbol}"
            )
            continue
        symbol_records = 0
        for timestamp, value in close.items():
            if value is None:
                continue
            try:
                price = float(
                    value
                )
            except (
                TypeError,
                ValueError,
            ):
                continue
            if price <= 0:
                continue
            records.append(
                {
                    "date": timestamp.strftime(
                        "%Y-%m-%d"
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
                    "close": price,
                }
            )
            symbol_records += 1
        print(
            "Pris: "
            f"{symbol} - "
            f"{symbol_records} observationer"
        )
    return records
def write_jsonl(
    records: list[dict],
    start: str,
    end: str | None = None,
) -> Path:
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    end_label = (
        end
        if end is not None
        else "latest"
    )
    filename = (
        "prices_"
        f"{start}_"
        f"{end_label}.jsonl"
    )
    path = (
        RAW_DIR
        / filename
    )
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path
