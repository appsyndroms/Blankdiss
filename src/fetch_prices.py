"""
Hämtar historiska dagskurser från Yahoo Finance.

Detta är V0.1-prototypen för prisdata.
ISIN -> ticker-mappningen hålls separat från prisinsamlingen.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


DEFAULT_TICKERS = {
    "VOLV-B.ST": "Aktiebolaget Volvo",
    "SAAB-B.ST": "SAAB",
    "HEXAB.ST": "Hexatronic Group",
    "INVE-B.ST": "Investor",
    "ERIC-B.ST": "Ericsson",
}


def fetch_prices(
    tickers: list[str],
    start: str = "2020-01-01",
) -> list[dict]:
    records: list[dict] = []

    for ticker in tickers:
        data = yf.download(
            ticker,
            start=start,
            auto_adjust=False,
            progress=False,
        )

        if data.empty:
            continue

        for timestamp, row in data.iterrows():
            close = row["Close"]

            if hasattr(close, "iloc"):
                close = close.iloc[0]

            records.append(
                {
                    "date": timestamp.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "close": float(close),
                }
            )

    return records


def write_jsonl(records: list[dict]) -> Path:
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        RAW_DIR
        / f"prices_{date.today().isoformat()}.jsonl"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(record)
                + "\n"
            )

    return path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ticker",
        action="append",
        dest="tickers",
        help="Yahoo-ticker. Kan anges flera gånger.",
    )

    parser.add_argument(
        "--start",
        default="2020-01-01",
    )

    args = parser.parse_args()

    tickers = (
        args.tickers
        if args.tickers
        else list(DEFAULT_TICKERS)
    )

    records = fetch_prices(
        tickers=tickers,
        start=args.start,
    )

    path = write_jsonl(records)

    print(
        f"Priser: {len(records)} observationer → {path}"
    )


if __name__ == "__main__":
    main()
