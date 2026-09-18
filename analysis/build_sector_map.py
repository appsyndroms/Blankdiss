from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yfinance as yf


INSTRUMENT_MAP_PATH = Path(
    "data/analysis/instrument_map.json"
)

OUTPUT_PATH = Path(
    "data/analysis/sector_map.json"
)


def _valid_symbol(
    value: Any,
) -> bool:
    if value is None:
        return False

    value = str(value).strip()

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


def _load_instruments() -> list[dict[str, Any]]:
    payload = json.loads(
        INSTRUMENT_MAP_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "instrument_map.json måste innehålla "
            "ett JSON-objekt."
        )

    instruments = []

    for item in payload.values():
        if not isinstance(item, dict):
            continue

        symbol = item.get(
            "yahoo_symbol"
        )

        if not _valid_symbol(symbol):
            continue

        instruments.append(
            {
                "yahoo_symbol": str(symbol).strip(),
                "issuer": item.get("issuer"),
                "isin": item.get("isin"),
                "lei": item.get("lei"),
            }
        )

    unique = {}

    for item in instruments:
        unique[
            item["yahoo_symbol"]
        ] = item

    return [
        unique[symbol]
        for symbol in sorted(unique)
    ]


def _get_sector(
    symbol: str,
) -> tuple[str | None, str | None]:
    ticker = yf.Ticker(
        symbol
    )

    try:
        info = ticker.info
    except Exception as exc:
        print(
            f"Sektor: kunde inte läsa {symbol}: "
            f"{type(exc).__name__}: {exc}"
        )
        return None, None

    sector = info.get(
        "sector"
    )

    industry = info.get(
        "industry"
    )

    if sector is not None:
        sector = str(
            sector
        ).strip() or None

    if industry is not None:
        industry = str(
            industry
        ).strip() or None

    return sector, industry


def build_sector_map() -> dict[str, dict[str, Any]]:
    instruments = _load_instruments()

    print(
        "Sektor: instrument:",
        len(instruments),
    )

    result: dict[
        str,
        dict[str, Any],
    ] = {}

    failed = 0

    for index, instrument in enumerate(
        instruments,
        start=1,
    ):
        symbol = instrument[
            "yahoo_symbol"
        ]

        print(
            f"Sektor: {index}/{len(instruments)} "
            f"{symbol}"
        )

        sector, industry = _get_sector(
            symbol
        )

        if sector is None:
            failed += 1
            continue

        result[symbol] = {
            "sector": sector,
            "industry": industry,
            "issuer": instrument.get(
                "issuer"
            ),
            "isin": instrument.get(
                "isin"
            ),
            "lei": instrument.get(
                "lei"
            ),
            "source": "Yahoo Finance",
        }

        time.sleep(
            0.05
        )

    print(
        "Sektor: mappade:",
        len(result),
    )

    print(
        "Sektor: saknas:",
        failed,
    )

    if not result:
        raise RuntimeError(
            "Ingen sektormappning kunde skapas."
        )

    return result


def main() -> None:
    if OUTPUT_PATH.exists():
        raise SystemExit(
            "sector_map.json finns redan. "
            "Ta bort filen om mappningen uttryckligen "
            "ska byggas om."
        )

    mapping = build_sector_map()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "version": 1,
        "source": "Yahoo Finance",
        "description": (
            "Fryst sektormappning för Blankdiss. "
            "Mappningen ska inte byggas om automatiskt "
            "vid varje research-körning."
        ),
        "instruments": mapping,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Sektor: skrev {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
