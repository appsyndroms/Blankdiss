"""
Uppdatera OMXSPI-historik för Blankdiss.

Regler:
- Historik byggs från och med 2022-01-01.
- Före 18:00 Europe/Stockholm hämtas högst senaste färdiga handelsdag.
- Från 18:00 hämtas även dagens OMXSPI-värde om Yahoo har publicerat det.
- Hela perioden hämtas från Yahoo varje gång.
- Hämtad data slås ihop med lokal JSONL och ersätter samma datum.
"""

from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from .market_cycles import (
    MARKET_RAW_PATH,
    YAHOO_DOWNLOAD_TIMEOUT_SECONDS,
    _normalize_yahoo_columns,
    _validate_market_frame,
    load_raw_market_data,
    merge_and_persist_market_data,
)

MARKET_START_DATE = pd.Timestamp("2022-01-01")

MARKET_SYMBOL = "OMXSPI"
MARKET_SOURCE_SERIES = "^OMXSPI"

STOCKHOLM_TZ = ZoneInfo("Europe/Stockholm")
MARKET_UPDATE_TIME = time(18, 0)


def _target_end_date(now: pd.Timestamp) -> pd.Timestamp:
    """
    Bestäm senaste datum som får användas.

    Före 18:00:
        endast senaste avslutade kalenderdag.

    Från 18:00:
        dagens värde får användas.
    """
    local_now = now.tz_convert(STOCKHOLM_TZ)

    if local_now.time() >= MARKET_UPDATE_TIME:
        return local_now.normalize()

    return (local_now - pd.Timedelta(days=1)).normalize()


def _download_history(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Hämta OMXSPI från Yahoo/yfinance.

    Yahoo har ett exklusivt end-datum, därför skickas
    end_date + 1 dag till yf.download().
    """
    yahoo_end = end_date + pd.Timedelta(days=1)

    print("Laddar OMXSPI via Yahoo Finance.")
    print(f"Yahoo-symbol: {MARKET_SOURCE_SERIES}")
    print(
        "Yahoo-intervall: "
        f"{start_date.strftime('%Y-%m-%d')} -> "
        f"{end_date.strftime('%Y-%m-%d')}"
    )

    raw = yf.download(
        MARKET_SOURCE_SERIES,
        start=start_date.strftime("%Y-%m-%d"),
        end=yahoo_end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        timeout=YAHOO_DOWNLOAD_TIMEOUT_SECONDS,
    )

    if raw is None or raw.empty:
        raise RuntimeError(
            "Yahoo/yfinance returnerade ingen OMXSPI-data."
        )

    print("Yahoo rå-data:")
    print(f"  Rader: {len(raw)}")
    print(f"  Kolumner: {list(raw.columns)}")

    market = _normalize_yahoo_columns(raw)

    if market.empty:
        raise RuntimeError(
            "Yahoo/yfinance-data kunde inte omvandlas till "
            "market_date/market_close."
        )

    market["market_date"] = pd.to_datetime(
        market["market_date"],
        errors="coerce",
    )

    if getattr(market["market_date"].dt, "tz", None) is not None:
        market["market_date"] = (
            market["market_date"]
            .dt.tz_localize(None)
        )

    market["market_date"] = (
        market["market_date"]
        .dt.normalize()
    )

    market["market_close"] = pd.to_numeric(
        market["market_close"],
        errors="coerce",
    )

    market = market[
        (market["market_date"] >= start_date)
        & (market["market_date"] <= end_date)
    ].copy()

    market = _validate_market_frame(
        market,
        "Yahoo/yfinance",
    )

    if market.empty:
        raise RuntimeError(
            "Yahoo/yfinance returnerade inga giltiga "
            "OMXSPI-observationer i det begärda intervallet."
        )

    return market


def main() -> None:
    print()
    print("==========================================")
    print("BLANKDISS OMXSPI UPDATE")
    print("==========================================")

    now = pd.Timestamp.now(tz=STOCKHOLM_TZ)

    print(
        "Aktuell tid Stockholm:",
        now.strftime("%Y-%m-%d %H:%M:%S %Z"),
    )

    target_end = _target_end_date(now)

    print(
        "Tillåtet OMXSPI-slutdatum:",
        target_end.strftime("%Y-%m-%d"),
    )

    if now.time() >= MARKET_UPDATE_TIME:
        print(
            "Klockan är >= 18:00. "
            "Dagens OMXSPI får hämtas."
        )
    else:
        print(
            "Klockan är < 18:00. "
            "Dagens OMXSPI får INTE användas."
        )

    existing = load_raw_market_data()

    if existing.empty:
        print("Ingen lokal OMXSPI-historik finns ännu.")
    else:
        print(
            "Lokal OMXSPI-historik:",
            f"{len(existing)} rader",
        )
        print(
            "Lokalt intervall:",
            f"{existing['market_date'].min().strftime('%Y-%m-%d')}"
            " -> "
            f"{existing['market_date'].max().strftime('%Y-%m-%d')}",
        )

    downloaded = _download_history(
        MARKET_START_DATE,
        target_end,
    )

    if downloaded.empty:
        print(
            "Yahoo returnerade inga handelsdagar "
            "i intervallet."
        )
        return

    # Säkerhetskontroll:
    # även om Yahoo skulle returnera något oväntat får
    # vi aldrig skriva ett datum efter target_end.
    downloaded = downloaded[
        downloaded["market_date"] <= target_end
    ].copy()

    if downloaded.empty:
        raise RuntimeError(
            "Efter datumfiltrering återstod ingen "
            "giltig OMXSPI-data."
        )

    print()
    print("Yahoo-parsering:")
    print(
        f"  Giltiga observationer: {len(downloaded)}"
    )
    print(
        "  Datum:",
        f"{downloaded['market_date'].min().strftime('%Y-%m-%d')}"
        " -> "
        f"{downloaded['market_date'].max().strftime('%Y-%m-%d')}",
    )
    print(
        "  Första värde:",
        float(downloaded.iloc[0]["market_close"]),
    )
    print(
        "  Sista värde:",
        float(downloaded.iloc[-1]["market_close"]),
    )

    print()
    print("Senaste fem OMXSPI-observationer:")

    print(
        downloaded.tail(5).to_string(
            index=False,
        )
    )

    market = merge_and_persist_market_data(
        existing,
        downloaded,
    )

    print()
    print(
        "Sparad OMXSPI-historik:",
        MARKET_RAW_PATH,
    )
    print(
        "Totalt OMXSPI-rader:",
        len(market),
    )
    print(
        "Totalt intervall:",
        f"{market['market_date'].min().strftime('%Y-%m-%d')}"
        " -> "
        f"{market['market_date'].max().strftime('%Y-%m-%d')}",
    )

    # Slutlig invariant:
    # filen får aldrig innehålla ett datum efter det datum
    # som updatern uttryckligen tillät.
    future_rows = market[
        market["market_date"] > target_end
    ]

    if not future_rows.empty:
        raise RuntimeError(
            "OMXSPI-historiken innehåller observationer "
            "efter tillåtet slutdatum: "
            f"{len(future_rows)} rader."
        )

    # Kontrollera att historiken verkligen börjar 2022-01-01
    # eller senare. Vi kräver inte att 2022-01-01 är en
    # handelsdag eftersom datumet var en helgdag.
    if market["market_date"].min() > MARKET_START_DATE:
        raise RuntimeError(
            "OMXSPI-historiken börjar för sent. "
            f"Förväntade data från {MARKET_START_DATE:%Y-%m-%d}, "
            f"men första observation är "
            f"{market['market_date'].min():%Y-%m-%d}."
        )

    print()
    print("==========================================")
    print("OMXSPI UPDATE KLAR")
    print("==========================================")


if __name__ == "__main__":
    main()
