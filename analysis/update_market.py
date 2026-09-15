"""
Uppdatera OMXSPI-historik för Blankdiss.

Regler:
- Historik byggs från och med 2022-01-01.
- Före 18:00 Europe/Stockholm används högst senaste färdiga handelsdag.
- Från 18:00 får även dagens OMXSPI-värde användas.
- All market_date-data sparas som timezone-naiva kalenderdatum.
- Hela perioden hämtas från Yahoo varje gång.
- Hämtad data slås ihop med lokal JSONL och ersätter samma datum.
- Lokala observationer efter tillåtet slutdatum tas bort.
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
    Bestäm senaste kalenderdatum som får användas.

    Före 18:00:
        endast senaste avslutade kalenderdag.

    Från 18:00:
        dagens värde får användas.

    Returnerar alltid ett timezone-naivt Timestamp eftersom
    market_date i Blankdiss representerar handelsdagar,
    inte tidpunkter.
    """
    local_now = now.tz_convert(STOCKHOLM_TZ)

    if local_now.time() >= MARKET_UPDATE_TIME:
        target = local_now.normalize()
    else:
        target = (
            local_now - pd.Timedelta(days=1)
        ).normalize()

    return target.tz_localize(None)


def _normalize_market_dates(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalisera market_date till timezone-naiva kalenderdatum.
    """
    frame = frame.copy()

    frame["market_date"] = pd.to_datetime(
        frame["market_date"],
        errors="coerce",
    )

    if getattr(frame["market_date"].dt, "tz", None) is not None:
        frame["market_date"] = (
            frame["market_date"]
            .dt.tz_convert(STOCKHOLM_TZ)
            .dt.tz_localize(None)
        )

    frame["market_date"] = (
        frame["market_date"]
        .dt.normalize()
    )

    return frame


def _download_history(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Hämta OMXSPI från Yahoo/yfinance.

    Yahoo har ett exklusivt end-datum, därför skickas
    end_date + 1 dag till yf.download().
    """
    start_date = pd.Timestamp(start_date).tz_localize(None)
    end_date = pd.Timestamp(end_date).tz_localize(None)

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

    market = _normalize_market_dates(market)

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

    now = pd.Timestamp.now(
        tz=STOCKHOLM_TZ,
    )

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
        print(
            "Ingen lokal OMXSPI-historik finns ännu."
        )
    else:
        existing = _normalize_market_dates(existing)

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

        # Om workflowet körs före 18:00 får en eventuell
        # tidigare intradagsobservation från idag inte ligga kvar.
        before_filter = len(existing)

        existing = existing[
            existing["market_date"] <= target_end
        ].copy()

        removed_future = (
            before_filter - len(existing)
        )

        if removed_future:
            print(
                "Tar bort lokala observationer efter "
                "tillåtet slutdatum:",
                removed_future,
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

    # Extra säkerhetskontroll.
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

    market = _normalize_market_dates(market)

    # Säkerställ att den sparade datamängden inte innehåller
    # observationer efter tillåtet slutdatum.
    market = market[
        market["market_date"] <= target_end
    ].copy()

    # Säkerställ unik sorterad historik.
    market = (
        market[
            [
                "market_date",
                "market_close",
            ]
        ]
        .drop_duplicates(
            subset=["market_date"],
            keep="last",
        )
        .sort_values("market_date")
        .reset_index(drop=True)
    )

    # Skriv den slutligt validerade datamängden.
    merge_and_persist_market_data(
        pd.DataFrame(
            columns=[
                "market_date",
                "market_close",
            ]
        ),
        market,
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

    future_rows = market[
        market["market_date"] > target_end
    ]

    if not future_rows.empty:
        raise RuntimeError(
            "OMXSPI-historiken innehåller observationer "
            "efter tillåtet slutdatum: "
            f"{len(future_rows)} rader."
        )

    if market["market_date"].min() > MARKET_START_DATE:
        raise RuntimeError(
            "OMXSPI-historiken börjar för sent. "
            f"Förväntade data från "
            f"{MARKET_START_DATE:%Y-%m-%d}, "
            f"men första observation är "
            f"{market['market_date'].min():%Y-%m-%d}."
        )

    print()
    print("==========================================")
    print("OMXSPI UPDATE KLAR")
    print("==========================================")


if __name__ == "__main__":
    main()
