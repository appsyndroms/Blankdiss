"""
Uppdatera OMXSPI-historik för Blankdiss.

Regler:
- Historik byggs från och med 2022-01-01.
- Före 18:00 Europe/Stockholm används högst senaste färdiga handelsdag.
- Från 18:00 får även dagens OMXSPI-värde användas.
- All market_date-data sparas som timezone-naiva kalenderdatum.
- Endast historik som saknas lokalt hämtas från Yahoo.
- Hämtad data slås ihop med lokal JSONL.
- Lokala observationer efter tillåtet slutdatum tas bort.
"""

from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

import numpy as np
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


MARKET_START_DATE = pd.Timestamp(
    "2022-01-01"
)

MARKET_SYMBOL = "OMXSPI"
MARKET_SOURCE_SERIES = "^OMXSPI"

STOCKHOLM_TZ = ZoneInfo(
    "Europe/Stockholm"
)

MARKET_UPDATE_TIME = time(
    18,
    0,
)


def _target_end_date(
    now: pd.Timestamp,
) -> pd.Timestamp:
    """
    Bestäm senaste kalenderdatum som får användas.

    Före 18:00:
        endast senaste avslutade kalenderdag.

    Från 18:00:
        dagens värde får användas.
    """

    local_now = now.tz_convert(
        STOCKHOLM_TZ
    )

    if (
        local_now.time()
        >= MARKET_UPDATE_TIME
    ):
        target = (
            local_now.normalize()
        )
    else:
        target = (
            local_now
            - pd.Timedelta(days=1)
        ).normalize()

    return target.tz_localize(
        None
    )


def _normalize_market_dates(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalisera market_date till
    timezone-naiva kalenderdatum.
    """

    frame = frame.copy()

    frame["market_date"] = (
        pd.to_datetime(
            frame["market_date"],
            errors="coerce",
        )
    )

    if (
        getattr(
            frame[
                "market_date"
            ].dt,
            "tz",
            None,
        )
        is not None
    ):
        frame["market_date"] = (
            frame[
                "market_date"
            ]
            .dt.tz_convert(
                STOCKHOLM_TZ
            )
            .dt.tz_localize(
                None
            )
        )

    frame["market_date"] = (
        frame[
            "market_date"
        ]
        .dt.normalize()
    )

    return frame


def _download_history(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Hämta endast det saknade OMXSPI-intervallet.
    """

    start_date = pd.Timestamp(
        start_date
    ).tz_localize(None)

    end_date = pd.Timestamp(
        end_date
    ).tz_localize(None)

    yahoo_end = (
        end_date
        + pd.Timedelta(days=1)
    )

    print(
        "Yahoo-intervall: "
        f"{start_date.strftime('%Y-%m-%d')} -> "
        f"{end_date.strftime('%Y-%m-%d')}"
    )

    raw = yf.download(
        MARKET_SOURCE_SERIES,
        start=start_date.strftime(
            "%Y-%m-%d"
        ),
        end=yahoo_end.strftime(
            "%Y-%m-%d"
        ),
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        timeout=(
            YAHOO_DOWNLOAD_TIMEOUT_SECONDS
        ),
    )

    if (
        raw is None
        or raw.empty
    ):
        raise RuntimeError(
            "Yahoo/yfinance returnerade "
            "ingen OMXSPI-data."
        )

    market = (
        _normalize_yahoo_columns(
            raw
        )
    )

    if market.empty:
        raise RuntimeError(
            "Yahoo/yfinance-data kunde "
            "inte omvandlas till "
            "market_date/market_close."
        )

    market = (
        _normalize_market_dates(
            market
        )
    )

    market[
        "market_close"
    ] = pd.to_numeric(
        market[
            "market_close"
        ],
        errors="coerce",
    )

    market = market[
        (
            market[
                "market_date"
            ]
            >= start_date
        )
        & (
            market[
                "market_date"
            ]
            <= end_date
        )
    ].copy()

    # Yahoo/yfinance kan ibland returnera enstaka
    # råobservationer utan ett giltigt Close-värde.
    #
    # Dessa är inte användbara marknadsobservationer
    # och ska därför filtreras bort innan den strikta
    # marknadsvalideringen körs.
    invalid_rows = market[
        (
            market[
                "market_date"
            ].isna()
        )
        |
        (
            market[
                "market_close"
            ].isna()
        )
        |
        (
            ~np.isfinite(
                market[
                    "market_close"
                ].to_numpy(
                    dtype=float
                )
            )
        )
        |
        (
            market[
                "market_close"
            ]
            <= 0
        )
    ]

    if not invalid_rows.empty:
        print(
            "Yahoo/yfinance returnerade "
            f"{len(invalid_rows)} ogiltiga "
            "råobservationer. "
            "Dessa ignoreras före validering."
        )

        print(
            invalid_rows.to_string(
                index=False
            )
        )

        market = market.drop(
            invalid_rows.index
        )

    market = _validate_market_frame(
        market,
        "Yahoo/yfinance",
    )

    if market.empty:
        raise RuntimeError(
            "Yahoo/yfinance returnerade "
            "inga giltiga OMXSPI-observationer "
            "i det begärda intervallet."
        )

    print(
        "Yahoo: "
        f"{len(market)} nya OMXSPI-observationer."
    )

    return market


def main() -> None:
    print()
    print(
        "=========================================="
    )
    print(
        "BLANKDISS OMXSPI UPDATE"
    )
    print(
        "=========================================="
    )

    now = pd.Timestamp.now(
        tz=STOCKHOLM_TZ,
    )

    print(
        "Aktuell tid Stockholm:",
        now.strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        ),
    )

    target_end = (
        _target_end_date(
            now
        )
    )

    print(
        "Tillåtet OMXSPI-slutdatum:",
        target_end.strftime(
            "%Y-%m-%d"
        ),
    )

    existing = (
        load_raw_market_data()
    )

    if existing.empty:
        print(
            "Ingen lokal OMXSPI-historik finns ännu."
        )

        local_start = (
            MARKET_START_DATE
        )

    else:
        existing = (
            _normalize_market_dates(
                existing
            )
        )

        print(
            "Lokal OMXSPI-historik:",
            f"{len(existing)} rader",
        )

        local_min = (
            existing[
                "market_date"
            ].min()
        )

        local_max = (
            existing[
                "market_date"
            ].max()
        )

        print(
            "Lokalt intervall:",
            f"{local_min:%Y-%m-%d}",
            "->",
            f"{local_max:%Y-%m-%d}",
        )

        before_filter = len(
            existing
        )

        existing = existing[
            existing[
                "market_date"
            ]
            <= target_end
        ].copy()

        removed_future = (
            before_filter
            - len(existing)
        )

        if removed_future:
            print(
                "Tar bort lokala observationer "
                "efter tillåtet slutdatum:",
                removed_future,
            )

        local_min = (
            existing[
                "market_date"
            ].min()
        )

        local_max = (
            existing[
                "market_date"
            ].max()
        )

        if pd.isna(local_max):
            local_start = (
                MARKET_START_DATE
            )
        else:
            local_start = (
                local_max
                + pd.Timedelta(days=1)
            )

    # Vi ska aldrig börja före den historiska startpunkten.
    local_start = max(
        local_start,
        MARKET_START_DATE,
    )

    if local_start > target_end:
        print(
            "OMXSPI är redan uppdaterad "
            "till tillåtet slutdatum."
        )

        market = existing.copy()

        if market.empty:
            raise RuntimeError(
                "OMXSPI saknar lokal historik "
                "trots att ingen hämtning behövdes."
            )

    else:
        print(
            "OMXSPI: hämtar endast saknad historik:"
        )

        print(
            f"  {local_start:%Y-%m-%d}"
            " -> "
            f"{target_end:%Y-%m-%d}"
        )

        downloaded = _download_history(
            local_start,
            target_end,
        )

        if existing.empty:
            market = downloaded
        else:
            market = pd.concat(
                [
                    existing,
                    downloaded,
                ],
                ignore_index=True,
            )

        market = (
            _normalize_market_dates(
                market
            )
        )

        market[
            "market_close"
        ] = pd.to_numeric(
            market[
                "market_close"
            ],
            errors="coerce",
        )

        market = (
            market[
                [
                    "market_date",
                    "market_close",
                ]
            ]
            .dropna(
                subset=[
                    "market_date",
                    "market_close",
                ]
            )
            .drop_duplicates(
                subset=[
                    "market_date"
                ],
                keep="last",
            )
            .sort_values(
                "market_date"
            )
            .reset_index(
                drop=True
            )
        )

    # Säkerställ att inget efter tillåtet slutdatum finns kvar.
    market = market[
        market[
            "market_date"
        ]
        <= target_end
    ].copy()

    # Sista valideringen.
    market = (
        market[
            [
                "market_date",
                "market_close",
            ]
        ]
        .drop_duplicates(
            subset=[
                "market_date"
            ],
            keep="last",
        )
        .sort_values(
            "market_date"
        )
        .reset_index(
            drop=True
        )
    )

    if market.empty:
        raise RuntimeError(
            "OMXSPI-historiken blev tom "
            "efter uppdateringen."
        )

    future_rows = market[
        market[
            "market_date"
        ]
        > target_end
    ]

    if not future_rows.empty:
        raise RuntimeError(
            "OMXSPI-historiken innehåller "
            "observationer efter tillåtet "
            "slutdatum: "
            f"{len(future_rows)} rader."
        )

    first_market_date = (
        market[
            "market_date"
        ].min()
    )

    if (
        first_market_date.year != 2022
        or first_market_date.month != 1
    ):
        raise RuntimeError(
            "OMXSPI-historiken börjar "
            "oväntat sent. "
            f"Första observation är "
            f"{first_market_date:%Y-%m-%d}."
        )

    # Persistera den redan validerade kompletta historiken.
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
        f"{market['market_date'].min():%Y-%m-%d}",
        "->",
        f"{market['market_date'].max():%Y-%m-%d}",
    )

    print()
    print(
        "=========================================="
    )
    print(
        "OMXSPI UPDATE KLAR"
    )
    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()
