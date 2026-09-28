"""Beräkning av marknads- och sektorsrelativa prisfeatures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.feature_config import (
    MARKET_PATH,
    RELATIVE_HORIZONS,
    SECTOR_MAP_PATH,
)


ROOT = Path(__file__).resolve().parents[1]


def load_market(
    path: Path = MARKET_PATH,
) -> pd.DataFrame:
    """Läser lokal OMXSPI-historik."""

    if not path.exists():
        raise FileNotFoundError(
            f"Saknar marknadsdata: {path}"
        )

    frame = pd.read_json(
        path,
        lines=True,
    )

    required = {
        "market_date",
        "market_close",
    }

    missing = sorted(
        required.difference(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            "Marknadsdata saknar kolumner:\n"
            + "\n".join(
                f"- {column}"
                for column in missing
            )
        )

    frame["market_date"] = pd.to_datetime(
        frame["market_date"],
        errors="coerce",
    )

    frame["market_close"] = pd.to_numeric(
        frame["market_close"],
        errors="coerce",
    )

    frame = frame.loc[
        frame["market_date"].notna()
        & frame["market_close"].notna()
        & np.isfinite(
            frame["market_close"]
        )
        & (
            frame["market_close"] > 0
        )
    ].copy()

    frame = (
        frame.sort_values(
            "market_date",
            kind="mergesort",
        )
        .drop_duplicates(
            subset=[
                "market_date",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return frame


def load_sector_map(
    path: Path = SECTOR_MAP_PATH,
) -> dict[str, str]:
    """Läser den frysta sektormappningen."""

    if not path.exists():
        raise FileNotFoundError(
            f"Saknar sektorkarta: {path}"
        )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    instruments = data.get(
        "instruments"
    )

    if not isinstance(
        instruments,
        dict,
    ):
        raise ValueError(
            "Sektorkartan saknar "
            "'instruments'."
        )

    mapping: dict[str, str] = {}

    for symbol, metadata in instruments.items():
        if not isinstance(
            metadata,
            dict,
        ):
            continue

        sector = metadata.get(
            "sector"
        )

        if (
            sector is None
            or not str(sector).strip()
        ):
            continue

        mapping[
            str(symbol).strip()
        ] = str(sector).strip()

    if not mapping:
        raise ValueError(
            "Sektorkartan innehåller "
            "inga giltiga sektorer."
        )

    return mapping


def _add_horizon_returns(
    frame: pd.DataFrame,
    *,
    date_column: str,
    close_column: str,
    prefix: str,
    horizons: tuple[int, ...],
    group_column: str | None = None,
) -> pd.DataFrame:
    """
    Lägger till historiska avkastningar.

    Return definieras som dagens close relativt close
    N handelsobservationer tidigare.

    Endast information på eller före aktuell dag används.
    """

    result = frame.copy()

    if group_column is None:
        groups = [
            (
                None,
                result.sort_values(
                    date_column,
                    kind="mergesort",
                ),
            )
        ]
    else:
        groups = result.groupby(
            group_column,
            sort=False,
        )

    parts: list[pd.DataFrame] = []

    for _, group in groups:
        group = group.sort_values(
            date_column,
            kind="mergesort",
        ).copy()

        close = pd.to_numeric(
            group[close_column],
            errors="coerce",
        )

        for horizon in horizons:
            previous = close.shift(
                horizon
            )

            group[
                f"{prefix}_{horizon}d"
            ] = (
                close / previous - 1.0
            )

        parts.append(group)

    if not parts:
        for horizon in horizons:
            result[
                f"{prefix}_{horizon}d"
            ] = np.nan

        return result

    return (
        pd.concat(
            parts,
            ignore_index=False,
        )
        .sort_index()
    )


def _build_sector_returns(
    prices: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """
    Beräknar sektorsavkastning som medianen av
    sektorbolagens respektive historiska prisavkastning.

    Medianen gör att ett enskilt extremt bolag inte
    dominerar sektorns kontrollvariabel.

    Sektormedlemskapet kommer från den frysta
    sektorkartan.
    """

    frame = prices.copy()

    frame["sector"] = frame[
        "yahoo_symbol"
    ].map(
        sector_map
    )

    frame = frame.loc[
        frame["sector"].notna()
        & frame["sector"].astype(str).str.strip().ne("")
    ].copy()

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "price_date",
                "sector",
                *[
                    f"sector_return_{horizon}d"
                    for horizon in RELATIVE_HORIZONS
                ],
            ]
        )

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    frame["close"] = pd.to_numeric(
        frame["close"],
        errors="coerce",
    )

    frame = frame.loc[
        frame["date"].notna()
        & frame["close"].notna()
        & np.isfinite(
            frame["close"]
        )
        & (
            frame["close"] > 0
        )
    ].copy()

    # En symbol ska endast ha ett pris per handelsdag.
    frame = (
        frame.sort_values(
            [
                "yahoo_symbol",
                "date",
            ],
            kind="mergesort",
        )
        .drop_duplicates(
            subset=[
                "yahoo_symbol",
                "date",
            ],
            keep="last",
        )
    )

    # Individuell historisk avkastning för varje aktie.
    for horizon in RELATIVE_HORIZONS:
        frame[
            f"_stock_return_{horizon}d"
        ] = (
            frame.groupby(
                "yahoo_symbol",
                sort=False,
            )["close"]
            .transform(
                lambda series, h=horizon:
                series / series.shift(h) - 1.0
            )
        )

    aggregation_columns = [
        f"_stock_return_{horizon}d"
        for horizon in RELATIVE_HORIZONS
    ]

    sector_returns = (
        frame.groupby(
            [
                "sector",
                "date",
            ],
            sort=False,
        )[aggregation_columns]
        .median()
        .reset_index()
    )

    rename_map = {
        f"_stock_return_{horizon}d":
        f"sector_return_{horizon}d"
        for horizon in RELATIVE_HORIZONS
    }

    sector_returns = (
        sector_returns.rename(
            columns=rename_map
        )
        .rename(
            columns={
                "date": "price_date"
            }
        )
    )

    return sector_returns[
        [
            "price_date",
            "sector",
            *[
                f"sector_return_{horizon}d"
                for horizon in RELATIVE_HORIZONS
            ],
        ]
    ]


def _build_market_returns(
    market: pd.DataFrame,
) -> pd.DataFrame:
    """Beräknar OMXSPI:s historiska avkastningar."""

    frame = market.copy()

    frame = _add_horizon_returns(
        frame,
        date_column="market_date",
        close_column="market_close",
        prefix="market_return",
        horizons=RELATIVE_HORIZONS,
    )

    frame = frame.rename(
        columns={
            "market_date": "price_date"
        }
    )

    return frame[
        [
            "price_date",
            *[
                f"market_return_{horizon}d"
                for horizon in RELATIVE_HORIZONS
            ],
        ]
    ]


def add_relative_features(
    result: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    market_path: Path = MARKET_PATH,
    sector_map_path: Path = SECTOR_MAP_PATH,
) -> pd.DataFrame:
    """
    Lägger till marknads- och sektorsrelativa features.

    För varje FI/prisobservation:

        market_return_Nd
            = OMXSPI:s N-dagarsavkastning

        sector_return_Nd
            = median av sektorbolagens N-dagarsavkastning

        price_return_Nd_relative_market
            = aktiens N-dagarsavkastning
              - OMXSPI:s N-dagarsavkastning

        price_return_Nd_relative_sector
            = aktiens N-dagarsavkastning
              - sektorns N-dagarsavkastning

    Alla avkastningar är bakåtblickande.
    """

    result = result.copy()

    market = load_market(
        market_path
    )

    sector_map = load_sector_map(
        sector_map_path
    )

    market_returns = _build_market_returns(
        market
    )

    sector_returns = _build_sector_returns(
        prices,
        sector_map,
    )

    result["sector"] = result[
        "yahoo_symbol"
    ].map(
        sector_map
    )

    result["price_date"] = pd.to_datetime(
        result["price_date"],
        errors="coerce",
    )

    result = result.merge(
        market_returns,
        on="price_date",
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        sector_returns,
        on=[
            "price_date",
            "sector",
        ],
        how="left",
        validate="many_to_one",
    )

    for horizon in RELATIVE_HORIZONS:
        stock_column = (
            f"price_return_{horizon}d"
        )

        market_column = (
            f"market_return_{horizon}d"
        )

        sector_column = (
            f"sector_return_{horizon}d"
        )

        result[
            f"price_return_{horizon}d_relative_market"
        ] = (
            pd.to_numeric(
                result[stock_column],
                errors="coerce",
            )
            - pd.to_numeric(
                result[market_column],
                errors="coerce",
            )
        )

        result[
            f"price_return_{horizon}d_relative_sector"
        ] = (
            pd.to_numeric(
                result[stock_column],
                errors="coerce",
            )
            - pd.to_numeric(
                result[sector_column],
                errors="coerce",
            )
        )

    return result
