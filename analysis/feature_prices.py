"""Inläsning och matchning av prisdata."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.feature_config import (
    PRICE_REQUIRED_COLUMNS,
)
from analysis.feature_utils import (
    normalize_text,
    security_key,
)
from shared.instrument_identity import (
    load_identity,
)


SEVERITY_HORIZON = 5


def find_price_files(
    price_dir: Path,
) -> list[Path]:
    files = sorted(
        price_dir.glob(
            "prices_*.jsonl"
        )
    )

    if not files:
        raise FileNotFoundError(
            "Ingen prices_*.jsonl "
            f"hittades i {price_dir}"
        )

    return files


def find_price_file(
    price_dir: Path,
) -> Path:
    """
    Bakåtkompatibel wrapper.

    Returnerar den äldsta prisfilen eftersom den
    normalt innehåller starten på den historiska serien.

    Nya featurebyggen ska använda find_price_files().
    """
    files = find_price_files(
        price_dir
    )

    return files[0]


def _load_price_file(
    path: Path,
) -> pd.DataFrame:
    frame = pd.read_json(
        path,
        lines=True,
    )

    missing = (
        PRICE_REQUIRED_COLUMNS
        .difference(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            "Prisdata saknar kolumner: "
            + ", ".join(
                sorted(missing)
            )
        )

    return frame


def load_prices(
    paths: Path | list[Path],
) -> pd.DataFrame:
    """
    Läs in en eller flera prisfiler.

    Alla filer slås ihop och dubbla
    (yahoo_symbol, date)-observationer tas bort.

    Den här funktionen gör att featurebygget kan
    använda den kompletta lokala prisarkivet.
    """

    if isinstance(
        paths,
        Path,
    ):
        paths = [paths]

    if not paths:
        raise FileNotFoundError(
            "Ingen prisfil angavs."
        )

    frames: list[
        pd.DataFrame
    ] = []

    for path in paths:
        frame = _load_price_file(
            path
        )

        if not frame.empty:
            frames.append(
                frame
            )

    if not frames:
        raise ValueError(
            "Alla prisfiler var tomma."
        )

    frame = pd.concat(
        frames,
        ignore_index=True,
    )

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    frame["close"] = pd.to_numeric(
        frame["close"],
        errors="coerce",
    )

    frame["isin"] = frame[
        "isin"
    ].where(
        frame["isin"].notna(),
        None,
    )

    frame["issuer"] = (
        frame["issuer"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    frame["yahoo_symbol"] = (
        frame["yahoo_symbol"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    frame = frame.loc[
        frame["date"].notna()
        & frame["close"].notna()
        & np.isfinite(
            frame["close"]
        )
        & (frame["close"] > 0)
    ].copy()

    # Samma Yahoo-symbol + datum får bara förekomma en gång.
    # Keep="last" gör att nyare prisfiler vinner vid eventuell
    # överlappning.
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

    frame["security_key"] = [
        security_key(
            isin,
            issuer,
        )
        for isin, issuer in zip(
            frame["isin"],
            frame["issuer"],
        )
    ]

    return (
        frame.sort_values(
            [
                "security_key",
                "date",
                "yahoo_symbol",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _resolve_price_entity_ids(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Kopplar varje prisobservation till Grunddatas entity_id.

    Identiteten löses mot prisobservationens eget datum. Det är
    viktigt eftersom samma entity kan ha haft olika ISIN över tid.

    Prisdata ändras inte semantiskt här. entity_id läggs endast till
    som ett internt identitetsfält för matchningen mot FI-data.
    """

    if prices.empty:
        result = prices.copy()

        if "entity_id" not in result.columns:
            result["entity_id"] = None

        return result

    identity = load_identity()

    entity_ids: list[str | None] = []

    for row in prices.itertuples(
        index=False
    ):
        observation_date = row.date

        if pd.isna(
            observation_date
        ):
            entity_ids.append(
                None
            )
            continue

        target_date = (
            observation_date.date()
            if hasattr(
                observation_date,
                "date",
            )
            else observation_date
        )

        match = identity.resolve(
            isin=getattr(
                row,
                "isin",
                None,
            ),
            lei=getattr(
                row,
                "lei",
                None,
            ),
            issuer=getattr(
                row,
                "issuer",
                None,
            ),
            yahoo_symbol=getattr(
                row,
                "yahoo_symbol",
                None,
            ),
            target_date=target_date,
        )

        entity_ids.append(
            match.entity_id
            if match is not None
            else None
        )

    result = prices.copy()

    result[
        "entity_id"
    ] = entity_ids

    return result


def _build_entity_price_lookup(
    prices: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Bygger en entity-baserad prislookup.

    Endast prisobservationer som Grunddata säkert kan koppla till
    en entity används här.

    Alla historiska prisobservationer för samma entity hamnar i
    samma serie, även när ISIN har bytts över tid.
    """

    if (
        "entity_id"
        not in prices.columns
    ):
        return {}

    matched = prices.loc[
        prices["entity_id"].notna()
        & (
            prices["entity_id"]
            .astype(str)
            .str.strip()
            != ""
        )
    ].copy()

    if matched.empty:
        return {}

    return {
        key: group.sort_values(
            "date",
            kind="mergesort",
        ).reset_index(
            drop=True
        )
        for key, group in matched.groupby(
            "entity_id",
            sort=False,
        )
    }


def build_price_lookup(
    prices: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        key: group.sort_values(
            "date",
            kind="mergesort",
        ).reset_index(
            drop=True
        )
        for key, group in prices.groupby(
            "security_key",
            sort=False,
        )
    }


def _resolve_fi_entity_id(
    row: Any,
) -> str | None:
    """
    Returnerar entity_id från en FI-rad.

    feature_fi.py har redan löst entity_id. Funktionen finns ändå
    här som defensiv fallback för äldre/anropade DataFrames.
    """

    entity_id = getattr(
        row,
        "entity_id",
        None,
    )

    if (
        entity_id is not None
        and str(entity_id).strip()
    ):
        return str(
            entity_id
        ).strip()

    snapshot_date = getattr(
        row,
        "snapshot_date",
        None,
    )

    if snapshot_date is None or pd.isna(
        snapshot_date
    ):
        return None

    target_date = (
        snapshot_date.date()
        if hasattr(
            snapshot_date,
            "date",
        )
        else snapshot_date
    )

    identity = load_identity()

    match = identity.resolve(
        isin=getattr(
            row,
            "isin",
            None,
        ),
        lei=getattr(
            row,
            "lei",
            None,
        ),
        issuer=getattr(
            row,
            "issuer",
            None,
        ),
        target_date=target_date,
    )

    if match is None:
        return None

    return match.entity_id


def _select_entity_price_series(
    row: Any,
    entity_lookup: dict[str, pd.DataFrame],
) -> pd.DataFrame | None:
    """
    Hittar prisserien för FI-radens entity.

    Entity-matchning används endast efter exakt security_key-matchning
    har misslyckats. Därmed behåller vi den befintliga deterministiska
    instrumentmatchningen när den fungerar.

    Vid ISIN-byte kan entity-matchningen däremot länka den nya FI-raden
    till den historiska Yahoo-serien för samma entity.
    """

    entity_id = _resolve_fi_entity_id(
        row
    )

    if not entity_id:
        return None

    series = entity_lookup.get(
        entity_id
    )

    if (
        series is None
        or series.empty
    ):
        return None

    return series


def add_price_history_features(
    result: dict[str, Any],
    series: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
) -> None:
    """
    Beräknar prisfeatures enbart från observationer
    på eller före signal-dagen.

    Ingen framtida prisinformation används.
    """

    closes = pd.to_numeric(
        series["close"],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    def previous_return(
        days: int,
    ) -> float:
        previous_idx = (
            entry_idx - days
        )

        if previous_idx < 0:
            return np.nan

        previous_price = closes[
            previous_idx
        ]

        if (
            not np.isfinite(
                previous_price
            )
            or previous_price <= 0
        ):
            return np.nan

        return (
            entry_price
            / previous_price
            - 1.0
        )

    result[
        "price_return_5d"
    ] = previous_return(5)

    result[
        "price_return_20d"
    ] = previous_return(20)

    result[
        "price_return_60d"
    ] = previous_return(60)

    start_20 = max(
        0,
        entry_idx - 20,
    )

    history_20 = closes[
        start_20:entry_idx + 1
    ]

    if len(history_20) >= 2:
        result[
            "price_volatility_20d"
        ] = float(
            pd.Series(
                history_20
            )
            .pct_change()
            .std()
        )
    else:
        result[
            "price_volatility_20d"
        ] = np.nan

    high_20 = (
        np.nanmax(
            history_20
        )
        if len(history_20)
        else np.nan
    )

    result[
        "price_distance_from_20d_high"
    ] = (
        entry_price / high_20 - 1.0
        if np.isfinite(
            high_20
        )
        and high_20 > 0
        else np.nan
    )

    start_60 = max(
        0,
        entry_idx - 60,
    )

    history_60 = closes[
        start_60:entry_idx + 1
    ]

    high_60 = (
        np.nanmax(
            history_60
        )
        if len(history_60)
        else np.nan
    )

    result[
        "price_distance_from_60d_high"
    ] = (
        entry_price / high_60 - 1.0
        if np.isfinite(
            high_60
        )
        and high_60 > 0
        else np.nan
    )


def add_severity_targets(
    result: dict[str, Any],
    series: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
) -> None:
    """
    Beräknar extrema femdagarsavkastningar efter entry.

    min_return_5d = lägsta avkastningen under
    handelsdag 1..5 efter entry.

    max_return_5d = högsta avkastningen under
    handelsdag 1..5 efter entry.

    Dessa kolumner är targets och får därför inte användas
    som ML-features.
    """

    severity_end_idx = (
        entry_idx
        + SEVERITY_HORIZON
    )

    if severity_end_idx >= len(series):
        result["min_return_5d"] = np.nan
        result["max_return_5d"] = np.nan
        result["min_return_5d_date"] = pd.NaT
        result["max_return_5d_date"] = pd.NaT
        return

    severity_window = series.iloc[
        entry_idx
        + 1 : severity_end_idx
        + 1
    ]

    severity_returns = (
        pd.to_numeric(
            severity_window["close"],
            errors="coerce",
        )
        .to_numpy(dtype=float)
        / entry_price
        - 1.0
    )

    valid = np.isfinite(
        severity_returns
    )

    if not valid.all():
        result["min_return_5d"] = np.nan
        result["max_return_5d"] = np.nan
        result["min_return_5d_date"] = pd.NaT
        result["max_return_5d_date"] = pd.NaT
        return

    min_idx = int(
        np.argmin(
            severity_returns
        )
    )

    max_idx = int(
        np.argmax(
            severity_returns
        )
    )

    result["min_return_5d"] = float(
        severity_returns[min_idx]
    )

    result["max_return_5d"] = float(
        severity_returns[max_idx]
    )

    result[
        "min_return_5d_date"
    ] = severity_window.iloc[
        min_idx
    ]["date"]

    result[
        "max_return_5d_date"
    ] = severity_window.iloc[
        max_idx
    ]["date"]


def _add_unmatched_price_fields(
    result: dict[str, Any],
) -> None:
    """
    Markerar en FI-observation som saknar användbar prismatchning.

    FI-observationen behålls i datasetet. Prisrelaterade features
    och targets lämnas som NaN/None och kan därför filtreras bort
    naturligt när ett ML-experiment kräver dessa data.
    """

    result[
        "price_date"
    ] = pd.NaT

    result[
        "close"
    ] = np.nan

    result[
        "close_on_signal_date"
    ] = np.nan

    result[
        "days_from_fi_to_price"
    ] = np.nan

    result[
        "price_match_available"
    ] = False

    result[
        "yahoo_symbol"
    ] = None

    result[
        "price_mapping_source"
    ] = "unmatched"

    result[
        "price_return_5d"
    ] = np.nan

    result[
        "price_return_20d"
    ] = np.nan

    result[
        "price_return_60d"
    ] = np.nan

    result[
        "price_volatility_20d"
    ] = np.nan

    result[
        "price_distance_from_20d_high"
    ] = np.nan

    result[
        "price_distance_from_60d_high"
    ] = np.nan

    result[
        "min_return_5d"
    ] = np.nan

    result[
        "max_return_5d"
    ] = np.nan

    result[
        "min_return_5d_date"
    ] = pd.NaT

    result[
        "max_return_5d_date"
    ] = pd.NaT


def attach_prices(
    fi: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
]:
    """
    Matchar FI-observationer mot prisdata.

    Viktig semantik:

    Alla FI-observationer behålls även om prisdata saknas.

    Matchade observationer får:
        price_match_available = True

    Omatchade observationer får:
        price_match_available = False
        price_mapping_source = "unmatched"

    Pris matchas i följande ordning:

    1. Exakt security_key.
    2. Entity via Grunddatas identity layer.

    Entity-matchningen gör det möjligt att använda samma Yahoo-serie
    över ett historiskt ISIN-byte, exempelvis Sinch.

    För omatchade observationer är prisfeatures,
    forward returns och severity-targets NaN.
    """

    lookup = build_price_lookup(
        prices
    )

    prices_with_identity = (
        _resolve_price_entity_ids(
            prices
        )
    )

    entity_lookup = (
        _build_entity_price_lookup(
            prices_with_identity
        )
    )

    rows: list[
        dict[str, Any]
    ] = []

    stats = {
        "fi_rows": int(
            len(fi)
        ),
        "matched_rows": 0,
        "unmatched_rows": 0,
        "matched_by_isin": 0,
        "matched_by_issuer": 0,
        "matched_by_entity": 0,
    }

    for row in fi.itertuples(
        index=False
    ):
        mapping_source = None

        series = lookup.get(
            row.security_key
        )

        if (
            series is not None
            and not series.empty
        ):
            mapping_source = (
                "isin"
                if normalize_text(
                    row.isin
                )
                else "issuer"
            )

        if (
            series is None
            and not normalize_text(
                row.isin
            )
        ):
            series = lookup.get(
                "ISSUER:"
                + normalize_text(
                    row.issuer
                )
            )

            if (
                series is not None
                and not series.empty
            ):
                mapping_source = (
                    "issuer"
                )

        # -----------------------------------------------------
        # Ny identity-baserad fallback.
        #
        # Detta används framför allt när FI har ett nytt ISIN
        # men prisarkivet historiskt ligger på ett annat ISIN
        # för samma entity.
        # -----------------------------------------------------

        if (
            series is None
            or series.empty
        ):
            series = _select_entity_price_series(
                row,
                entity_lookup,
            )

            if (
                series is not None
                and not series.empty
            ):
                mapping_source = (
                    "entity"
                )

        result = row._asdict()

        # -----------------------------------------------------
        # Ingen prisserie för instrumentet/entityn.
        #
        # Behåll FI-raden. Prisrelaterade fält markeras som
        # saknade i stället för att observationen försvinner.
        # -----------------------------------------------------

        if (
            series is None
            or series.empty
        ):
            stats[
                "unmatched_rows"
            ] += 1

            _add_unmatched_price_fields(
                result
            )

            rows.append(
                result
            )

            continue

        dates = series[
            "date"
        ].to_numpy(
            dtype="datetime64[ns]"
        )

        snapshot = np.datetime64(
            row.snapshot_date.to_datetime64(),
            "ns",
        )

        entry_idx = int(
            np.searchsorted(
                dates,
                snapshot,
                side="left",
            )
        )

        # FI-datumet ligger efter sista tillgängliga
        # prisobservation. Även denna FI-rad behålls.
        if entry_idx >= len(
            series
        ):
            stats[
                "unmatched_rows"
            ] += 1

            _add_unmatched_price_fields(
                result
            )

            rows.append(
                result
            )

            continue

        stats[
            "matched_rows"
        ] += 1

        if (
            mapping_source
            == "isin"
        ):
            stats[
                "matched_by_isin"
            ] += 1

        elif (
            mapping_source
            == "issuer"
        ):
            stats[
                "matched_by_issuer"
            ] += 1

        elif (
            mapping_source
            == "entity"
        ):
            stats[
                "matched_by_entity"
            ] += 1

        entry = series.iloc[
            entry_idx
        ]

        entry_price = float(
            entry["close"]
        )

        result[
            "price_date"
        ] = entry["date"]

        result[
            "close"
        ] = entry_price

        result[
            "close_on_signal_date"
        ] = entry_price

        result[
            "days_from_fi_to_price"
        ] = (
            entry["date"]
            - row.snapshot_date
        ).days

        result[
            "price_match_available"
        ] = True

        result[
            "yahoo_symbol"
        ] = entry[
            "yahoo_symbol"
        ]

        result[
            "price_mapping_source"
        ] = mapping_source

        add_price_history_features(
            result,
            series,
            entry_idx,
            entry_price,
        )

        add_severity_targets(
            result,
            series,
            entry_idx,
            entry_price,
        )

        rows.append(
            result
        )

    if not rows:
        return (
            fi.iloc[0:0].copy(),
            stats,
        )

    return (
        pd.DataFrame(rows),
        stats,
    )
