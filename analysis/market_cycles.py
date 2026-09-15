"""
Marknadsrelativ avkastning och cykliska blankningsmönster.

Detta analyssteg gör två saker:

1. Jämför aktiernas framtida avkastning med OMXSPI.
2. Identifierar återkommande lokala toppar och dalar i
   short interest för enskilda bolag.

Marknadsdata hämtas via Yahoo Finance med yfinance.

Yahoo-symbol:
    ^OMXSPI

Marknadsdata persisteras lokalt som JSONL.

Arkitektur:

    Yahoo Finance / yfinance / ^OMXSPI
        |
        v
    data/raw/market/omxspi.jsonl
        |
        v
    market_cycles.py
        |
        +--> market_adjusted_returns.jsonl
        |
        +--> short_cycles.jsonl
        |
        +--> short_cycles_summary.json

Marknadsdata är diagnostik/analysdata och används ännu inte
som ML-feature.

Yahoo/yfinance används här eftersom den tidigare direkta
Yahoo HTTP-klientlösningen gav HTTP 429 i GitHub Actions.

Viktigt:
    Vi accepterar inte ett tomt eller felaktigt svar som giltig
    marknadsdata. Data valideras innan den sparas.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


FEATURES_PATH = Path(
    "data/processed/analysis/features.jsonl"
)

MARKET_RAW_PATH = Path(
    "data/raw/market/omxspi.jsonl"
)

MARKET_ADJUSTED_PATH = Path(
    "data/processed/analysis/market_adjusted_returns.jsonl"
)

SHORT_CYCLES_PATH = Path(
    "data/processed/analysis/short_cycles.jsonl"
)

SHORT_CYCLES_SUMMARY_PATH = Path(
    "data/processed/analysis/short_cycles_summary.json"
)


MARKET_SYMBOL = "OMXSPI"

MARKET_SOURCE = "YAHOO_FINANCE"

MARKET_SOURCE_SERIES = "^OMXSPI"


RETURN_HORIZONS = (
    5,
    20,
    60,
)


MIN_CYCLE_PROMINENCE = 0.25

MIN_DAYS_BETWEEN_CYCLE_EVENTS = 30

CYCLE_LOOKBACK_DAYS = 180


YAHOO_START_PADDING_DAYS = 10

YAHOO_DOWNLOAD_TIMEOUT_SECONDS = 60


def _parse_number(
    value: Any,
) -> float:
    """
    Konvertera ett numeriskt värde till float.
    """

    if value is None:
        return math.nan

    if isinstance(
        value,
        (int, float),
    ):
        return float(value)

    text = str(value).strip()

    if not text:
        return math.nan

    text = (
        text
        .replace("\xa0", "")
        .replace(" ", "")
        .replace("%", "")
    )

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = (
                text
                .replace(".", "")
                .replace(",", ".")
            )
        else:
            text = text.replace(",", "")

    elif "," in text:
        text = text.replace(",", ".")

    try:
        return float(text)

    except ValueError:
        return math.nan


def _json_value(
    value: Any,
) -> Any:
    """
    Gör ett värde säkert för JSON.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            np.integer,
            np.int64,
            np.int32,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            np.float64,
            np.float32,
        ),
    ):
        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    if isinstance(
        value,
        (
            pd.Timestamp,
            np.datetime64,
        ),
    ):
        return pd.Timestamp(
            value
        ).strftime(
            "%Y-%m-%d"
        )

    if pd.isna(value):
        return None

    return value


def load_features() -> pd.DataFrame:
    """
    Läs befintliga Blankdiss-features.
    """

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Featurefil saknas: "
            f"{FEATURES_PATH}"
        )

    frame = pd.read_json(
        FEATURES_PATH,
        lines=True,
    )

    if frame.empty:
        raise RuntimeError(
            "Featurefilen är tom."
        )

    required_columns = {
        "security_key",
        "snapshot_date",
        "price_date",
        "short_interest_pct",
    }

    missing = (
        required_columns
        - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            "Features saknar obligatoriska "
            f"kolumner: {sorted(missing)}"
        )

    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    frame["price_date"] = pd.to_datetime(
        frame["price_date"],
        errors="coerce",
    )

    frame["short_interest_pct"] = pd.to_numeric(
        frame["short_interest_pct"],
        errors="coerce",
    )

    for horizon in RETURN_HORIZONS:
        column = (
            f"forward_return_{horizon}d"
        )

        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    frame = frame.dropna(
        subset=[
            "security_key",
            "snapshot_date",
        ]
    ).copy()

    return frame


def _validate_market_frame(
    market: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    """
    Strikt validering av marknadsdata.

    Vi accepterar inte:
        - tom data
        - ogiltiga datum
        - NaN
        - oändliga värden
        - negativa/noll priser
        - dubbletter per handelsdag
    """

    if market is None or market.empty:
        raise RuntimeError(
            f"{source} returnerade ingen "
            "marknadsdata."
        )

    required_columns = {
        "market_date",
        "market_close",
    }

    missing = (
        required_columns
        - set(market.columns)
    )

    if missing:
        raise RuntimeError(
            f"{source} saknar obligatoriska "
            f"kolumner: {sorted(missing)}"
        )

    result = market.copy()

    result["market_date"] = pd.to_datetime(
        result["market_date"],
        errors="coerce",
    )

    result["market_close"] = pd.to_numeric(
        result["market_close"],
        errors="coerce",
    )

    invalid_dates = int(
        result["market_date"].isna().sum()
    )

    invalid_prices = int(
        result["market_close"].isna().sum()
    )

    if invalid_dates:
        raise RuntimeError(
            f"{source} innehåller "
            f"{invalid_dates} ogiltiga datum."
        )

    if invalid_prices:
        raise RuntimeError(
            f"{source} innehåller "
            f"{invalid_prices} ogiltiga "
            "indexvärden."
        )

    nonfinite = ~np.isfinite(
        result["market_close"].to_numpy(
            dtype=float
        )
    )

    if nonfinite.any():
        raise RuntimeError(
            f"{source} innehåller "
            f"{int(nonfinite.sum())} "
            "icke-finit(a) indexvärden."
        )

    nonpositive = (
        result["market_close"] <= 0
    )

    if nonpositive.any():
        raise RuntimeError(
            f"{source} innehåller "
            f"{int(nonpositive.sum())} "
            "icke-positiva indexvärden."
        )

    duplicate_dates = int(
        result["market_date"]
        .duplicated()
        .sum()
    )

    if duplicate_dates:
        raise RuntimeError(
            f"{source} innehåller "
            f"{duplicate_dates} dubbletter "
            "av handelsdagar."
        )

    result = (
        result
        .sort_values(
            "market_date"
        )
        .reset_index(
            drop=True
        )
    )

    return result


def _normalize_yahoo_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalisera yfinance-kolumner.

    yfinance kan returnera exempelvis:

        Close
        High
        Low
        Open
        Volume

    eller MultiIndex:

        ('Close', '^OMXSPI')
        ('High', '^OMXSPI')
        ...

    Vi reducerar detta till ett enkelt DataFrame.
    """

    result = frame.copy()

    if isinstance(
        result.columns,
        pd.MultiIndex,
    ):
        close_candidates = []

        for column in result.columns:
            parts = [
                str(part)
                for part in column
            ]

            if any(
                part.lower() == "close"
                for part in parts
            ):
                close_candidates.append(
                    column
                )

        if not close_candidates:
            raise RuntimeError(
                "Yahoo/yfinance returnerade "
                "MultiIndex-data men ingen "
                "Close-kolumn kunde hittas."
            )

        close_column = (
            close_candidates[0]
        )

        close = result[
            close_column
        ]

        if isinstance(
            close,
            pd.DataFrame,
        ):
            if close.shape[1] != 1:
                raise RuntimeError(
                    "Yahoo/yfinance returnerade "
                    "flera Close-kolumner för "
                    f"{MARKET_SOURCE_SERIES}."
                )

            close = close.iloc[
                :,
                0
            ]

    else:

        normalized_columns = {
            str(column).strip().lower(): column
            for column in result.columns
        }

        close_column = (
            normalized_columns.get(
                "close"
            )
        )

        if close_column is None:
            raise RuntimeError(
                "Yahoo/yfinance returnerade "
                "ingen Close-kolumn.\n"
                f"Kolumner: {list(result.columns)}"
            )

        close = result[
            close_column
        ]

    output = pd.DataFrame(
        {
            "market_date": result.index,
            "market_close": close,
        }
    )

    return output


def _download_yahoo(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Hämta OMXSPI via yfinance.

    Vi använder download() direkt eftersom det
    ger en enkel DataFrame och inte kräver att
    vi håller ett Ticker-objekt vid liv.

    end-datumet görs exklusivt genom yfinance,
    därför lägger vi till en extra kalenderdag.
    """

    requested_start = (
        pd.Timestamp(start_date)
        - pd.Timedelta(
            days=YAHOO_START_PADDING_DAYS
        )
    )

    requested_end = (
        pd.Timestamp(end_date)
        + pd.Timedelta(
            days=YAHOO_START_PADDING_DAYS
        )
    )

    yahoo_end = (
        requested_end
        + pd.Timedelta(
            days=1
        )
    )

    print(
        "Laddar OMXSPI via Yahoo Finance."
    )

    print(
        "Yahoo-symbol: "
        f"{MARKET_SOURCE_SERIES}"
    )

    print(
        "Yahoo-intervall: "
        f"{requested_start.date()}"
        " -> "
        f"{requested_end.date()}"
    )

    try:
        raw = yf.download(
            MARKET_SOURCE_SERIES,
            start=requested_start.strftime(
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
            timeout=YAHOO_DOWNLOAD_TIMEOUT_SECONDS,
        )

    except Exception as error:
        raise RuntimeError(
            "Yahoo/yfinance kunde inte "
            "hämta OMXSPI: "
            f"{error}"
        ) from error

    if raw is None:
        raise RuntimeError(
            "Yahoo/yfinance returnerade None."
        )

    print(
        "Yahoo rå-data:"
    )

    print(
        f"  Rader: {len(raw):,}"
    )

    print(
        f"  Kolumner: {list(raw.columns)}"
    )

    if raw.empty:
        raise RuntimeError(
            "Yahoo/yfinance returnerade "
            "en tom DataFrame för "
            f"{MARKET_SOURCE_SERIES}."
        )

    market = _normalize_yahoo_columns(
        raw
    )

    market["market_date"] = pd.to_datetime(
        market["market_date"],
        errors="coerce",
    ).tz_localize(
        None
    )

    market["market_close"] = pd.to_numeric(
        market["market_close"],
        errors="coerce",
    )

    market = market[
        (
            market["market_date"]
            >= requested_start
        )
        & (
            market["market_date"]
            <= requested_end
        )
    ].copy()

    market = _validate_market_frame(
        market,
        "Yahoo/yfinance",
    )

    print(
        "Yahoo-parsering:"
    )

    print(
        "  Giltiga observationer: "
        f"{len(market):,}"
    )

    print(
        "  Datum: "
        f"{market['market_date'].min().date()}"
        " -> "
        f"{market['market_date'].max().date()}"
    )

    print(
        "  Första värde: "
        f"{market.iloc[0]['market_close']}"
    )

    print(
        "  Sista värde: "
        f"{market.iloc[-1]['market_close']}"
    )

    print(
        "Senaste fem OMXSPI-observationer:"
    )

    print(
        market.tail(5).to_string(
            index=False
        )
    )

    return market


def load_raw_market_data() -> pd.DataFrame:
    """
    Läs redan sparad OMXSPI-historik.
    """

    if not MARKET_RAW_PATH.exists():
        return pd.DataFrame(
            columns=[
                "market_date",
                "market_close",
            ]
        )

    rows = []

    with MARKET_RAW_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line in handle:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(
                    line
                )

            except json.JSONDecodeError:
                continue

            rows.append(
                record
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "market_date",
                "market_close",
            ]
        )

    frame = pd.DataFrame(
        rows
    )

    if (
        "market_date"
        not in frame.columns
        or
        "market_close"
        not in frame.columns
    ):
        return pd.DataFrame(
            columns=[
                "market_date",
                "market_close",
            ]
        )

    frame["market_date"] = pd.to_datetime(
        frame["market_date"],
        errors="coerce",
    )

    frame["market_close"] = pd.to_numeric(
        frame["market_close"],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            "market_date",
            "market_close",
        ]
    )

    frame = frame[
        np.isfinite(
            frame["market_close"]
        )
    ]

    frame = frame[
        frame["market_close"] > 0
    ]

    frame = (
        frame
        .drop_duplicates(
            subset=[
                "market_date",
            ]
        )
        .sort_values(
            "market_date"
        )
        .reset_index(
            drop=True
        )
    )

    return frame


def merge_and_persist_market_data(
    existing: pd.DataFrame,
    downloaded: pd.DataFrame,
) -> pd.DataFrame:
    """
    Slå ihop lokal och ny marknadsdata.

    Ny Yahoo-data ersätter eventuella gamla
    observationer för samma handelsdag.
    """

    combined = pd.concat(
        [
            existing,
            downloaded,
        ],
        ignore_index=True,
    )

    if combined.empty:
        raise RuntimeError(
            "Ingen marknadsdata att spara."
        )

    combined["market_date"] = pd.to_datetime(
        combined["market_date"],
        errors="coerce",
    )

    combined["market_close"] = pd.to_numeric(
        combined["market_close"],
        errors="coerce",
    )

    combined = combined.dropna(
        subset=[
            "market_date",
            "market_close",
        ]
    )

    combined = combined[
        np.isfinite(
            combined["market_close"]
        )
    ]

    combined = combined[
        combined["market_close"] > 0
    ]

    combined = (
        combined
        .drop_duplicates(
            subset=[
                "market_date",
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

    combined = _validate_market_frame(
        combined,
        "Sammanslagen OMXSPI-historik",
    )

    MARKET_RAW_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MARKET_RAW_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for record in combined.to_dict(
            orient="records"
        ):

            cleaned = {
                key: _json_value(value)
                for key, value in record.items()
            }

            handle.write(
                json.dumps(
                    cleaned,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        "Sparad OMXSPI-historik: "
        f"{MARKET_RAW_PATH}"
    )

    print(
        "Totalt OMXSPI-rader: "
        f"{len(combined):,}"
    )

    return combined


def add_market_forward_returns(
    market: pd.DataFrame,
) -> pd.DataFrame:
    """
    Beräkna framtida OMXSPI-avkastning
    per handelsdag.
    """

    result = market.copy()

    result = (
        result
        .sort_values(
            "market_date"
        )
        .reset_index(
            drop=True
        )
    )

    prices = (
        result["market_close"]
        .to_numpy(
            dtype=float
        )
    )

    for horizon in RETURN_HORIZONS:

        target_index = (
            np.arange(
                len(result)
            )
            + horizon
        )

        values = np.full(
            len(result),
            np.nan,
            dtype=float,
        )

        valid = (
            target_index
            < len(result)
        )

        if valid.any():
            values[valid] = (
                prices[
                    target_index[valid]
                ]
                / prices[valid]
                - 1.0
            )

        result[
            f"market_forward_return_{horizon}d"
        ] = values

    return result


def align_market_returns(
    features: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """
    Matcha varje aktierad mot första
    OMXSPI-handelsdagen på eller efter
    aktiens price_date.
    """

    frame = features.copy()

    market = (
        market
        .sort_values(
            "market_date"
        )
        .reset_index(
            drop=True
        )
    )

    market_dates = (
        market["market_date"]
        .to_numpy()
    )

    market_lookup = market.set_index(
        "market_date"
    )

    for horizon in RETURN_HORIZONS:

        column = (
            f"market_forward_return_{horizon}d"
        )

        values: list[float] = []

        for price_date in frame[
            "price_date"
        ]:

            if pd.isna(price_date):
                values.append(
                    np.nan
                )
                continue

            target_date = pd.Timestamp(
                price_date
            )

            index = np.searchsorted(
                market_dates,
                target_date,
                side="left",
            )

            if index >= len(
                market_dates
            ):
                values.append(
                    np.nan
                )
                continue

            matched_date = (
                market_dates[index]
            )

            value = market_lookup.loc[
                matched_date,
                column,
            ]

            values.append(
                float(value)
                if pd.notna(value)
                else np.nan
            )

        frame[column] = values

    return frame


def build_market_adjusted_returns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Skapa abnormal return:

        aktiens framtida avkastning
        -
        OMXSPI:s framtida avkastning
    """

    result = frame[
        [
            "security_key",
            "snapshot_date",
            "price_date",
        ]
    ].copy()

    for horizon in RETURN_HORIZONS:

        stock_column = (
            f"forward_return_{horizon}d"
        )

        market_column = (
            f"market_forward_return_{horizon}d"
        )

        abnormal_column = (
            f"abnormal_return_{horizon}d"
        )

        if stock_column not in frame.columns:
            result[
                abnormal_column
            ] = np.nan

            continue

        result[
            stock_column
        ] = frame[
            stock_column
        ]

        result[
            market_column
        ] = frame[
            market_column
        ]

        result[
            abnormal_column
        ] = (
            frame[
                stock_column
            ]
            - frame[
                market_column
            ]
        )

    return result


def detect_local_extrema(
    series: pd.Series,
    dates: pd.Series,
) -> list[dict[str, Any]]:
    """
    Identifiera lokala toppar och dalar
    i short interest.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    date_values = pd.to_datetime(
        dates,
        errors="coerce",
    ).to_numpy()

    events: list[
        dict[str, Any]
    ] = []

    for index in range(
        1,
        len(values) - 1,
    ):

        current = values[
            index
        ]

        if not np.isfinite(
            current
        ):
            continue

        previous = values[
            index - 1
        ]

        following = values[
            index + 1
        ]

        if not (
            np.isfinite(previous)
            and np.isfinite(following)
        ):
            continue

        is_peak = (
            current > previous
            and current >= following
        )

        is_trough = (
            current < previous
            and current <= following
        )

        if not (
            is_peak
            or is_trough
        ):
            continue

        current_date = pd.Timestamp(
            date_values[index]
        )

        if pd.isna(
            current_date
        ):
            continue

        lookback_start = (
            current_date
            - pd.Timedelta(
                days=CYCLE_LOOKBACK_DAYS
            )
        )

        historical_values = []

        for j in range(
            index + 1
        ):

            historical_date = pd.Timestamp(
                date_values[j]
            )

            if pd.isna(
                historical_date
            ):
                continue

            if (
                historical_date
                < lookback_start
            ):
                continue

            value = values[j]

            if np.isfinite(value):
                historical_values.append(
                    value
                )

        if not historical_values:
            continue

        local_minimum = min(
            historical_values
        )

        local_maximum = max(
            historical_values
        )

        if is_peak:

            prominence = (
                current
                - local_minimum
            )

            event_type = "peak"

        else:

            prominence = (
                local_maximum
                - current
            )

            event_type = "trough"

        if (
            prominence
            < MIN_CYCLE_PROMINENCE
        ):
            continue

        if events:

            previous_event_date = pd.Timestamp(
                events[-1][
                    "snapshot_date"
                ]
            )

            days_since_previous = (
                current_date
                - previous_event_date
            ).days

            if (
                days_since_previous
                < MIN_DAYS_BETWEEN_CYCLE_EVENTS
            ):
                continue

        events.append(
            {
                "snapshot_date": (
                    current_date.strftime(
                        "%Y-%m-%d"
                    )
                ),
                "event_type": event_type,
                "short_interest_pct": float(
                    current
                ),
                "prominence_pct": float(
                    prominence
                ),
            }
        )

    return events


def detect_short_cycles(
    features: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Identifiera blankningscykler per bolag.
    """

    cycle_rows: list[
        dict[str, Any]
    ] = []

    summary_rows: list[
        dict[str, Any]
    ] = []

    grouped = (
        features
        .sort_values(
            [
                "security_key",
                "snapshot_date",
            ]
        )
        .groupby(
            "security_key",
            sort=False,
        )
    )

    for security_key, group in grouped:

        group = group.reset_index(
            drop=True
        )

        events = detect_local_extrema(
            group["short_interest_pct"],
            group["snapshot_date"],
        )

        if not events:
            continue

        for event in events:

            cycle_rows.append(
                {
                    "security_key": security_key,
                    **event,
                }
            )

        peaks = sum(
            event["event_type"]
            == "peak"
            for event in events
        )

        troughs = sum(
            event["event_type"]
            == "trough"
            for event in events
        )

        summary_rows.append(
            {
                "security_key": security_key,
                "cycle_events": len(
                    events
                ),
                "peaks": peaks,
                "troughs": troughs,
                "first_event": min(
                    event[
                        "snapshot_date"
                    ]
                    for event in events
                ),
                "last_event": max(
                    event[
                        "snapshot_date"
                    ]
                    for event in events
                ),
            }
        )

    cycles = pd.DataFrame(
        cycle_rows
    )

    summary = pd.DataFrame(
        summary_rows
    )

    if not cycles.empty:

        cycles = (
            cycles
            .sort_values(
                [
                    "security_key",
                    "snapshot_date",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    if not summary.empty:

        summary = (
            summary
            .sort_values(
                [
                    "cycle_events",
                    "security_key",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )

    return cycles, summary


def write_jsonl(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """
    Skriv DataFrame som JSONL.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for record in frame.to_dict(
            orient="records"
        ):

            cleaned = {
                key: _json_value(value)
                for key, value in record.items()
            }

            handle.write(
                json.dumps(
                    cleaned,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_summary(
    summary: pd.DataFrame,
    path: Path,
) -> None:
    """
    Skriv cykelsummering som JSON.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for record in summary.to_dict(
        orient="records"
    ):

        records.append(
            {
                key: _json_value(value)
                for key, value in record.items()
            }
        )

    payload = {
        "market_symbol": MARKET_SYMBOL,
        "market_source": MARKET_SOURCE,
        "market_source_series": (
            MARKET_SOURCE_SERIES
        ),
        "min_cycle_prominence_pct": (
            MIN_CYCLE_PROMINENCE
        ),
        "min_days_between_events": (
            MIN_DAYS_BETWEEN_CYCLE_EVENTS
        ),
        "cycle_lookback_days": (
            CYCLE_LOOKBACK_DAYS
        ),
        "companies_with_cycles": len(
            records
        ),
        "companies": records,
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )


def get_required_market_interval(
    features: pd.DataFrame,
) -> tuple[
    pd.Timestamp,
    pd.Timestamp,
]:
    """
    Bestäm vilket marknadsintervall som krävs
    av featurefilen.
    """

    price_dates = features[
        "price_date"
    ].dropna()

    if price_dates.empty:
        raise RuntimeError(
            "Features innehåller inga "
            "giltiga price_date."
        )

    return (
        pd.Timestamp(
            price_dates.min()
        ).normalize(),
        pd.Timestamp(
            price_dates.max()
        ).normalize(),
    )


def market_covers_required_interval(
    market: pd.DataFrame,
    required_start: pd.Timestamp,
    required_end: pd.Timestamp,
) -> bool:
    """
    Kontrollera om lokal marknadsdata täcker
    hela det intervall som krävs.
    """

    if market.empty:
        return False

    market_start = pd.Timestamp(
        market["market_date"].min()
    ).normalize()

    market_end = pd.Timestamp(
        market["market_date"].max()
    ).normalize()

    return (
        market_start <= required_start
        and market_end >= required_end
    )


def update_market_data(
    existing: pd.DataFrame,
    required_start: pd.Timestamp,
    required_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Uppdatera lokal OMXSPI-data.

    Första körningen:
        hämta hela intervallet från Yahoo.

    Senare körningar:
        komplettera från strax före senaste
        lokala observation till required_end.

    Om lokal data redan täcker intervallet
    används den utan extern hämtning.
    """

    if existing.empty:

        print(
            "Ingen lokal OMXSPI-historik "
            "finns ännu."
        )

        downloaded = _download_yahoo(
            required_start,
            required_end,
        )

        return merge_and_persist_market_data(
            existing,
            downloaded,
        )

    print(
        "Lokal OMXSPI-historik: "
        f"{existing['market_date'].min().date()}"
        " -> "
        f"{existing['market_date'].max().date()}"
    )

    if market_covers_required_interval(
        existing,
        required_start,
        required_end,
    ):

        print(
            "Lokal OMXSPI-historik täcker "
            "hela det nödvändiga intervallet."
        )

        print(
            "Ingen extern hämtning behövs."
        )

        return existing

    existing_end = pd.Timestamp(
        existing["market_date"].max()
    ).normalize()

    download_start = min(
        required_start,
        existing_end
        - pd.Timedelta(
            days=YAHOO_START_PADDING_DAYS
        ),
    )

    download_end = max(
        required_end,
        existing_end,
    )

    print(
        "Lokal historik behöver kompletteras."
    )

    print(
        "Kompletterar: "
        f"{download_start.date()}"
        " -> "
        f"{download_end.date()}"
    )

    downloaded = _download_yahoo(
        download_start,
        download_end,
    )

    return merge_and_persist_market_data(
        existing,
        downloaded,
    )


def main() -> None:
    print(
        "=========================================="
    )

    print(
        "BLANKDISS MARKET & SHORT CYCLE ANALYSIS"
    )

    print(
        "=========================================="
    )

    print(
        "Marknadskälla: Yahoo Finance"
    )

    print(
        "Yahoo-symbol: ^OMXSPI"
    )

    features = load_features()

    print(
        "Feature-rader: "
        f"{len(features):,}"
    )

    print(
        "Datum: "
        f"{features['snapshot_date'].min().date()}"
        " -> "
        f"{features['snapshot_date'].max().date()}"
    )

    required_start, required_end = (
        get_required_market_interval(
            features
        )
    )

    print(
        "Krävd marknadsperiod: "
        f"{required_start.date()}"
        " -> "
        f"{required_end.date()}"
    )

    existing_market = (
        load_raw_market_data()
    )

    market = update_market_data(
        existing_market,
        required_start,
        required_end,
    )

    market = add_market_forward_returns(
        market
    )

    aligned = align_market_returns(
        features,
        market,
    )

    market_adjusted = (
        build_market_adjusted_returns(
            aligned
        )
    )

    cycles, cycle_summary = (
        detect_short_cycles(
            features
        )
    )

    write_jsonl(
        market_adjusted,
        MARKET_ADJUSTED_PATH,
    )

    write_jsonl(
        cycles,
        SHORT_CYCLES_PATH,
    )

    write_summary(
        cycle_summary,
        SHORT_CYCLES_SUMMARY_PATH,
    )

    companies_with_cycles = len(
        cycle_summary
    )

    print(
        "=========================================="
    )

    print(
        "MARKET ANALYS KLAR"
    )

    print(
        "=========================================="
    )

    print(
        "Market-adjusted rows: "
        f"{len(market_adjusted):,}"
    )

    print(
        "Cycle events: "
        f"{len(cycles):,}"
    )

    print(
        "Bolag med cykler: "
        f"{companies_with_cycles:,}"
    )

    print(
        "OMXSPI-period: "
        f"{market['market_date'].min().date()}"
        " -> "
        f"{market['market_date'].max().date()}"
    )

    print(
        "OMXSPI-rader: "
        f"{len(market):,}"
    )

    print(
        "Output:"
    )

    print(
        f"  {MARKET_RAW_PATH}"
    )

    print(
        f"  {MARKET_ADJUSTED_PATH}"
    )

    print(
        f"  {SHORT_CYCLES_PATH}"
    )

    print(
        f"  {SHORT_CYCLES_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
