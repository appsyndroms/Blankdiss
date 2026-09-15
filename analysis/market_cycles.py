"""
Marknadsrelativ avkastning och cykliska blankningsmönster.

Detta är ett diagnostiskt analyssteg ovanpå de befintliga
Blankdiss-featuresen.

Det gör två saker:

1. Jämför aktiernas framtida avkastning med OMXSPI.
2. Identifierar återkommande lokala toppar och dalar i
   short interest för enskilda bolag.

Resultaten används ännu inte som ML-features.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


FEATURES_PATH = Path(
    "data/processed/analysis/features.jsonl"
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

NASDAQ_HISTORICAL_URL = (
    "https://api.nasdaq.com/api/quote/"
    "OMXSPI/historical"
)

NASDAQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/131.0 Safari/537.36"
    ),
    "Accept": (
        "application/json, text/plain, */*"
    ),
    "Accept-Language": "sv-SE,sv;q=0.9,en;q=0.8",
    "Referer": (
        "https://indexes.nasdaq.com/"
    ),
}

RETURN_HORIZONS = (
    5,
    20,
    60,
)

MIN_CYCLE_PROMINENCE = 0.25

MIN_DAYS_BETWEEN_CYCLE_EVENTS = 30

CYCLE_LOOKBACK_DAYS = 180


def _parse_nasdaq_number(
    value: Any,
) -> float:
    """
    Konvertera Nasdaq-värden som t.ex.

        '2,345.67'
        '2 345,67'
        '2345.67'

    till float.
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


def load_features() -> pd.DataFrame:
    """
    Läs befintliga features.
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
        column = f"forward_return_{horizon}d"

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


def download_market_data(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Download daily OMXSPI history directly from Nasdaq.

    The requested period is based on the actual stock price
    history available in Blankdiss.

    Nasdaq does not need future dates. In particular, the
    todate parameter must not be placed in the future.

    We therefore:
      - add a small buffer before the first required date
      - cap the end date at today's date
      - keep enough historical observations for the
        60-trading-day forward calculation where data exists

    If Nasdaq returns an unexpected JSON structure, the
    response metadata and a short response preview are printed
    to make the failure diagnosable.
    """

    today = pd.Timestamp.now().normalize()

    requested_start = (
        pd.Timestamp(start_date)
        - pd.Timedelta(days=10)
    )

    requested_end = min(
        pd.Timestamp(end_date)
        + pd.Timedelta(days=120),
        today,
    )

    start_text = requested_start.strftime(
        "%Y-%m-%d"
    )

    end_text = requested_end.strftime(
        "%Y-%m-%d"
    )

    print(
        "Laddar marknadsdata: "
        "Nasdaq/OMXSPI"
    )

    print(
        "Nasdaq-intervall: "
        f"{start_text} -> {end_text}"
    )

    params = {
        "assetclass": "index",
        "fromdate": start_text,
        "todate": end_text,
        "limit": 5000,
    }

    response = None
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            print(
                "Nasdaq-försök "
                f"{attempt}/3..."
            )

            response = requests.get(
                NASDAQ_HISTORICAL_URL,
                params=params,
                headers=NASDAQ_HEADERS,
                timeout=30,
            )

            print(
                "Nasdaq HTTP-status: "
                f"{response.status_code}"
            )

            print(
                "Nasdaq Content-Type: "
                f"{response.headers.get('Content-Type', '')}"
            )

            response.raise_for_status()

            break

        except requests.RequestException as error:
            last_error = error

            print(
                "Nasdaq-anrop misslyckades: "
                f"{error}"
            )

            if attempt < 3:
                print(
                    "Försöker igen..."
                )

                time.sleep(2)

    if response is None:
        raise RuntimeError(
            "Kunde inte hämta "
            "OMXSPI från Nasdaq efter "
            "3 försök."
        ) from last_error

    try:
        payload = response.json()

    except ValueError as error:
        preview = response.text[:1000]

        raise RuntimeError(
            "Nasdaq returnerade ett svar som "
            "inte kunde tolkas som JSON.\n"
            "Svar:\n"
            f"{preview}"
        ) from error

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Nasdaq returnerade ett JSON-svar "
            "som inte är ett objekt.\n"
            f"Svarstyp: {type(payload).__name__}\n"
            f"Svar: {str(payload)[:1000]}"
        )

    data = payload.get("data")

    if not isinstance(data, dict):
        print(
            "Nasdaq-svaret har oväntad struktur."
        )

        print(
            "JSON-nycklar: "
            f"{list(payload.keys())}"
        )

        payload_preview = json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )[:2000]

        print(
            "Nasdaq-svar, början:\n"
            f"{payload_preview}"
        )

        raise RuntimeError(
            "Nasdaq-svaret saknar 'data'. "
            "Se diagnostiken ovan."
        )

    trades_table = data.get(
        "tradesTable"
    )

    if not isinstance(
        trades_table,
        dict,
    ):
        print(
            "Nasdaq 'data' har oväntad "
            "struktur."
        )

        print(
            "data-nycklar: "
            f"{list(data.keys())}"
        )

        raise RuntimeError(
            "Nasdaq-svaret saknar "
            "'data.tradesTable'. "
            "Se diagnostiken ovan."
        )

    rows = trades_table.get(
        "rows"
    )

    if not isinstance(rows, list):
        print(
            "Nasdaq 'tradesTable' har "
            "oväntad struktur."
        )

        print(
            "tradesTable-nycklar: "
            f"{list(trades_table.keys())}"
        )

        raise RuntimeError(
            "Nasdaq-svaret saknar "
            "'data.tradesTable.rows'. "
            "Se diagnostiken ovan."
        )

    total_records = data.get(
        "totalRecords"
    )

    print(
        "Nasdaq returnerade "
        f"{len(rows):,} rader"
        + (
            f" av {total_records:,}"
            if isinstance(
                total_records,
                int,
            )
            else ""
        )
    )

    market_rows: list[
        dict[str, Any]
    ] = []

    for row in rows:
        if not isinstance(
            row,
            dict,
        ):
            continue

        raw_date = row.get(
            "date"
        )

        raw_close = row.get(
            "close"
        )

        if raw_date is None:
            continue

        if raw_close is None:
            continue

        try:
            market_date = pd.to_datetime(
                str(raw_date),
                format="%m/%d/%Y",
                errors="coerce",
            )

            market_close = (
                _parse_nasdaq_number(
                    raw_close
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if pd.isna(market_date):
            continue

        if not np.isfinite(
            market_close
        ):
            continue

        if market_close <= 0:
            continue

        market_rows.append(
            {
                "market_date": market_date,
                "market_close": market_close,
            }
        )

    market = pd.DataFrame(
        market_rows
    )

    if market.empty:
        raise RuntimeError(
            "Nasdaq/OMXSPI innehåller inga "
            "giltiga observationer."
        )

    market = (
        market
        .drop_duplicates(
            subset=[
                "market_date",
            ]
        )
        .sort_values(
            "market_date"
        )
        .reset_index(drop=True)
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
    ].reset_index(drop=True)

    if market.empty:
        raise RuntimeError(
            "Nasdaq/OMXSPI innehåller inga "
            "observationer inom det begärda "
            "intervallet. "
            f"Begärt intervall: "
            f"{requested_start.date()} -> "
            f"{requested_end.date()}"
        )

    print(
        "OMXSPI-period: "
        f"{market['market_date'].min().date()} "
        "-> "
        f"{market['market_date'].max().date()}"
    )

    print(
        "OMXSPI-observationer: "
        f"{len(market):,}"
    )

    return market


def add_market_forward_returns(
    market: pd.DataFrame,
) -> pd.DataFrame:
    """
    Beräkna framtida OMXSPI-avkastning per handelsdag.
    """

    result = market.copy()

    for horizon in RETURN_HORIZONS:
        target_index = (
            np.arange(len(result))
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
                result["market_close"]
                .to_numpy()[target_index[valid]]
                / result["market_close"]
                .to_numpy()[valid]
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
    Matcha varje aktierad mot första OMXSPI-handelsdagen
    på eller efter aktiens price_date.
    """

    frame = features.copy()

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

            target_date = (
                pd.Timestamp(price_date)
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
        minus
        OMXSPI:s framtida avkastning.
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
            result[abnormal_column] = np.nan
            continue

        result[stock_column] = (
            frame[stock_column]
        )

        result[market_column] = (
            frame[market_column]
        )

        result[abnormal_column] = (
            frame[stock_column]
            - frame[market_column]
        )

    return result


def detect_local_extrema(
    series: pd.Series,
    dates: pd.Series,
) -> list[dict[str, Any]]:
    """
    Identifiera lokala toppar och dalar i short interest.

    Ett event måste:
      - ha minst 0.25 procentenheters prominence
      - ligga minst 30 dagar från föregående event
      - ha en lokal referensnivå inom 180 dagar
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
        current = values[index]

        if not np.isfinite(current):
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

        if pd.isna(current_date):
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

    Returnerar:

      1. Alla identifierade toppar/dalar.
      2. Summering per bolag.
    """

    cycle_rows: list[
        dict[str, Any]
    ] = []

    summary_rows: list[
        dict[str, Any]
    ] = []

    grouped = features.sort_values(
        [
            "security_key",
            "snapshot_date",
        ]
    ).groupby(
        "security_key",
        sort=False,
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
        cycles = cycles.sort_values(
            [
                "security_key",
                "snapshot_date",
            ]
        ).reset_index(
            drop=True
        )

    if not summary.empty:
        summary = summary.sort_values(
            [
                "cycle_events",
                "security_key",
            ],
            ascending=[
                False,
                True,
            ],
        ).reset_index(
            drop=True
        )

    return cycles, summary


def _json_value(
    value: Any,
) -> Any:
    """
    Gör värden säkra för JSON.
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
                key: _json_value(
                    value
                )
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

    price_dates = features[
        "price_date"
    ].dropna()

    if price_dates.empty:
        raise RuntimeError(
            "Features innehåller inga "
            "giltiga price_date."
        )

    market_start = price_dates.min()
    market_end = price_dates.max()

    market = download_market_data(
        market_start,
        market_end,
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
        "Output:"
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
