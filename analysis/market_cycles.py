"""
Market-adjusted returns and recurring short-interest cycle analysis.
This module performs two diagnostic analyses:
1. Market-adjusted returns
   - Downloads OMXSPI history directly from Nasdaq
   - Calculates market forward returns for 5/20/60 trading observations
   - Calculates abnormal stock returns relative to OMXSPI
2. Short-interest cycles
   - Detects local short-interest peaks and troughs
   - Requires a minimum local prominence
   - Requires minimum separation between events
   - Measures subsequent stock and abnormal returns
   - Produces per-company cycle summaries
This is intentionally a diagnostic layer.
The resulting market-adjusted and cycle variables are not yet
fed into the ML models.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import requests
FEATURES_PATH = Path(
    "data/processed/analysis/features.jsonl"
)
OUTPUT_DIR = Path(
    "data/processed/analysis"
)
MARKET_RETURNS_PATH = (
    OUTPUT_DIR
    / "market_adjusted_returns.jsonl"
)
CYCLES_PATH = (
    OUTPUT_DIR
    / "short_cycles.jsonl"
)
CYCLE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "short_cycles_summary.json"
)
MARKET_SYMBOL = "OMXSPI"
NASDAQ_HISTORICAL_URL = (
    "https://api.nasdaq.com/api/quote/"
    f"{MARKET_SYMBOL}/historical"
)
HORIZONS = (
    5,
    20,
    60,
)
MIN_PROMINENCE_PP = 0.25
MIN_EVENT_SEPARATION_DAYS = 30
PROMINENCE_WINDOW_DAYS = 180
NASDAQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(X11; Linux x86_64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/131.0.0.0 "
        "Safari/537.36"
    ),
    "Accept": (
        "application/json, text/plain, */*"
    ),
    "Accept-Language": (
        "en-US,en;q=0.9"
    ),
    "Referer": (
        "https://www.nasdaq.com/"
    ),
    "Origin": (
        "https://www.nasdaq.com"
    ),
}
def load_features() -> pd.DataFrame:
    """Load the existing feature dataset."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Featurefil saknas: {FEATURES_PATH}"
        )
    rows: list[dict[str, Any]] = []
    with FEATURES_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(
                json.loads(line)
            )
    if not rows:
        raise ValueError(
            "features.jsonl är tom."
        )
    frame = pd.DataFrame(rows)
    required = {
        "security_key",
        "snapshot_date",
        "price_date",
        "short_interest_pct",
        "forward_return_5d",
        "forward_return_20d",
        "forward_return_60d",
    }
    missing = sorted(
        required.difference(
            frame.columns
        )
    )
    if missing:
        raise ValueError(
            "Följande kolumner saknas i "
            "feature-datasetet: "
            + ", ".join(missing)
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
    for horizon in HORIZONS:
        column = (
            f"forward_return_{horizon}d"
        )
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )
    frame = frame.dropna(
        subset=[
            "security_key",
            "snapshot_date",
            "price_date",
            "short_interest_pct",
        ]
    )
    frame = (
        frame
        .sort_values(
            [
                "security_key",
                "snapshot_date",
                "price_date",
            ]
        )
        .reset_index(drop=True)
    )
    return frame
def _parse_nasdaq_number(
    value: Any,
) -> float:
    """
    Parse Nasdaq numeric strings.
    Examples:
        "1,123.45" -> 1123.45
        "$1,123.45" -> 1123.45
        "1,123.45%" -> 1123.45
    """
    if value is None:
        return float("nan")
    if isinstance(
        value,
        (int, float),
    ):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    text = (
        text
        .replace(",", "")
        .replace("$", "")
        .replace("%", "")
        .strip()
    )
    return float(text)
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
        print(
            "Nasdaq-svar, början:\n"
            f"{json.dumps("
            "payload, "
            "ensure_ascii=False, "
            "default=str"
            ")[:2000]}"
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
    # Blankdiss har just nu bara prisdata från
    # 2026-03-09, så ett års eller mindre
    # marknadshistorik är legitimt för denna analys.
    #
    # De senaste observationerna kommer naturligt
    # att sakna 60-dagars forward return eftersom
    # framtida marknadsdata ännu inte finns.
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
def calculate_forward_returns(
    prices: pd.Series,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """
    Calculate forward returns using trading observations.
    Example:
        horizon=5
    means:
        close[t+5] / close[t] - 1
    """
    result = pd.DataFrame(
        index=prices.index
    )
    for horizon in horizons:
        result[
            f"market_forward_return_{horizon}d"
        ] = (
            prices.shift(-horizon)
            / prices
            - 1.0
        )
    return result
def build_market_returns(
    frame: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align stock price_date with OMXSPI and calculate
    market forward returns.
    The first OMXSPI trading observation on or after
    the stock's price_date is used.
    """
    result = frame.copy()
    market = market.copy()
    market_returns = (
        calculate_forward_returns(
            market["market_close"],
            HORIZONS,
        )
    )
    market = pd.concat(
        [
            market,
            market_returns,
        ],
        axis=1,
    )
    stock_dates = (
        result["price_date"]
        .dt.normalize()
        .to_numpy()
    )
    market_dates = (
        market["market_date"]
        .dt.normalize()
        .to_numpy()
    )
    market_index = np.searchsorted(
        market_dates,
        stock_dates,
        side="left",
    )
    valid = (
        market_index
        < len(market)
    )
    result["market_date"] = pd.NaT
    for horizon in HORIZONS:
        result[
            f"market_forward_return_{horizon}d"
        ] = np.nan
    if valid.any():
        target_positions = (
            market_index[valid]
        )
        result.loc[
            valid,
            "market_date",
        ] = (
            market.iloc[
                target_positions
            ]["market_date"]
            .to_numpy()
        )
        for horizon in HORIZONS:
            column = (
                f"market_forward_return_{horizon}d"
            )
            result.loc[
                valid,
                column,
            ] = (
                market.iloc[
                    target_positions
                ][column]
                .to_numpy()
            )
    for horizon in HORIZONS:
        stock_column = (
            f"forward_return_{horizon}d"
        )
        market_column = (
            f"market_forward_return_{horizon}d"
        )
        abnormal_column = (
            f"abnormal_return_{horizon}d"
        )
        result[abnormal_column] = (
            pd.to_numeric(
                result[stock_column],
                errors="coerce",
            )
            - pd.to_numeric(
                result[market_column],
                errors="coerce",
            )
        )
    return result
def local_prominence(
    values: pd.Series,
    dates: pd.Series,
    position: int,
    event_type: str,
) -> float:
    """
    Calculate local prominence for one candidate.
    Peak:
        peak - max(left minimum, right minimum)
    Trough:
        min(left maximum, right maximum) - trough
    The calculation is restricted to a finite local
    window instead of using the entire company history.
    """
    event_date = dates.iloc[position]
    window_start = (
        event_date
        - pd.Timedelta(
            days=PROMINENCE_WINDOW_DAYS
        )
    )
    window_end = (
        event_date
        + pd.Timedelta(
            days=PROMINENCE_WINDOW_DAYS
        )
    )
    mask = (
        (dates >= window_start)
        & (dates <= window_end)
    )
    positions = np.flatnonzero(
        mask.to_numpy()
    )
    left_positions = positions[
        positions < position
    ]
    right_positions = positions[
        positions > position
    ]
    if len(left_positions) < 2:
        return 0.0
    if len(right_positions) < 2:
        return 0.0
    current_value = float(
        values.iloc[position]
    )
    if event_type == "peak":
        left_base = float(
            values.iloc[
                left_positions
            ].min()
        )
        right_base = float(
            values.iloc[
                right_positions
            ].min()
        )
        return (
            current_value
            - max(
                left_base,
                right_base,
            )
        )
    if event_type == "trough":
        left_base = float(
            values.iloc[
                left_positions
            ].max()
        )
        right_base = float(
            values.iloc[
                right_positions
            ].max()
        )
        return (
            min(
                left_base,
                right_base,
            )
            - current_value
        )
    raise ValueError(
        f"Okänd event_type: {event_type}"
    )
def candidate_events(
    company: pd.DataFrame,
    event_type: str,
) -> list[dict[str, Any]]:
    """
    Find candidate local peaks or troughs.
    Candidates must be higher/lower than their immediate
    neighbouring observations.
    """
    values = (
        company[
            "short_interest_pct"
        ]
        .astype(float)
        .reset_index(drop=True)
    )
    dates = (
        company[
            "snapshot_date"
        ]
        .reset_index(drop=True)
    )
    candidates: list[
        dict[str, Any]
    ] = []
    if len(company) < 3:
        return candidates
    for position in range(
        1,
        len(company) - 1,
    ):
        previous_value = float(
            values.iloc[
                position - 1
            ]
        )
        current_value = float(
            values.iloc[position]
        )
        next_value = float(
            values.iloc[
                position + 1
            ]
        )
        if event_type == "peak":
            is_candidate = (
                current_value
                > previous_value
                and current_value
                >= next_value
            )
        elif event_type == "trough":
            is_candidate = (
                current_value
                < previous_value
                and current_value
                <= next_value
            )
        else:
            raise ValueError(
                f"Okänd event_type: {event_type}"
            )
        if not is_candidate:
            continue
        prominence = local_prominence(
            values,
            dates,
            position,
            event_type,
        )
        if prominence < MIN_PROMINENCE_PP:
            continue
        candidates.append(
            {
                "position": position,
                "snapshot_date": dates.iloc[
                    position
                ],
                "short_interest_pct": (
                    current_value
                ),
                "event_type": event_type,
                "prominence_pp": prominence,
            }
        )
    return candidates
def select_separated_events(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Keep meaningful events separated in time.
    When two candidates occur too close together,
    keep the more prominent event.
    """
    if not candidates:
        return []
    ordered = sorted(
        candidates,
        key=lambda item: item[
            "snapshot_date"
        ],
    )
    selected: list[
        dict[str, Any]
    ] = []
    for candidate in ordered:
        if not selected:
            selected.append(
                candidate
            )
            continue
        previous = selected[-1]
        days_apart = abs(
            (
                candidate[
                    "snapshot_date"
                ]
                - previous[
                    "snapshot_date"
                ]
            ).days
        )
        if (
            days_apart
            >= MIN_EVENT_SEPARATION_DAYS
        ):
            selected.append(
                candidate
            )
            continue
        if (
            candidate[
                "prominence_pp"
            ]
            > previous[
                "prominence_pp"
            ]
        ):
            selected[-1] = candidate
    return selected
def detect_company_cycles(
    company: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Detect recurring short-interest peaks and troughs
    for one security.
    """
    company = (
        company
        .sort_values(
            "snapshot_date"
        )
        .reset_index(drop=True)
    )
    peaks = select_separated_events(
        candidate_events(
            company,
            "peak",
        )
    )
    troughs = select_separated_events(
        candidate_events(
            company,
            "trough",
        )
    )
    events = (
        peaks
        + troughs
    )
    events.sort(
        key=lambda item: item[
            "snapshot_date"
        ]
    )
    return events
def attach_event_outcomes(
    events: list[dict[str, Any]],
    company: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Attach stock and abnormal forward returns to
    detected cycle events.
    The event itself is based on short-interest history.
    Returns are read from the feature row corresponding
    to the event snapshot.
    """
    if not events:
        return []
    result: list[
        dict[str, Any]
    ] = []
    for event in events:
        event_date = event[
            "snapshot_date"
        ]
        matches = company[
            company[
                "snapshot_date"
            ]
            == event_date
        ]
        if matches.empty:
            continue
        row = matches.iloc[0]
        item: dict[str, Any] = {
            "security_key": str(
                row["security_key"]
            ),
            "snapshot_date": (
                event_date.strftime(
                    "%Y-%m-%d"
                )
            ),
            "price_date": (
                row["price_date"].strftime(
                    "%Y-%m-%d"
                )
            ),
            "event_type": event[
                "event_type"
            ],
            "short_interest_pct": float(
                event[
                    "short_interest_pct"
                ]
            ),
            "prominence_pp": float(
                event[
                    "prominence_pp"
                ]
            ),
        }
        for horizon in HORIZONS:
            stock_column = (
                f"forward_return_{horizon}d"
            )
            abnormal_column = (
                f"abnormal_return_{horizon}d"
            )
            stock_value = pd.to_numeric(
                row.get(
                    stock_column
                ),
                errors="coerce",
            )
            abnormal_value = pd.to_numeric(
                row.get(
                    abnormal_column
                ),
                errors="coerce",
            )
            item[
                stock_column
            ] = (
                None
                if pd.isna(stock_value)
                else float(stock_value)
            )
            item[
                abnormal_column
            ] = (
                None
                if pd.isna(abnormal_value)
                else float(abnormal_value)
            )
        result.append(item)
    return result
def build_cycle_analysis(
    frame: pd.DataFrame,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Detect cycles for every security.
    Returns:
        events
        summaries
    """
    all_events: list[
        dict[str, Any]
    ] = []
    summaries: list[
        dict[str, Any]
    ] = []
    grouped = frame.groupby(
        "security_key",
        sort=False,
    )
    for security_key, company in grouped:
        events = detect_company_cycles(
            company
        )
        events = attach_event_outcomes(
            events,
            company,
        )
        all_events.extend(
            events
        )
        peaks = [
            event
            for event in events
            if event[
                "event_type"
            ] == "peak"
        ]
        troughs = [
            event
            for event in events
            if event[
                "event_type"
            ] == "trough"
        ]
        summary: dict[str, Any] = {
            "security_key": str(
                security_key
            ),
            "cycle_event_count": len(
                events
            ),
            "peak_count": len(
                peaks
            ),
            "trough_count": len(
                troughs
            ),
        }
        for event_type, event_list in (
            (
                "peak",
                peaks,
            ),
            (
                "trough",
                troughs,
            ),
        ):
            for horizon in HORIZONS:
                column = (
                    f"abnormal_return_{horizon}d"
                )
                values = [
                    event[column]
                    for event in event_list
                    if event.get(column)
                    is not None
                ]
                if values:
                    summary[
                        f"{event_type}_mean_abnormal_return_{horizon}d"
                    ] = float(
                        np.mean(values)
                    )
                    summary[
                        f"{event_type}_median_abnormal_return_{horizon}d"
                    ] = float(
                        np.median(values)
                    )
                else:
                    summary[
                        f"{event_type}_mean_abnormal_return_{horizon}d"
                    ] = None
                    summary[
                        f"{event_type}_median_abnormal_return_{horizon}d"
                    ] = None
        summaries.append(
            summary
        )
    summaries.sort(
        key=lambda item: (
            -item[
                "cycle_event_count"
            ],
            item[
                "security_key"
            ],
        )
    )
    return (
        all_events,
        summaries,
    )
def _json_value(
    value: Any,
) -> Any:
    """
    Convert pandas/numpy values to JSON-safe
    Python values.
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
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(
        value,
        pd.Timestamp,
    ):
        if pd.isna(value):
            return None
        return value.strftime(
            "%Y-%m-%d"
        )
    if pd.isna(value):
        return None
    return value
def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write rows as UTF-8 JSONL."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            clean_row = {
                key: _json_value(value)
                for key, value in row.items()
            }
            handle.write(
                json.dumps(
                    clean_row,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
def write_summary(
    path: Path,
    summaries: list[dict[str, Any]],
) -> None:
    """Write cycle summaries as JSON."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    clean_summaries = []
    for summary in summaries:
        clean_summaries.append(
            {
                key: _json_value(value)
                for key, value in summary.items()
            }
        )
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            clean_summaries,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
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
    frame = load_features()
    print(
        "Feature-rader: "
        f"{len(frame):,}"
    )
    print(
        "Datum: "
        f"{frame['snapshot_date'].min().date()} "
        "-> "
        f"{frame['snapshot_date'].max().date()}"
    )
    price_dates = frame[
        "price_date"
    ].dropna()
    if price_dates.empty:
        raise RuntimeError(
            "Feature-datasetet innehåller inga "
            "price_date-värden."
        )
    market = download_market_data(
        price_dates.min(),
        price_dates.max(),
    )
    market_frame = build_market_returns(
        frame,
        market,
    )
    market_rows = (
        market_frame
        .replace(
            {
                pd.NaT: None,
                np.nan: None,
            }
        )
        .to_dict(
            orient="records"
        )
    )
    write_jsonl(
        MARKET_RETURNS_PATH,
        market_rows,
    )
    print(
        "Market-adjusted returns skrivna: "
        f"{MARKET_RETURNS_PATH}"
    )
    events, summaries = (
        build_cycle_analysis(
            market_frame
        )
    )
    write_jsonl(
        CYCLES_PATH,
        events,
    )
    write_summary(
        CYCLE_SUMMARY_PATH,
        summaries,
    )
    companies_with_cycles = sum(
        1
        for summary in summaries
        if summary[
            "cycle_event_count"
        ] > 0
    )
    print(
        "Short-cycle events: "
        f"{len(events):,}"
    )
    print(
        "Bolag med cycle events: "
        f"{companies_with_cycles:,}"
    )
    print(
        "Cycle events skrivna: "
        f"{CYCLES_PATH}"
    )
    print(
        "Cycle summary skriven: "
        f"{CYCLE_SUMMARY_PATH}"
    )
    print(
        "=========================================="
    )
    print(
        "MARKET & SHORT CYCLE ANALYSIS KLAR"
    )
    print(
        "=========================================="
    )
if __name__ == "__main__":
    main()
