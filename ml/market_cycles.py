"""
Market-adjusted returns and recurring short-interest cycle analysis.

This module performs two diagnostic analyses:

1. Market-adjusted returns
   - Downloads OMXSPI (^OMXSPI)
   - Calculates market forward returns for 5/20/60 trading observations
   - Calculates abnormal stock returns relative to OMXSPI

2. Short-interest cycles
   - Detects local short-interest peaks and troughs
   - Requires a minimum local prominence
   - Requires minimum separation between peaks/troughs
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
import yfinance as yf


FEATURES_PATH = Path(
    "data/processed/analysis/features.jsonl"
)

OUTPUT_DIR = Path(
    "data/processed/analysis"
)

MARKET_RETURNS_PATH = (
    OUTPUT_DIR / "market_adjusted_returns.jsonl"
)

CYCLES_PATH = (
    OUTPUT_DIR / "short_cycles.jsonl"
)

CYCLE_SUMMARY_PATH = (
    OUTPUT_DIR / "short_cycles_summary.json"
)

MARKET_SYMBOL = "^OMXSPI"

HORIZONS = (
    5,
    20,
    60,
)

# Minimum difference between a local peak and its
# surrounding local base, expressed in percentage points.
MIN_PROMINENCE_PP = 0.25

# Minimum calendar days between separate peaks/troughs.
MIN_EVENT_SEPARATION_DAYS = 30

# Local window used when calculating prominence.
PROMINENCE_WINDOW_DAYS = 180


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

            rows.append(json.loads(line))

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
        required.difference(frame.columns)
    )

    if missing:
        raise ValueError(
            "Följande kolumner saknas i feature-datasetet: "
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
        column = f"forward_return_{horizon}d"

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

    frame = frame.sort_values(
        [
            "security_key",
            "snapshot_date",
            "price_date",
        ]
    ).reset_index(drop=True)

    return frame


def download_market_data(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Download OMXSPI history.

    A small buffer is added on both sides so that the first/last
    observations can still be used for forward-return calculations.
    """

    start = (
        start_date
        - pd.Timedelta(days=10)
    ).strftime("%Y-%m-%d")

    end = (
        end_date
        + pd.Timedelta(days=120)
    ).strftime("%Y-%m-%d")

    print(
        f"Laddar marknadsdata: {MARKET_SYMBOL}"
    )

    data = yf.download(
        MARKET_SYMBOL,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise RuntimeError(
            "Kunde inte hämta OMXSPI-data."
        )

    # yfinance can return MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"]

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

        else:
            raise RuntimeError(
                "OMXSPI-data saknar Close-kolumn."
            )

    else:
        if "Close" not in data.columns:
            raise RuntimeError(
                "OMXSPI-data saknar Close-kolumn."
            )

        close = data["Close"]

    market = pd.DataFrame(
        {
            "market_date": pd.to_datetime(
                close.index
            ),
            "market_close": pd.to_numeric(
                close,
                errors="coerce",
            ),
        }
    )

    market = market.dropna(
        subset=["market_close"]
    )

    market = (
        market
        .drop_duplicates(
            subset=["market_date"]
        )
        .sort_values("market_date")
        .reset_index(drop=True)
    )

    return market


def calculate_forward_returns(
    prices: pd.Series,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """
    Calculate forward returns using trading observations.

    Example:
        horizon=5 means close[t+5] / close[t] - 1.
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
    Align each stock price_date with OMXSPI and calculate
    market forward returns.
    """

    result = frame.copy()

    market = market.copy()

    market_returns = calculate_forward_returns(
        market["market_close"],
        HORIZONS,
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
        target_positions = market_index[valid]

        result.loc[
            valid,
            "market_date",
        ] = market.iloc[
            target_positions
        ]["market_date"].to_numpy()

        for horizon in HORIZONS:
            column = (
                f"market_forward_return_{horizon}d"
            )

            result.loc[
                valid,
                column,
            ] = market.iloc[
                target_positions
            ][column].to_numpy()

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
    Calculate local prominence for one candidate peak/trough.

    For a peak:
        prominence =
            peak - max(left minimum, right minimum)

    For a trough:
        prominence =
            min(left maximum, right maximum) - trough

    The calculation is deliberately restricted to a finite
    time window rather than using the entire company history.
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

    window_positions = np.flatnonzero(
        mask.to_numpy()
    )

    left_positions = (
        window_positions[
            window_positions < position
        ]
    )

    right_positions = (
        window_positions[
            window_positions > position
        ]
    )

    if len(left_positions) < 2:
        return 0.0

    if len(right_positions) < 2:
        return 0.0

    current_value = float(
        values.iloc[position]
    )

    if event_type == "peak":
        left_base = float(
            values.iloc[left_positions].min()
        )

        right_base = float(
            values.iloc[right_positions].min()
        )

        base = max(
            left_base,
            right_base,
        )

        return current_value - base

    if event_type == "trough":
        left_base = float(
            values.iloc[left_positions].max()
        )

        right_base = float(
            values.iloc[right_positions].max()
        )

        base = min(
            left_base,
            right_base,
        )

        return base - current_value

    raise ValueError(
        f"Okänd event_type: {event_type}"
    )


def candidate_events(
    company: pd.DataFrame,
    event_type: str,
) -> list[dict[str, Any]]:
    """
    Find candidate local peaks or troughs.

    A candidate must be strictly higher/lower than the immediate
    neighbouring observations.
    """

    values = (
        company["short_interest_pct"]
        .astype(float)
        .reset_index(drop=True)
    )

    dates = (
        company["snapshot_date"]
        .reset_index(drop=True)
    )

    candidates: list[dict[str, Any]] = []

    if len(company) < 3:
        return candidates

    for position in range(
        1,
        len(company) - 1,
    ):
        previous_value = float(
            values.iloc[position - 1]
        )

        current_value = float(
            values.iloc[position]
        )

        next_value = float(
            values.iloc[position + 1]
        )

        if event_type == "peak":
            is_candidate = (
                current_value >= previous_value
                and current_value > next_value
            )

        elif event_type == "trough":
            is_candidate = (
                current_value <= previous_value
                and current_value < next_value
            )

        else:
            raise ValueError(
                f"Okänd event_type: {event_type}"
            )

        if not is_candidate:
            continue

        prominence = local_prominence(
            values=values,
            dates=dates,
            position=position,
            event_type=event_type,
        )

        if prominence < MIN_PROMINENCE_PP:
            continue

        candidates.append(
            {
                "position": position,
                "snapshot_date": dates.iloc[position],
                "short_interest_pct": current_value,
                "prominence_pp": prominence,
            }
        )

    return candidates


def select_separated_events(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Enforce minimum separation between events.

    If two events occur within the minimum separation window,
    retain the event with the greatest prominence.

    This prevents a noisy rise/fall/rise sequence from being
    interpreted as multiple independent cycles.
    """

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda event: (
            event["prominence_pp"],
            event["snapshot_date"],
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []

    for candidate in candidates:
        candidate_date = pd.Timestamp(
            candidate["snapshot_date"]
        )

        conflicts = False

        for existing in selected:
            existing_date = pd.Timestamp(
                existing["snapshot_date"]
            )

            separation = abs(
                (
                    candidate_date
                    - existing_date
                ).days
            )

            if (
                separation
                < MIN_EVENT_SEPARATION_DAYS
            ):
                conflicts = True
                break

        if not conflicts:
            selected.append(candidate)

    selected.sort(
        key=lambda event: event["snapshot_date"]
    )

    return selected


def detect_company_events(
    company: pd.DataFrame,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Detect robust peaks and troughs for one company."""

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

    return peaks, troughs


def row_for_event(
    company: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> pd.Series | None:
    """Return the feature row closest to an event date."""

    matches = company[
        company["snapshot_date"]
        == snapshot_date
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def build_cycle_records(
    company: pd.DataFrame,
    peaks: list[dict[str, Any]],
    troughs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build cycle records.

    Each peak is connected to the next trough and next peak.
    This gives us both the immediate post-peak behaviour and
    the length of the recurring cycle.
    """

    records: list[dict[str, Any]] = []

    security_key = str(
        company["security_key"].iloc[0]
    )

    ordered_events = sorted(
        [
            (
                "peak",
                event,
            )
            for event in peaks
        ]
        + [
            (
                "trough",
                event,
            )
            for event in troughs
        ],
        key=lambda item: item[1][
            "snapshot_date"
        ],
    )

    for event_type, event in ordered_events:
        event_date = pd.Timestamp(
            event["snapshot_date"]
        )

        event_row = row_for_event(
            company,
            event_date,
        )

        if event_row is None:
            continue

        if event_type == "peak":
            following_troughs = [
                trough
                for trough in troughs
                if pd.Timestamp(
                    trough["snapshot_date"]
                ) > event_date
            ]

            following_peaks = [
                peak
                for peak in peaks
                if pd.Timestamp(
                    peak["snapshot_date"]
                ) > event_date
            ]

            next_trough = (
                following_troughs[0]
                if following_troughs
                else None
            )

            next_peak = (
                following_peaks[0]
                if following_peaks
                else None
            )

            record: dict[str, Any] = {
                "security_key": security_key,
                "event_type": "peak",
                "snapshot_date": event_date.strftime(
                    "%Y-%m-%d"
                ),
                "price_date": pd.Timestamp(
                    event_row["price_date"]
                ).strftime(
                    "%Y-%m-%d"
                ),
                "short_interest_pct": float(
                    event["short_interest_pct"]
                ),
                "prominence_pp": float(
                    event["prominence_pp"]
                ),
            }

            if next_trough is not None:
                trough_date = pd.Timestamp(
                    next_trough["snapshot_date"]
                )

                trough_row = row_for_event(
                    company,
                    trough_date,
                )

                record[
                    "next_trough_date"
                ] = trough_date.strftime(
                    "%Y-%m-%d"
                )

                record[
                    "days_to_next_trough"
                ] = int(
                    (
                        trough_date
                        - event_date
                    ).days
                )

                record[
                    "next_trough_short_interest_pct"
                ] = float(
                    next_trough[
                        "short_interest_pct"
                    ]
                )

                if trough_row is not None:
                    for horizon in HORIZONS:
                        record[
                            f"forward_return_to_trough_{horizon}d"
                        ] = _safe_float(
                            trough_row[
                                f"forward_return_{horizon}d"
                            ]
                        )

                        record[
                            f"abnormal_return_to_trough_{horizon}d"
                        ] = _safe_float(
                            trough_row[
                                f"abnormal_return_{horizon}d"
                            ]
                        )

            else:
                record[
                    "next_trough_date"
                ] = None

                record[
                    "days_to_next_trough"
                ] = None

                record[
                    "next_trough_short_interest_pct"
                ] = None

            if next_peak is not None:
                next_peak_date = pd.Timestamp(
                    next_peak["snapshot_date"]
                )

                record[
                    "next_peak_date"
                ] = next_peak_date.strftime(
                    "%Y-%m-%d"
                )

                record[
                    "days_to_next_peak"
                ] = int(
                    (
                        next_peak_date
                        - event_date
                    ).days
                )

                record[
                    "next_peak_short_interest_pct"
                ] = float(
                    next_peak[
                        "short_interest_pct"
                    ]
                )

            else:
                record[
                    "next_peak_date"
                ] = None

                record[
                    "days_to_next_peak"
                ] = None

                record[
                    "next_peak_short_interest_pct"
                ] = None

            records.append(record)

        else:
            following_peaks = [
                peak
                for peak in peaks
                if pd.Timestamp(
                    peak["snapshot_date"]
                ) > event_date
            ]

            next_peak = (
                following_peaks[0]
                if following_peaks
                else None
            )

            record = {
                "security_key": security_key,
                "event_type": "trough",
                "snapshot_date": event_date.strftime(
                    "%Y-%m-%d"
                ),
                "price_date": pd.Timestamp(
                    event_row["price_date"]
                ).strftime(
                    "%Y-%m-%d"
                ),
                "short_interest_pct": float(
                    event["short_interest_pct"]
                ),
                "prominence_pp": float(
                    event["prominence_pp"]
                ),
            }

            if next_peak is not None:
                next_peak_date = pd.Timestamp(
                    next_peak["snapshot_date"]
                )

                record[
                    "next_peak_date"
                ] = next_peak_date.strftime(
                    "%Y-%m-%d"
                )

                record[
                    "days_to_next_peak"
                ] = int(
                    (
                        next_peak_date
                        - event_date
                    ).days
                )

                record[
                    "next_peak_short_interest_pct"
                ] = float(
                    next_peak[
                        "short_interest_pct"
                    ]
                )

            else:
                record[
                    "next_peak_date"
                ] = None

                record[
                    "days_to_next_peak"
                ] = None

                record[
                    "next_peak_short_interest_pct"
                ] = None

            records.append(record)

    return records


def _safe_float(
    value: Any,
) -> float | None:
    """Convert numeric values while preserving missing values."""

    if value is None:
        return None

    try:
        numeric = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(numeric):
        return None

    return numeric


def build_company_summary(
    company: pd.DataFrame,
    peaks: list[dict[str, Any]],
    troughs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one summary row per company."""

    security_key = str(
        company["security_key"].iloc[0]
    )

    peak_prominences = [
        float(
            event["prominence_pp"]
        )
        for event in peaks
    ]

    trough_prominences = [
        float(
            event["prominence_pp"]
        )
        for event in troughs
    ]

    peak_dates = [
        pd.Timestamp(
            event["snapshot_date"]
        )
        for event in peaks
    ]

    cycle_lengths: list[int] = []

    for index in range(
        1,
        len(peak_dates),
    ):
        cycle_lengths.append(
            int(
                (
                    peak_dates[index]
                    - peak_dates[index - 1]
                ).days
            )
        )

    summary: dict[str, Any] = {
        "security_key": security_key,
        "observations": int(
            len(company)
        ),
        "first_snapshot_date": pd.Timestamp(
            company["snapshot_date"].min()
        ).strftime("%Y-%m-%d"),
        "last_snapshot_date": pd.Timestamp(
            company["snapshot_date"].max()
        ).strftime("%Y-%m-%d"),
        "peak_count": len(peaks),
        "trough_count": len(troughs),
        "max_peak_prominence_pp": (
            max(peak_prominences)
            if peak_prominences
            else None
        ),
        "median_peak_prominence_pp": (
            float(
                np.median(
                    peak_prominences
                )
            )
            if peak_prominences
            else None
        ),
        "max_trough_prominence_pp": (
            max(trough_prominences)
            if trough_prominences
            else None
        ),
        "median_trough_prominence_pp": (
            float(
                np.median(
                    trough_prominences
                )
            )
            if trough_prominences
            else None
        ),
        "median_days_between_peaks": (
            float(
                np.median(
                    cycle_lengths
                )
            )
            if cycle_lengths
            else None
        ),
        "mean_days_between_peaks": (
            float(
                np.mean(
                    cycle_lengths
                )
            )
            if cycle_lengths
            else None
        ),
    }

    # A company with repeated peaks is especially interesting
    # for the "cyclical shorting" hypothesis.
    summary[
        "repeated_cycle_candidate"
    ] = len(peaks) >= 3

    return summary


def analyze_cycles(
    frame: pd.DataFrame,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Run cycle analysis for all companies."""

    cycle_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    grouped = frame.groupby(
        "security_key",
        sort=False,
    )

    company_count = 0

    for security_key, company in grouped:
        company_count += 1

        company = (
            company
            .sort_values(
                [
                    "snapshot_date",
                    "price_date",
                ]
            )
            .reset_index(drop=True)
        )

        peaks, troughs = detect_company_events(
            company
        )

        company_records = build_cycle_records(
            company,
            peaks,
            troughs,
        )

        cycle_records.extend(
            company_records
        )

        summaries.append(
            build_company_summary(
                company,
                peaks,
                troughs,
            )
        )

    print(
        f"Analyserade bolag: {company_count}"
    )

    print(
        f"Identifierade toppar: "
        f"{sum(1 for record in cycle_records if record['event_type'] == 'peak')}"
    )

    print(
        f"Identifierade dalar: "
        f"{sum(1 for record in cycle_records if record['event_type'] == 'trough')}"
    )

    return cycle_records, summaries


def write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write JSON Lines output."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def write_summary(
    path: Path,
    summaries: list[dict[str, Any]],
) -> None:
    """Write cycle summary JSON."""

    repeated = [
        summary
        for summary in summaries
        if summary[
            "repeated_cycle_candidate"
        ]
    ]

    repeated.sort(
        key=lambda summary: (
            summary[
                "peak_count"
            ],
            summary[
                "max_peak_prominence_pp"
            ]
            or 0.0,
        ),
        reverse=True,
    )

    output = {
        "configuration": {
            "market_symbol": MARKET_SYMBOL,
            "horizons": list(HORIZONS),
            "min_prominence_pp": (
                MIN_PROMINENCE_PP
            ),
            "min_event_separation_days": (
                MIN_EVENT_SEPARATION_DAYS
            ),
            "prominence_window_days": (
                PROMINENCE_WINDOW_DAYS
            ),
        },
        "company_count": len(
            summaries
        ),
        "repeated_cycle_company_count": len(
            repeated
        ),
        "companies": summaries,
        "repeated_cycle_candidates": repeated,
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Run the complete market/cycle analysis."""

    print()
    print(
        "=========================================="
    )
    print(
        "BLANKDISS MARKET & SHORT CYCLE ANALYSIS"
    )
    print(
        "=========================================="
    )
    print()

    frame = load_features()

    print(
        f"Feature-rader: {len(frame):,}"
    )

    print(
        "Datum: "
        f"{frame['snapshot_date'].min().date()} "
        "-> "
        f"{frame['snapshot_date'].max().date()}"
    )

    market = download_market_data(
        start_date=frame["price_date"].min(),
        end_date=frame["price_date"].max(),
    )

    print(
        f"OMXSPI-observationer: {len(market):,}"
    )

    result = build_market_returns(
        frame,
        market,
    )

    result = result.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    market_rows = (
        result
        .to_dict(orient="records")
    )

    # Convert timestamps into JSON-safe strings.
    for row in market_rows:
        for key, value in list(
            row.items()
        ):
            if isinstance(
                value,
                pd.Timestamp,
            ):
                if pd.isna(value):
                    row[key] = None
                else:
                    row[key] = (
                        value.strftime(
                            "%Y-%m-%d"
                        )
                    )

            elif pd.isna(value):
                row[key] = None

            elif isinstance(
                value,
                np.generic,
            ):
                row[key] = value.item()

    write_jsonl(
        MARKET_RETURNS_PATH,
        market_rows,
    )

    print()
    print(
        f"Skrev: {MARKET_RETURNS_PATH}"
    )

    cycle_records, summaries = (
        analyze_cycles(result)
    )

    write_jsonl(
        CYCLES_PATH,
        cycle_records,
    )

    write_summary(
        CYCLE_SUMMARY_PATH,
        summaries,
    )

    print(
        f"Skrev: {CYCLES_PATH}"
    )

    print(
        f"Skrev: {CYCLE_SUMMARY_PATH}"
    )

    print()
    print(
        "=========================================="
    )
    print(
        "MARKET & CYCLE ANALYSIS KLAR"
    )
    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()
