"""
Analyserar marknadsjusterad avkastning och
återkommande blankningsmönster per bolag.
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

MARKET_SYMBOL = "^OMXSPI"

HORIZONS = (
    5,
    20,
    60,
)


def load_features() -> pd.DataFrame:
    frame = pd.read_json(
        FEATURES_PATH,
        lines=True,
    )

    required = {
        "security_key",
        "snapshot_date",
        "price_date",
        "short_interest_pct",
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Features saknar kolumner: "
            + ", ".join(
                sorted(missing)
            )
        )

    for column in (
        "snapshot_date",
        "price_date",
    ):
        frame[column] = pd.to_datetime(
            frame[column],
            errors="coerce",
        )

    numeric_columns = [
        "short_interest_pct",
        "forward_return_5d",
        "forward_return_20d",
        "forward_return_60d",
    ]

    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    frame = frame.loc[
        frame["security_key"].notna()
        & frame["price_date"].notna()
        & frame["short_interest_pct"].notna()
    ].copy()

    return frame.sort_values(
        [
            "security_key",
            "price_date",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )


def download_market_prices(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    data = yf.download(
        tickers=MARKET_SYMBOL,
        start=start.strftime(
            "%Y-%m-%d"
        ),
        end=(
            end
            + pd.Timedelta(days=5)
        ).strftime(
            "%Y-%m-%d"
        ),
        auto_adjust=False,
        actions=False,
        threads=False,
        timeout=30,
        progress=False,
        group_by="column",
    )

    if data is None or data.empty:
        raise RuntimeError(
            "Yahoo returnerade ingen "
            f"marknadsdata för {MARKET_SYMBOL}."
        )

    if isinstance(
        data.columns,
        pd.MultiIndex,
    ):
        if (
            "Close"
            not in data.columns.get_level_values(0)
        ):
            raise RuntimeError(
                "Marknadsdata saknar Close."
            )

        close = data["Close"]

        if isinstance(
            close,
            pd.DataFrame,
        ):
            close = close.iloc[:, 0]

    else:
        if "Close" not in data.columns:
            raise RuntimeError(
                "Marknadsdata saknar Close."
            )

        close = data["Close"]

    dates = pd.to_datetime(
        close.index,
        errors="coerce",
    )

    if getattr(
        dates,
        "tz",
        None,
    ) is not None:
        dates = dates.tz_localize(
            None
        )

    market = pd.DataFrame(
        {
            "date": dates,
            "market_close": pd.to_numeric(
                close,
                errors="coerce",
            ).to_numpy(
                dtype=float
            ),
        }
    )

    market = market.loc[
        market["date"].notna()
        & np.isfinite(
            market["market_close"]
        )
        & (
            market["market_close"]
            > 0
        )
    ].copy()

    return market.sort_values(
        "date",
        kind="mergesort",
    ).reset_index(
        drop=True
    )


def market_forward_returns(
    market: pd.DataFrame,
    dates: pd.Series,
) -> pd.DataFrame:
    market_dates = market[
        "date"
    ].to_numpy(
        dtype="datetime64[ns]"
    )

    closes = market[
        "market_close"
    ].to_numpy(
        dtype=float
    )

    rows: list[
        dict[str, Any]
    ] = []

    for value in dates:
        timestamp = np.datetime64(
            value.to_datetime64(),
            "ns",
        )

        entry_idx = int(
            np.searchsorted(
                market_dates,
                timestamp,
                side="left",
            )
        )

        result: dict[
            str,
            Any,
        ] = {
            "market_price_date": None,
        }

        if entry_idx >= len(
            market
        ):
            rows.append(result)
            continue

        result[
            "market_price_date"
        ] = (
            market.iloc[
                entry_idx
            ]["date"].strftime(
                "%Y-%m-%d"
            )
        )

        entry_price = closes[
            entry_idx
        ]

        for horizon in HORIZONS:
            target_idx = (
                entry_idx
                + horizon
            )

            column = (
                "market_forward_return_"
                f"{horizon}d"
            )

            if target_idx < len(
                closes
            ):
                result[column] = (
                    closes[target_idx]
                    / entry_price
                    - 1.0
                )
            else:
                result[column] = np.nan

        rows.append(result)

    return pd.DataFrame(
        rows,
        index=dates.index,
    )


def build_market_adjusted(
    features: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    market_part = (
        market_forward_returns(
            market,
            features["price_date"],
        )
    )

    result = features[
        [
            "security_key",
            "snapshot_date",
            "price_date",
            "short_interest_pct",
        ]
        + [
            column
            for column in (
                "forward_return_5d",
                "forward_return_20d",
                "forward_return_60d",
            )
            if column in features.columns
        ]
    ].copy()

    for column in market_part.columns:
        result[column] = (
            market_part[
                column
            ].values
        )

    for horizon in HORIZONS:
        stock_column = (
            f"forward_return_{horizon}d"
        )

        market_column = (
            "market_forward_return_"
            f"{horizon}d"
        )

        abnormal_column = (
            f"abnormal_return_{horizon}d"
        )

        if (
            stock_column in result.columns
            and market_column in result.columns
        ):
            result[
                abnormal_column
            ] = (
                result[stock_column]
                - result[market_column]
            )

    return result


def add_short_dynamics(
    group: pd.DataFrame,
) -> pd.DataFrame:
    group = group.sort_values(
        "price_date",
        kind="mergesort",
    ).copy()

    group[
        "short_delta_pp"
    ] = (
        group[
            "short_interest_pct"
        ].diff()
    )

    previous_short = (
        group[
            "short_interest_pct"
        ].shift(1)
    )

    next_short = (
        group[
            "short_interest_pct"
        ].shift(-1)
    )

    group[
        "short_local_peak"
    ] = (
        (
            group[
                "short_interest_pct"
            ]
            >= previous_short
        )
        & (
            group[
                "short_interest_pct"
            ]
            > next_short
        )
    )

    group[
        "short_local_trough"
    ] = (
        (
            group[
                "short_interest_pct"
            ]
            <= previous_short
        )
        & (
            group[
                "short_interest_pct"
            ]
            < next_short
        )
    )

    return group


def value_or_none(
    row: pd.Series,
    column: str,
) -> float | None:
    if column not in row.index:
        return None

    value = row[column]

    if pd.isna(value):
        return None

    return float(value)


def mean_or_none(
    values: pd.Series,
) -> float | None:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return None

    return float(
        numeric.mean()
    )


def build_cycles(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    cycle_rows: list[
        dict[str, Any]
    ] = []

    company_rows: list[
        dict[str, Any]
    ] = []

    for (
        security_key,
        group,
    ) in frame.groupby(
        "security_key",
        sort=False,
    ):
        group = add_short_dynamics(
            group
        )

        peaks = group.loc[
            group[
                "short_local_peak"
            ]
        ].copy()

        troughs = group.loc[
            group[
                "short_local_trough"
            ]
        ].copy()

        peak_dates = (
            peaks[
                "price_date"
            ].tolist()
        )

        trough_dates = (
            troughs[
                "price_date"
            ].tolist()
        )

        for _, row in peaks.iterrows():
            later_peaks = [
                value
                for value in peak_dates
                if value
                > row["price_date"]
            ]

            next_peak = (
                min(later_peaks)
                if later_peaks
                else None
            )

            later_troughs = [
                value
                for value in trough_dates
                if value
                > row["price_date"]
            ]

            next_trough = (
                min(later_troughs)
                if later_troughs
                else None
            )

            cycle_rows.append(
                {
                    "security_key": (
                        security_key
                    ),
                    "event": (
                        "short_peak"
                    ),
                    "event_date": (
                        row[
                            "price_date"
                        ].strftime(
                            "%Y-%m-%d"
                        )
                    ),
                    "short_interest_pct": (
                        float(
                            row[
                                "short_interest_pct"
                            ]
                        )
                    ),
                    "short_delta_pp": (
                        value_or_none(
                            row,
                            "short_delta_pp",
                        )
                    ),
                    "next_peak_date": (
                        next_peak.strftime(
                            "%Y-%m-%d"
                        )
                        if next_peak
                        is not None
                        else None
                    ),
                    "days_to_next_peak": (
                        (
                            next_peak
                            - row[
                                "price_date"
                            ]
                        ).days
                        if next_peak
                        is not None
                        else None
                    ),
                    "next_trough_date": (
                        next_trough.strftime(
                            "%Y-%m-%d"
                        )
                        if next_trough
                        is not None
                        else None
                    ),
                    "days_to_next_trough": (
                        (
                            next_trough
                            - row[
                                "price_date"
                            ]
                        ).days
                        if next_trough
                        is not None
                        else None
                    ),
                    "forward_return_5d": (
                        value_or_none(
                            row,
                            "forward_return_5d",
                        )
                    ),
                    "forward_return_20d": (
                        value_or_none(
                            row,
                            "forward_return_20d",
                        )
                    ),
                    "forward_return_60d": (
                        value_or_none(
                            row,
                            "forward_return_60d",
                        )
                    ),
                    "abnormal_return_5d": (
                        value_or_none(
                            row,
                            "abnormal_return_5d",
                        )
                    ),
                    "abnormal_return_20d": (
                        value_or_none(
                            row,
                            "abnormal_return_20d",
                        )
                    ),
                    "abnormal_return_60d": (
                        value_or_none(
                            row,
                            "abnormal_return_60d",
                        )
                    ),
                }
            )

        company_rows.append(
            {
                "security_key": (
                    security_key
                ),
                "rows": int(
                    len(group)
                ),
                "short_peaks": int(
                    len(peaks)
                ),
                "short_troughs": int(
                    len(troughs)
                ),
                "mean_short_interest_pct": (
                    float(
                        group[
                            "short_interest_pct"
                        ].mean()
                    )
                ),
                "median_short_interest_pct": (
                    float(
                        group[
                            "short_interest_pct"
                        ].median()
                    )
                ),
                "mean_short_delta_pp": (
                    mean_or_none(
                        group[
                            "short_delta_pp"
                        ]
                    )
                ),
                "mean_peak_forward_return_20d": (
                    mean_or_none(
                        peaks[
                            "forward_return_20d"
                        ]
                    )
                ),
                "mean_peak_abnormal_return_20d": (
                    mean_or_none(
                        peaks[
                            "abnormal_return_20d"
                        ]
                    )
                ),
            }
        )

    cycles = pd.DataFrame(
        cycle_rows
    )

    summary = {
        "company_count": int(
            len(company_rows)
        ),
        "cycle_event_count": int(
            len(cycles)
        ),
        "companies_with_at_least_3_peaks": int(
            sum(
                row[
                    "short_peaks"
                ] >= 3
                for row in company_rows
            )
        ),
        "mean_peak_forward_return_20d": (
            mean_or_none(
                cycles.get(
                    "forward_return_20d",
                    pd.Series(
                        dtype=float
                    ),
                )
            )
        ),
        "mean_peak_abnormal_return_20d": (
            mean_or_none(
                cycles.get(
                    "abnormal_return_20d",
                    pd.Series(
                        dtype=float
                    ),
                )
            )
        ),
        "companies": company_rows,
    }

    return (
        cycles,
        summary,
    )


def write_jsonl(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean = frame.copy()

    clean = clean.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    clean.to_json(
        path,
        orient="records",
        lines=True,
        force_ascii=False,
        date_format="iso",
    )


def main() -> None:
    print(
        "Market/cykelanalys: startar."
    )

    features = load_features()

    start = features[
        "price_date"
    ].min()

    end = features[
        "price_date"
    ].max()

    print(
        "Market/cykelanalys: "
        f"hämtar {MARKET_SYMBOL} "
        f"{start.date()} -> "
        f"{end.date()}"
    )

    market = (
        download_market_prices(
            start,
            end,
        )
    )

    adjusted = (
        build_market_adjusted(
            features,
            market,
        )
    )

    cycles, summary = (
        build_cycles(
            adjusted
        )
    )

    write_jsonl(
        adjusted,
        MARKET_RETURNS_PATH,
    )

    write_jsonl(
        cycles,
        CYCLES_PATH,
    )

    CYCLE_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CYCLE_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "Market/cykelanalys: "
        f"{len(adjusted):,} "
        "marknadsjusterade rader."
    )

    print(
        "Market/cykelanalys: "
        f"{len(cycles):,} "
        "lokala short-toppar."
    )

    print(
        "Market/cykelanalys: "
        f"{summary['companies_with_at_least_3_peaks']:,} "
        "bolag har minst tre "
        "lokala short-toppar."
    )

    print(
        "Market/cykelanalys: klart."
    )


if __name__ == "__main__":
    main()
