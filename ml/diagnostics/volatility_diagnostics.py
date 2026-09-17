"""
Blankdiss volatility diagnostics.

Descriptive analysis of the relationship between:

    - price_volatility_20d
    - FI short-interest features
    - downside event probability
    - forward 5-day returns

This script does not train ML models.

No repository files are modified by this script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.dataset import load_features


TARGET_RETURN_COLUMN = "forward_return_5d"
VOLATILITY_COLUMN = "price_volatility_20d"
FI_COLUMN = "short_interest_pct"

TARGET_THRESHOLD = -0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def add_target(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the economic downside target.

    The Blankdiss target definition for down_5pct_5d is:

        forward_return_5d <= -0.05
    """

    result = frame.copy()

    result["down_5pct_5d"] = (
        pd.to_numeric(
            result[TARGET_RETURN_COLUMN],
            errors="coerce",
        )
        <= TARGET_THRESHOLD
    ).astype(int)

    return result


def print_quantiles(
    frame: pd.DataFrame,
    column: str,
):
    """
    Print standard quantiles for a numeric column.
    """

    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return

    quantiles = values.quantile(
        [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ]
    )

    print(
        f"{column} quantiles:"
    )

    for quantile, value in quantiles.items():
        print(
            f"  {quantile:.1f}: "
            f"{value:.6f}"
        )


def print_quintile_analysis(
    frame: pd.DataFrame,
    column: str,
    label: str,
):
    """
    Print downside event rate by quintile.
    """

    data = frame.dropna(
        subset=[
            column,
            "down_5pct_5d",
        ]
    ).copy()

    if data.empty:
        return

    data["quintile"] = pd.qcut(
        data[column],
        q=5,
        labels=[
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "Q5",
        ],
        duplicates="drop",
    )

    baseline = data[
        "down_5pct_5d"
    ].mean()

    print()
    print(
        f"{label} quintiles:"
    )

    for quintile in [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
    ]:
        subset = data[
            data["quintile"] == quintile
        ]

        if subset.empty:
            continue

        event_rate = subset[
            "down_5pct_5d"
        ].mean()

        lift = (
            event_rate / baseline
            if baseline > 0
            else np.nan
        )

        print(
            f"  {quintile}: "
            f"n={len(subset):,} "
            f"event={event_rate:.4f} "
            f"lift={lift:.2f}x"
        )


def print_fi_volatility_matrix(
    frame: pd.DataFrame,
):
    """
    Print downside event rate for a 5x5 FI x volatility matrix.
    """

    data = frame.dropna(
        subset=[
            FI_COLUMN,
            VOLATILITY_COLUMN,
            "down_5pct_5d",
        ]
    ).copy()

    if data.empty:
        return

    data["fi_q"] = pd.qcut(
        data[FI_COLUMN],
        q=5,
        labels=[
            "F1",
            "F2",
            "F3",
            "F4",
            "F5",
        ],
        duplicates="drop",
    )

    data["vol_q"] = pd.qcut(
        data[VOLATILITY_COLUMN],
        q=5,
        labels=[
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
        ],
        duplicates="drop",
    )

    matrix = data.pivot_table(
        index="fi_q",
        columns="vol_q",
        values="down_5pct_5d",
        aggfunc="mean",
    )

    matrix = matrix.reindex(
        index=[
            "F1",
            "F2",
            "F3",
            "F4",
            "F5",
        ],
        columns=[
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
        ],
    )

    print()
    print("=" * 100)
    print("DOWNSIDE EVENT RATE: FI x VOLATILITY")
    print("=" * 100)
    print(
        "Rows = FI quintile, "
        "columns = volatility quintile"
    )
    print()

    print(
        matrix.to_string(
            float_format=lambda value:
                f"{value:.4f}"
        )
    )


def print_extreme_groups(
    frame: pd.DataFrame,
):
    """
    Compare high/low FI and high/low volatility groups.
    """

    data = frame.dropna(
        subset=[
            FI_COLUMN,
            VOLATILITY_COLUMN,
            "down_5pct_5d",
        ]
    ).copy()

    if data.empty:
        return

    fi_high = data[
        FI_COLUMN
        >= data[FI_COLUMN].quantile(0.8)
    ]

    fi_low = data[
        FI_COLUMN
        <= data[FI_COLUMN].quantile(0.2)
    ]

    vol_high = data[
        VOLATILITY_COLUMN
        >= data[VOLATILITY_COLUMN].quantile(
            0.8
        )
    ]

    vol_low = data[
        VOLATILITY_COLUMN
        <= data[VOLATILITY_COLUMN].quantile(
            0.2
        )
    ]

    fi_high_vol_high = data[
        (
            FI_COLUMN
            >= data[FI_COLUMN].quantile(0.8)
        )
        & (
            VOLATILITY_COLUMN
            >= data[VOLATILITY_COLUMN].quantile(
                0.8
            )
        )
    ]

    fi_high_vol_low = data[
        (
            FI_COLUMN
            >= data[FI_COLUMN].quantile(0.8)
        )
        & (
            VOLATILITY_COLUMN
            <= data[VOLATILITY_COLUMN].quantile(
                0.2
            )
        )
    ]

    fi_low_vol_high = data[
        (
            FI_COLUMN
            <= data[FI_COLUMN].quantile(0.2)
        )
        & (
            VOLATILITY_COLUMN
            >= data[VOLATILITY_COLUMN].quantile(
                0.8
            )
        )
    ]

    baseline = data[
        "down_5pct_5d"
    ].mean()

    print()
    print("=" * 100)
    print("EXTREME GROUPS")
    print("=" * 100)

    groups = [
        (
            "All",
            data,
        ),
        (
            "High FI",
            fi_high,
        ),
        (
            "Low FI",
            fi_low,
        ),
        (
            "High volatility",
            vol_high,
        ),
        (
            "Low volatility",
            vol_low,
        ),
        (
            "High FI + high volatility",
            fi_high_vol_high,
        ),
        (
            "High FI + low volatility",
            fi_high_vol_low,
        ),
        (
            "Low FI + high volatility",
            fi_low_vol_high,
        ),
    ]

    for label, subset in groups:
        if subset.empty:
            continue

        event_rate = subset[
            "down_5pct_5d"
        ].mean()

        lift = (
            event_rate / baseline
            if baseline > 0
            else np.nan
        )

        print(
            f"{label:30s} "
            f"n={len(subset):7,d} "
            f"event={event_rate:.4f} "
            f"lift={lift:.2f}x"
        )


def print_period_analysis(
    frame: pd.DataFrame,
    year: int,
):
    """
    Print volatility quintiles for one calendar year.
    """

    data = frame[
        frame["snapshot_date"].dt.year
        == year
    ].copy()

    if data.empty:
        return

    print()
    print(
        f"{year} volatility quintiles:"
    )

    print_quintile_analysis(
        data,
        VOLATILITY_COLUMN,
        f"{year} volatility",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 100)
    print(
        "BLANKDISS VOLATILITY DIAGNOSTICS"
    )
    print("=" * 100)

    frame = load_features()

    print(
        f"Feature-rader: "
        f"{len(frame):,}"
    )

    frame = add_target(
        frame
    )

    frame = frame.dropna(
        subset=[
            TARGET_RETURN_COLUMN,
            VOLATILITY_COLUMN,
            FI_COLUMN,
        ]
    ).copy()

    print(
        f"Rows: "
        f"{len(frame):,}"
    )

    print(
        f"Date range: "
        f"{frame['snapshot_date'].min():%Y-%m-%d}"
        f" -> "
        f"{frame['snapshot_date'].max():%Y-%m-%d}"
    )

    baseline_event_rate = frame[
        "down_5pct_5d"
    ].mean()

    baseline_mean_return = frame[
        TARGET_RETURN_COLUMN
    ].mean()

    print(
        f"Baseline event rate: "
        f"{baseline_event_rate:.4f}"
    )

    print(
        f"Baseline mean return: "
        f"{baseline_mean_return:.4%}"
    )

    print()
    print_quantiles(
        frame,
        VOLATILITY_COLUMN,
    )

    print()
    print_quantiles(
        frame,
        FI_COLUMN,
    )

    print_quintile_analysis(
        frame,
        VOLATILITY_COLUMN,
        "Volatility",
    )

    print_quintile_analysis(
        frame,
        FI_COLUMN,
        "FI",
    )

    print_fi_volatility_matrix(
        frame
    )

    print_extreme_groups(
        frame
    )

    for year in [
        2025,
        2026,
    ]:
        print_period_analysis(
            frame,
            year,
        )

    print()
    print("=" * 100)
    print("VOLATILITY DIAGNOSTICS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
