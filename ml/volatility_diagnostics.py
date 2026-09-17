from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import TARGET_DEFINITIONS
from ml.dataset import load_features


TARGET_NAME = "down_5pct_5d"

VOLATILITY_COLUMN = "price_volatility_20d"
FI_COLUMN = "short_interest_pct"

TEST_PERIODS = {
    "2025": ("2025-01-01", "2025-12-31"),
    "2026": ("2026-01-01", "2026-12-31"),
}


def print_header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def build_target(df: pd.DataFrame) -> pd.Series:
    definition = TARGET_DEFINITIONS[TARGET_NAME]

    target_return = df[definition.return_column]
    horizon = definition.horizon_days

    if definition.direction == "down":
        return (
            df[definition.min_return_column]
            <= -definition.threshold
        ).astype(int)

    if definition.direction == "up":
        return (
            df[definition.max_return_column]
            >= definition.threshold
        ).astype(int)

    raise ValueError(
        f"Unsupported target direction for {TARGET_NAME}: "
        f"{definition.direction}"
    )


def add_quantile_bins(
    df: pd.DataFrame,
    column: str,
    bins: int = 5,
) -> pd.DataFrame:
    result = df.copy()

    values = result[column]

    try:
        result[f"{column}_q"] = pd.qcut(
            values,
            q=bins,
            labels=[f"Q{i}" for i in range(1, bins + 1)],
            duplicates="drop",
        )
    except ValueError:
        result[f"{column}_q"] = pd.cut(
            values,
            bins=bins,
            labels=[f"Q{i}" for i in range(1, bins + 1)],
        )

    return result


def print_basic_summary(df: pd.DataFrame) -> None:
    print_header("DATASET")

    print(f"Rows:              {len(df):,}")
    print(
        f"Date range:        "
        f"{df['snapshot_date'].min()} -> "
        f"{df['snapshot_date'].max()}"
    )

    print()
    print(
        f"Baseline event rate: "
        f"{df[TARGET_NAME].mean():.4f}"
    )

    print(
        f"Baseline mean return: "
        f"{df['target_return'].mean():+.4f}%"
    )

    print(
        f"Baseline median return: "
        f"{df['target_return'].median():+.4f}%"
    )

    print()
    print("Feature availability:")

    for column in [VOLATILITY_COLUMN, FI_COLUMN]:
        available = df[column].notna().sum()
        print(
            f"  {column:<35} "
            f"{available:,} / {len(df):,} "
            f"({available / len(df):.1%})"
        )


def print_bucket_summary(
    df: pd.DataFrame,
    bucket_column: str,
    title: str,
) -> None:
    print_header(title)

    baseline_event = df[TARGET_NAME].mean()

    grouped = (
        df.dropna(subset=[bucket_column])
        .groupby(bucket_column, observed=True)
        .agg(
            rows=(TARGET_NAME, "size"),
            event_rate=(TARGET_NAME, "mean"),
            mean_return=("target_return", "mean"),
            median_return=("target_return", "median"),
        )
        .reset_index()
    )

    grouped["lift"] = (
        grouped["event_rate"] / baseline_event
    )

    print(
        f"{'Bucket':<10}"
        f"{'Rows':>10}"
        f"{'Event':>12}"
        f"{'Lift':>10}"
        f"{'Mean ret':>14}"
        f"{'Median ret':>14}"
    )

    print("-" * 70)

    for _, row in grouped.iterrows():
        print(
            f"{str(row[bucket_column]):<10}"
            f"{int(row['rows']):>10,}"
            f"{row['event_rate']:>12.4f}"
            f"{row['lift']:>10.2f}x"
            f"{row['mean_return']:>+14.4f}%"
            f"{row['median_return']:>+14.4f}%"
        )


def print_quantile_boundaries(
    df: pd.DataFrame,
    column: str,
) -> None:
    print_header(f"QUANTILE BOUNDARIES: {column}")

    values = df[column].dropna()

    quantiles = values.quantile(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )

    for q, value in quantiles.items():
        print(f"{q:>4.0%}: {value:.6f}")


def print_year_summary(df: pd.DataFrame) -> None:
    print_header("VOLATILITY BY YEAR")

    for year, (start, end) in TEST_PERIODS.items():
        period = df[
            (df["snapshot_date"] >= start)
            & (df["snapshot_date"] <= end)
        ].copy()

        if period.empty:
            continue

        print()
        print(f"--- {year} ---")

        print(f"Rows: {len(period):,}")

        print(
            f"Baseline event rate: "
            f"{period[TARGET_NAME].mean():.4f}"
        )

        print(
            f"Baseline mean return: "
            f"{period['target_return'].mean():+.4f}%"
        )

        print(
            f"Volatility mean: "
            f"{period[VOLATILITY_COLUMN].mean():.6f}"
        )

        print(
            f"Volatility median: "
            f"{period[VOLATILITY_COLUMN].median():.6f}"
        )

        print_bucket_summary(
            period,
            f"{VOLATILITY_COLUMN}_q",
            f"{year}: VOLATILITY QUINTILES",
        )


def print_fi_volatility_matrix(df: pd.DataFrame) -> None:
    print_header("5 × 5 FI SHORTNESS × VOLATILITY")

    clean = df.dropna(
        subset=[
            FI_COLUMN,
            VOLATILITY_COLUMN,
        ]
    ).copy()

    baseline_event = clean[TARGET_NAME].mean()

    matrix = (
        clean.groupby(
            [
                f"{FI_COLUMN}_q",
                f"{VOLATILITY_COLUMN}_q",
            ],
            observed=True,
        )
        .agg(
            rows=(TARGET_NAME, "size"),
            event_rate=(TARGET_NAME, "mean"),
            mean_return=("target_return", "mean"),
            median_return=("target_return", "median"),
        )
        .reset_index()
    )

    for fi_bucket in sorted(
        matrix[f"{FI_COLUMN}_q"].dropna().unique(),
        key=str,
    ):
        print()
        print(f"FI {fi_bucket}")

        print(
            f"{'Vol':<8}"
            f"{'Rows':>10}"
            f"{'Event':>12}"
            f"{'Lift':>10}"
            f"{'Mean ret':>14}"
            f"{'Median ret':>14}"
        )

        print("-" * 70)

        subset = matrix[
            matrix[f"{FI_COLUMN}_q"] == fi_bucket
        ].sort_values(
            f"{VOLATILITY_COLUMN}_q"
        )

        for _, row in subset.iterrows():
            print(
                f"{str(row[f'{VOLATILITY_COLUMN}_q']):<8}"
                f"{int(row['rows']):>10,}"
                f"{row['event_rate']:>12.4f}"
                f"{row['event_rate'] / baseline_event:>10.2f}x"
                f"{row['mean_return']:>+14.4f}%"
                f"{row['median_return']:>+14.4f}%"
            )


def print_extreme_combinations(df: pd.DataFrame) -> None:
    print_header("EXTREME COMBINATIONS")

    clean = df.dropna(
        subset=[
            FI_COLUMN,
            VOLATILITY_COLUMN,
        ]
    ).copy()

    fi_q80 = clean[FI_COLUMN].quantile(0.80)
    vol_q80 = clean[VOLATILITY_COLUMN].quantile(0.80)

    combinations = {
        "All rows": clean,
        "High FI (top 20%)": clean[
            clean[FI_COLUMN] >= fi_q80
        ],
        "High volatility (top 20%)": clean[
            clean[VOLATILITY_COLUMN] >= vol_q80
        ],
        "High FI + high volatility": clean[
            (clean[FI_COLUMN] >= fi_q80)
            & (clean[VOLATILITY_COLUMN] >= vol_q80)
        ],
        "High FI + low volatility": clean[
            (clean[FI_COLUMN] >= fi_q80)
            & (clean[VOLATILITY_COLUMN] < vol_q80)
        ],
        "Low FI + high volatility": clean[
            (clean[FI_COLUMN] < fi_q80)
            & (clean[VOLATILITY_COLUMN] >= vol_q80)
        ],
    }

    baseline_event = clean[TARGET_NAME].mean()

    print(
        f"{'Group':<32}"
        f"{'Rows':>10}"
        f"{'Event':>12}"
        f"{'Lift':>10}"
        f"{'Mean ret':>14}"
        f"{'Median ret':>14}"
    )

    print("-" * 90)

    for name, subset in combinations.items():
        if subset.empty:
            continue

        event_rate = subset[TARGET_NAME].mean()

        print(
            f"{name:<32}"
            f"{len(subset):>10,}"
            f"{event_rate:>12.4f}"
            f"{event_rate / baseline_event:>10.2f}x"
            f"{subset['target_return'].mean():>+14.4f}%"
            f"{subset['target_return'].median():>+14.4f}%"
        )


def print_year_extremes(df: pd.DataFrame) -> None:
    print_header("EXTREME COMBINATIONS BY YEAR")

    for year, (start, end) in TEST_PERIODS.items():
        period = df[
            (df["snapshot_date"] >= start)
            & (df["snapshot_date"] <= end)
        ].copy()

        if period.empty:
            continue

        clean = period.dropna(
            subset=[
                FI_COLUMN,
                VOLATILITY_COLUMN,
            ]
        ).copy()

        fi_q80 = clean[FI_COLUMN].quantile(0.80)
        vol_q80 = clean[VOLATILITY_COLUMN].quantile(0.80)

        groups = {
            "High FI": clean[
                clean[FI_COLUMN] >= fi_q80
            ],
            "High volatility": clean[
                clean[VOLATILITY_COLUMN] >= vol_q80
            ],
            "High FI + high volatility": clean[
                (clean[FI_COLUMN] >= fi_q80)
                & (clean[VOLATILITY_COLUMN] >= vol_q80)
            ],
        }

        baseline_event = clean[TARGET_NAME].mean()

        print()
        print(f"--- {year} ---")

        print(
            f"{'Group':<30}"
            f"{'Rows':>10}"
            f"{'Event':>12}"
            f"{'Lift':>10}"
            f"{'Mean ret':>14}"
        )

        print("-" * 80)

        for name, subset in groups.items():
            if subset.empty:
                continue

            event_rate = subset[TARGET_NAME].mean()

            print(
                f"{name:<30}"
                f"{len(subset):>10,}"
                f"{event_rate:>12.4f}"
                f"{event_rate / baseline_event:>10.2f}x"
                f"{subset['target_return'].mean():>+14.4f}%"
            )


def main() -> None:
    print_header("BLANKDISS VOLATILITY DIAGNOSTICS")

    print("Loading feature data...")

    df = load_features()

    if df.empty:
        raise RuntimeError("No feature data found.")

    required_columns = {
        "snapshot_date",
        "security_key",
        "target_return",
        VOLATILITY_COLUMN,
        FI_COLUMN,
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df.copy()

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"]
    ).dt.date

    df["target_return"] = pd.to_numeric(
        df["target_return"],
        errors="coerce",
    )

    df[VOLATILITY_COLUMN] = pd.to_numeric(
        df[VOLATILITY_COLUMN],
        errors="coerce",
    )

    df[FI_COLUMN] = pd.to_numeric(
        df[FI_COLUMN],
        errors="coerce",
    )

    df[TARGET_NAME] = build_target(df)

    df = df.dropna(
        subset=[
            "target_return",
            "snapshot_date",
        ]
    ).copy()

    print_basic_summary(df)

    print_quantile_boundaries(
        df,
        VOLATILITY_COLUMN,
    )

    print_quantile_boundaries(
        df,
        FI_COLUMN,
    )

    df = add_quantile_bins(
        df,
        VOLATILITY_COLUMN,
    )

    df = add_quantile_bins(
        df,
        FI_COLUMN,
    )

    print_bucket_summary(
        df,
        f"{VOLATILITY_COLUMN}_q",
        "VOLATILITY QUINTILES — ALL DATA",
    )

    print_bucket_summary(
        df,
        f"{FI_COLUMN}_q",
        "FI SHORTNESS QUINTILES — ALL DATA",
    )

    print_year_summary(df)

    print_fi_volatility_matrix(df)

    print_extreme_combinations(df)

    print_year_extremes(df)

    print_header("DONE")

    print(
        "No model training was performed."
    )
    print(
        "No files were modified."
    )


if __name__ == "__main__":
    main()
