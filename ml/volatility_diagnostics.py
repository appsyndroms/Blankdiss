from __future__ import annotations

import pandas as pd

from ml.dataset import load_features


TARGET_NAME = "down_5pct_5d"
RETURN_COLUMN = "forward_return_5d"
VOLATILITY_COLUMN = "price_volatility_20d"
FI_COLUMN = "short_interest_pct"


def print_header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def add_quantiles(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    result = df.copy()

    result[f"{column}_q"] = pd.qcut(
        result[column],
        q=5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        duplicates="drop",
    )

    return result


def print_bucket_summary(
    df: pd.DataFrame,
    bucket_column: str,
    title: str,
) -> None:
    print_header(title)

    clean = df.dropna(
        subset=[
            bucket_column,
            RETURN_COLUMN,
        ]
    )

    if clean.empty:
        print("No data.")
        return

    baseline_event = clean[TARGET_NAME].mean()

    grouped = (
        clean.groupby(
            bucket_column,
            observed=True,
        )
        .agg(
            rows=(TARGET_NAME, "size"),
            event_rate=(TARGET_NAME, "mean"),
            mean_return=(RETURN_COLUMN, "mean"),
            median_return=(RETURN_COLUMN, "median"),
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


def print_matrix(df: pd.DataFrame) -> None:
    print_header(
        "5 x 5 FI SHORTNESS × VOLATILITY"
    )

    clean = df.dropna(
        subset=[
            FI_COLUMN,
            VOLATILITY_COLUMN,
            RETURN_COLUMN,
        ]
    ).copy()

    if clean.empty:
        print("No data.")
        return

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
            mean_return=(RETURN_COLUMN, "mean"),
            median_return=(RETURN_COLUMN, "median"),
        )
        .reset_index()
    )

    for fi_bucket in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        subset = matrix[
            matrix[f"{FI_COLUMN}_q"] == fi_bucket
        ].sort_values(
            f"{VOLATILITY_COLUMN}_q"
        )

        if subset.empty:
            continue

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

        for _, row in subset.iterrows():
            print(
                f"{str(row[f'{VOLATILITY_COLUMN}_q']):<8}"
                f"{int(row['rows']):>10,}"
                f"{row['event_rate']:>12.4f}"
                f"{row['event_rate'] / baseline_event:>10.2f}x"
                f"{row['mean_return']:>+14.4f}%"
                f"{row['median_return']:>+14.4f}%"
            )


def print_extremes(df: pd.DataFrame) -> None:
    print_header("EXTREME COMBINATIONS")

    clean = df.dropna(
        subset=[
            FI_COLUMN,
            VOLATILITY_COLUMN,
            RETURN_COLUMN,
        ]
    ).copy()

    if clean.empty:
        print("No data.")
        return

    fi_q80 = clean[FI_COLUMN].quantile(0.80)
    vol_q80 = clean[VOLATILITY_COLUMN].quantile(0.80)

    groups = {
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

    for name, subset in groups.items():
        if subset.empty:
            continue

        event_rate = subset[TARGET_NAME].mean()

        print(
            f"{name:<32}"
            f"{len(subset):>10,}"
            f"{event_rate:>12.4f}"
            f"{event_rate / baseline_event:>10.2f}x"
            f"{subset[RETURN_COLUMN].mean():>+14.4f}%"
            f"{subset[RETURN_COLUMN].median():>+14.4f}%"
        )


def print_year_analysis(df: pd.DataFrame) -> None:
    print_header("YEAR-BY-YEAR VOLATILITY ANALYSIS")

    for year in [2025, 2026]:
        period = df[
            df["snapshot_date"].dt.year == year
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
            f"{period[RETURN_COLUMN].mean():+.4f}%"
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

        print_bucket_summary(
            period,
            f"{FI_COLUMN}_q",
            f"{year}: FI SHORTNESS QUINTILES",
        )


def main() -> None:
    print_header("BLANKDISS VOLATILITY DIAGNOSTICS")

    print("Loading feature data...")

    df = load_features()

    if df.empty:
        raise RuntimeError("No feature data found.")

    required = {
        "snapshot_date",
        "security_key",
        RETURN_COLUMN,
        VOLATILITY_COLUMN,
        FI_COLUMN,
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df.copy()

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"],
        errors="coerce",
    )

    df[RETURN_COLUMN] = pd.to_numeric(
        df[RETURN_COLUMN],
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

    # down_5pct_5d:
    # forward_return_5d <= -5%.
    df[TARGET_NAME] = (
        df[RETURN_COLUMN] <= -0.05
    ).astype(int)

    df = df.dropna(
        subset=[
            "snapshot_date",
            RETURN_COLUMN,
        ]
    ).copy()

    print()
    print(f"Rows: {len(df):,}")

    print(
        f"Date range: "
        f"{df['snapshot_date'].min().date()} -> "
        f"{df['snapshot_date'].max().date()}"
    )

    print(
        f"Baseline event rate: "
        f"{df[TARGET_NAME].mean():.4f}"
    )

    print(
        f"Baseline mean return: "
        f"{df[RETURN_COLUMN].mean():+.4f}%"
    )

    print_header("VOLATILITY QUANTILE BOUNDARIES")

    print(
        df[VOLATILITY_COLUMN]
        .dropna()
        .quantile(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )
        .to_string()
    )

    print_header("FI QUANTILE BOUNDARIES")

    print(
        df[FI_COLUMN]
        .dropna()
        .quantile(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )
        .to_string()
    )

    df = add_quantiles(
        df,
        VOLATILITY_COLUMN,
    )

    df = add_quantiles(
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

    print_year_analysis(df)

    print_matrix(df)

    print_extremes(df)

    print_header("DIAGNOSTICS COMPLETE")

    print("No additional ML model was trained.")
    print("No repository files were modified.")


if __name__ == "__main__":
    main()
