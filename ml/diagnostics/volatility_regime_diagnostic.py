"""
Blankdiss volatility regime diagnostic.

Tests whether FI short-interest information adds predictive value
within different volatility regimes.

The analysis uses the same walk-forward setup and economic target
as the main ML benchmark.

For each OOS observation we retain:
    - price_volatility_20d
    - short_interest_pct
    - FI-only model score
    - FI+volatility model score
    - target
    - target_return

The analysis then compares:
    1. FI-only vs FI+volatility AUC within volatility quintiles.
    2. Top-1% economic ranking within volatility quintiles.
    3. A 5x5 volatility x FI event-rate matrix.
    4. The same regime analysis separately by OOS year.

No repository files are modified by this script.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import ml.walk_forward as walk_forward
from ml.config import TARGETS, WALK_FORWARD_WINDOWS
from ml.dataset import (
    load_features,
    prepare_feature_set,
    prepare_ml_data_from_feature_set,
)
from ml.models import build_models as original_build_models


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BENCHMARK_TREES = 100
ECONOMIC_TARGET = "down_5pct_5d"

TOP_FRACTION = 0.01
VOLATILITY_QUANTILES = 5

BASE_FEATURES = "fi_only"
VOL_FEATURES = "fi_plus_volatility_20d"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


def build_benchmark_models(
    random_state: int,
    task: str = "classification",
):
    """
    Build the normal model set while forcing the RF benchmark settings.
    """

    models = original_build_models(
        random_state,
        task=task,
    )

    if task == "classification":
        random_forest = models.get(
            "random_forest"
        )

        if random_forest is not None:
            random_forest.set_params(
                model__n_estimators=BENCHMARK_TREES,
                model__n_jobs=1,
            )

    return models


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def build_feature_sets(
    feature_df: pd.DataFrame,
):
    """
    Build the two feature sets required by the diagnostic.
    """

    fi_only_data, fi_only_columns = (
        prepare_feature_set(
            feature_df,
            include_price_features=False,
            price_features=None,
        )
    )

    volatility_data, volatility_columns = (
        prepare_feature_set(
            feature_df,
            include_price_features=True,
            price_features={
                "price_volatility_20d"
            },
        )
    )

    return {
        BASE_FEATURES: (
            fi_only_data,
            fi_only_columns,
        ),
        VOL_FEATURES: (
            volatility_data,
            volatility_columns,
        ),
    }


# ---------------------------------------------------------------------------
# Walk-forward model execution
# ---------------------------------------------------------------------------


def run_feature_set(
    name: str,
    feature_data: pd.DataFrame,
    feature_columns: list[str],
    target_definition,
):
    """
    Run the actual Blankdiss walk-forward engine for one feature set.

    The selected model is determined independently in every
    walk-forward window by validation score.
    """

    all_oos_rows: list[pd.DataFrame] = []
    window_results: list[dict] = []

    total_start = time.perf_counter()

    (
        ml_data,
        y,
        feature_columns,
    ) = prepare_ml_data_from_feature_set(
        feature_data,
        feature_columns,
        target_definition,
    )

    for window_index, window in enumerate(
        WALK_FORWARD_WINDOWS,
        start=1,
    ):
        window_start = time.perf_counter()

        (
            results,
            oos_predictions,
            timing,
        ) = walk_forward.train_window(
            ml_data,
            y,
            feature_columns,
            window,
            task=target_definition.task,
            direction=target_definition.direction,
        )

        if not results or not oos_predictions:
            continue

        selected_results = [
            result
            for result in results
            if result["selected_for_oos"]
        ]

        if not selected_results:
            continue

        selected_result = selected_results[0]

        oos = pd.DataFrame(
            oos_predictions
        )

        oos["feature_set"] = name
        oos["window_index"] = window_index
        oos["selected_model"] = (
            selected_result["model"]
        )

        all_oos_rows.append(oos)

        window_results.append(
            {
                "window": window_index,
                "train_end": window.train_end,
                "validation_end": (
                    window.validation_end
                ),
                "test_end": window.test_end,
                "validation_score": (
                    selected_result[
                        "validation_score"
                    ]
                ),
                "selected_model": (
                    selected_result["model"]
                ),
                "seconds": (
                    time.perf_counter()
                    - window_start
                ),
                "fit_seconds": timing[
                    "fit_seconds"
                ],
                "oos_prediction_seconds": timing[
                    "oos_prediction_seconds"
                ],
            }
        )

    if all_oos_rows:
        oos = pd.concat(
            all_oos_rows,
            ignore_index=True,
        )
    else:
        oos = pd.DataFrame()

    return {
        "name": name,
        "oos": oos,
        "windows": window_results,
        "total_seconds": (
            time.perf_counter()
            - total_start
        ),
    }


# ---------------------------------------------------------------------------
# Source feature attachment
# ---------------------------------------------------------------------------


def attach_source_features(
    oos: pd.DataFrame,
    source_features: pd.DataFrame,
):
    """
    Attach the raw FI and volatility features to OOS predictions.
    """

    required_columns = [
        "snapshot_date",
        "security_key",
        "price_volatility_20d",
        "short_interest_pct",
        "short_interest_delta_pp",
        "short_interest_acceleration_pp",
    ]

    available_columns = [
        column
        for column in required_columns
        if column in source_features.columns
    ]

    source = source_features[
        available_columns
    ].copy()

    source["snapshot_date"] = pd.to_datetime(
        source["snapshot_date"],
        errors="coerce",
    )

    source = source.drop_duplicates(
        subset=[
            "snapshot_date",
            "security_key",
        ],
        keep="last",
    )

    result = oos.copy()

    result["snapshot_date"] = pd.to_datetime(
        result["snapshot_date"],
        errors="coerce",
    )

    result["security_key"] = (
        result["security_key"].astype(str)
    )

    source["security_key"] = (
        source["security_key"].astype(str)
    )

    return result.merge(
        source,
        on=[
            "snapshot_date",
            "security_key",
        ],
        how="left",
        suffixes=("", "_source"),
    )


# ---------------------------------------------------------------------------
# Volatility regimes
# ---------------------------------------------------------------------------


def add_volatility_regimes(
    frame: pd.DataFrame,
):
    """
    Assign volatility quintiles using the complete OOS population.

    The thresholds are derived only from OOS observations.
    """

    result = frame.copy()

    result["volatility_regime"] = pd.qcut(
        result["price_volatility_20d"],
        q=VOLATILITY_QUANTILES,
        labels=[
            "Q1_low",
            "Q2",
            "Q3",
            "Q4",
            "Q5_high",
        ],
        duplicates="drop",
    )

    return result


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------


def safe_auc(
    y_true,
    scores,
):
    """
    Calculate ROC AUC when both classes are present.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    valid = (
        np.isfinite(y_true)
        & np.isfinite(scores)
    )

    y_true = y_true[valid]
    scores = scores[valid]

    if len(y_true) == 0:
        return np.nan

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y_true,
            scores,
        )
    )


# ---------------------------------------------------------------------------
# Economic metrics
# ---------------------------------------------------------------------------


def calculate_top_metrics(
    frame: pd.DataFrame,
    score_column: str,
):
    """
    Calculate economic metrics for the highest-scored 1%.
    """

    if frame.empty:
        return {
            "n": 0,
            "event_rate": np.nan,
            "lift": np.nan,
            "mean_return": np.nan,
            "median_return": np.nan,
        }

    n = max(
        1,
        int(
            np.ceil(
                len(frame)
                * TOP_FRACTION
            )
        ),
    )

    ranked = (
        frame
        .sort_values(
            score_column,
            ascending=False,
        )
        .head(n)
    )

    event_rate = ranked[
        "target"
    ].mean()

    baseline_event_rate = frame[
        "target"
    ].mean()

    if baseline_event_rate > 0:
        lift = (
            event_rate
            / baseline_event_rate
        )
    else:
        lift = np.nan

    return {
        "n": len(ranked),
        "event_rate": event_rate,
        "lift": lift,
        "mean_return": ranked[
            "target_return"
        ].mean(),
        "median_return": ranked[
            "target_return"
        ].median(),
    }


# ---------------------------------------------------------------------------
# Regime analysis
# ---------------------------------------------------------------------------


def build_comparison_frame(
    fi_oos: pd.DataFrame,
    volatility_oos: pd.DataFrame,
):
    """
    Match FI-only and FI+volatility predictions on the same OOS rows.
    """

    fi = fi_oos[
        [
            "snapshot_date",
            "security_key",
            "target_return",
            "prediction",
            "score",
            "window_index",
            "selected_model",
        ]
    ].rename(
        columns={
            "prediction": "fi_prediction",
            "score": "fi_score",
            "selected_model": (
                "fi_selected_model"
            ),
        }
    )

    volatility = volatility_oos[
        [
            "snapshot_date",
            "security_key",
            "prediction",
            "score",
            "window_index",
            "selected_model",
        ]
    ].rename(
        columns={
            "prediction": (
                "vol_prediction"
            ),
            "score": "vol_score",
            "selected_model": (
                "vol_selected_model"
            ),
        }
    )

    merged = fi.merge(
        volatility,
        on=[
            "snapshot_date",
            "security_key",
            "window_index",
        ],
        how="inner",
    )

    merged["target"] = (
        merged["target_return"] <= -0.05
    ).astype(int)

    return merged


def print_regime_analysis(
    comparison: pd.DataFrame,
):
    """
    Compare FI-only and FI+volatility within each volatility regime.
    """

    data = comparison.dropna(
        subset=[
            "price_volatility_20d",
            "fi_score",
            "vol_score",
            "target",
            "target_return",
        ]
    ).copy()

    data = add_volatility_regimes(
        data
    )

    print()
    print("=" * 100)
    print("FI SIGNAL WITHIN VOLATILITY REGIMES")
    print("=" * 100)

    for regime in [
        "Q1_low",
        "Q2",
        "Q3",
        "Q4",
        "Q5_high",
    ]:
        subset = data[
            data["volatility_regime"]
            == regime
        ]

        if subset.empty:
            continue

        fi_auc = safe_auc(
            subset["target"],
            subset["fi_score"],
        )

        vol_auc = safe_auc(
            subset["target"],
            subset["vol_score"],
        )

        fi_top = calculate_top_metrics(
            subset,
            "fi_score",
        )

        vol_top = calculate_top_metrics(
            subset,
            "vol_score",
        )

        print()
        print(regime)
        print(
            f"  Rows: {len(subset):,}"
        )
        print(
            f"  Volatility: "
            f"{subset['price_volatility_20d'].min():.6f}"
            f" -> "
            f"{subset['price_volatility_20d'].max():.6f}"
        )
        print(
            f"  Event rate: "
            f"{subset['target'].mean():.4f}"
        )

        print()
        print("  FI-only")
        print(
            f"    AUC: {fi_auc:.6f}"
        )
        print(
            f"    Top 1%: "
            f"event={fi_top['event_rate']:.4f} "
            f"lift={fi_top['lift']:.2f}x "
            f"mean={fi_top['mean_return']:.4%} "
            f"median={fi_top['median_return']:.4%}"
        )

        print()
        print("  FI + volatility")
        print(
            f"    AUC: {vol_auc:.6f}"
        )
        print(
            f"    Top 1%: "
            f"event={vol_top['event_rate']:.4f} "
            f"lift={vol_top['lift']:.2f}x "
            f"mean={vol_top['mean_return']:.4%} "
            f"median={vol_top['median_return']:.4%}"
        )

        print()
        print(
            f"  AUC difference: "
            f"{vol_auc - fi_auc:+.6f}"
        )


# ---------------------------------------------------------------------------
# FI x volatility matrix
# ---------------------------------------------------------------------------


def print_fi_within_volatility_matrix(
    comparison: pd.DataFrame,
):
    """
    Print a 5x5 event-rate matrix.

    Rows:
        volatility quintiles

    Columns:
        short-interest quintiles
    """

    data = comparison.dropna(
        subset=[
            "price_volatility_20d",
            "short_interest_pct",
            "target",
        ]
    ).copy()

    data["vol_q"] = pd.qcut(
        data["price_volatility_20d"],
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

    data["fi_q"] = pd.qcut(
        data["short_interest_pct"],
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

    matrix = data.pivot_table(
        index="vol_q",
        columns="fi_q",
        values="target",
        aggfunc="mean",
    )

    matrix = matrix.reindex(
        index=[
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
        ],
        columns=[
            "F1",
            "F2",
            "F3",
            "F4",
            "F5",
        ],
    )

    print()
    print("=" * 100)
    print("DOWNSIDE EVENT RATE: VOLATILITY x FI")
    print("=" * 100)
    print(
        "Rows = volatility quintile, "
        "columns = short-interest quintile"
    )
    print()

    print(
        matrix.to_string(
            float_format=lambda value:
                f"{value:.4f}"
        )
    )


# ---------------------------------------------------------------------------
# Year-by-year analysis
# ---------------------------------------------------------------------------


def print_year_analysis(
    comparison: pd.DataFrame,
):
    """
    Repeat the volatility-regime analysis independently for each
    OOS calendar year.
    """

    data = comparison.dropna(
        subset=[
            "price_volatility_20d",
            "fi_score",
            "vol_score",
            "target",
        ]
    ).copy()

    data["year"] = pd.to_datetime(
        data["snapshot_date"]
    ).dt.year

    print()
    print("=" * 100)
    print("YEAR-BY-YEAR VOLATILITY REGIMES")
    print("=" * 100)

    for year in sorted(
        data["year"].unique()
    ):
        year_data = data[
            data["year"] == year
        ].copy()

        year_data = add_volatility_regimes(
            year_data
        )

        print()
        print(
            f"YEAR {int(year)}"
        )
        print("-" * 100)

        for regime in [
            "Q1_low",
            "Q2",
            "Q3",
            "Q4",
            "Q5_high",
        ]:
            regime_data = year_data[
                year_data[
                    "volatility_regime"
                ] == regime
            ]

            if regime_data.empty:
                continue

            fi_auc = safe_auc(
                regime_data["target"],
                regime_data["fi_score"],
            )

            vol_auc = safe_auc(
                regime_data["target"],
                regime_data["vol_score"],
            )

            print(
                f"{regime:8s} "
                f"n={len(regime_data):6,d} "
                f"event="
                f"{regime_data['target'].mean():.4f} "
                f"FI_AUC={fi_auc:.4f} "
                f"FI+VOL_AUC={vol_auc:.4f} "
                f"delta="
                f"{vol_auc - fi_auc:+.4f}"
            )


# ---------------------------------------------------------------------------
# Window analysis
# ---------------------------------------------------------------------------


def print_window_results(
    result: dict,
):
    """
    Print which model was selected in each walk-forward window.
    """

    print()
    print("=" * 100)
    print(
        f"WINDOW RESULTS: "
        f"{result['name']}"
    )
    print("=" * 100)

    for window in result["windows"]:
        print()
        print(
            f"Window {window['window']}"
        )
        print(
            f"  Train <= "
            f"{window['train_end']}"
        )
        print(
            f"  Validation <= "
            f"{window['validation_end']}"
        )
        print(
            f"  Test <= "
            f"{window['test_end']}"
        )
        print(
            f"  Selected model: "
            f"{window['selected_model']}"
        )
        print(
            f"  Validation score: "
            f"{window['validation_score']:.6f}"
        )
        print(
            f"  Time: "
            f"{window['seconds']:.2f}s"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 100)
    print(
        "BLANKDISS VOLATILITY REGIME DIAGNOSTIC"
    )
    print("=" * 100)
    print(
        f"RF trees: {BENCHMARK_TREES}"
    )
    print("RF n_jobs: 1")
    print(
        f"Walk-forward windows: "
        f"{len(WALK_FORWARD_WINDOWS)}"
    )
    print(
        f"Economic target: "
        f"{ECONOMIC_TARGET}"
    )

    print()
    print("Loading feature data...")

    feature_df = load_features()

    print(
        f"Feature rows: "
        f"{len(feature_df):,}"
    )

    target_definition = next(
        target
        for target in TARGETS
        if target.name == ECONOMIC_TARGET
    )

    print()
    print("Building feature sets...")

    feature_sets = build_feature_sets(
        feature_df
    )

    # train_window() imports build_models into the
    # ml.walk_forward module namespace, so patch that
    # symbol rather than trying to call a nonexistent
    # run_walk_forward() API.
    walk_forward.build_models = (
        build_benchmark_models
    )

    print()
    print("=" * 100)
    print("RUNNING FI-ONLY")
    print("=" * 100)

    fi_data, fi_columns = feature_sets[
        BASE_FEATURES
    ]

    fi_result = run_feature_set(
        BASE_FEATURES,
        fi_data,
        fi_columns,
        target_definition,
    )

    print_window_results(
        fi_result
    )

    print()
    print("=" * 100)
    print("RUNNING FI + VOLATILITY")
    print("=" * 100)

    volatility_data, volatility_columns = (
        feature_sets[VOL_FEATURES]
    )

    volatility_result = run_feature_set(
        VOL_FEATURES,
        volatility_data,
        volatility_columns,
        target_definition,
    )

    print_window_results(
        volatility_result
    )

    # Attach raw source features.
    fi_oos = attach_source_features(
        fi_result["oos"],
        feature_df,
    )

    volatility_oos = attach_source_features(
        volatility_result["oos"],
        feature_df,
    )

    comparison = build_comparison_frame(
        fi_oos,
        volatility_oos,
    )

    print()
    print(
        f"Matched OOS rows: "
        f"{len(comparison):,}"
    )

    print_regime_analysis(
        comparison
    )

    print_fi_within_volatility_matrix(
        comparison
    )

    print_year_analysis(
        comparison
    )

    print()
    print("=" * 100)
    print("OVERALL OOS COMPARISON")
    print("=" * 100)

    fi_auc = safe_auc(
        comparison["target"],
        comparison["fi_score"],
    )

    volatility_auc = safe_auc(
        comparison["target"],
        comparison["vol_score"],
    )

    print(
        f"FI-only OOS AUC: "
        f"{fi_auc:.6f}"
    )

    print(
        f"FI + volatility OOS AUC: "
        f"{volatility_auc:.6f}"
    )

    print(
        f"AUC difference: "
        f"{volatility_auc - fi_auc:+.6f}"
    )

    print()
    print("=" * 100)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
