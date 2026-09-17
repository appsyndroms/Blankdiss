"""
Blankdiss volatility regime diagnostic.

Tests whether FI short-interest information adds predictive value
within different volatility regimes.

The analysis uses the same walk-forward setup and the same economic
target as the main ML benchmark.

For each OOS observation we retain:
    - price_volatility_20d
    - FI features
    - target
    - FI-only model score
    - FI+volatility model score

We then split OOS observations into volatility quintiles and compare
FI-only vs FI+volatility inside each regime.

No repository files are modified by this script.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.config import TARGETS, WALK_FORWARD_WINDOWS
from ml.dataset import (
    load_features,
    prepare_feature_set,
    prepare_ml_data_from_feature_set,
)
from ml.walk_forward import build_models as original_build_models
import ml.walk_forward as walk_forward


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


def build_benchmark_models():
    """
    Build the normal model set but force the RF model to the benchmark
    configuration used elsewhere in Blankdiss.
    """
    models = original_build_models()

    for model_name, model in models.items():
        if model_name == "random_forest":
            model.set_params(
                n_estimators=BENCHMARK_TREES,
                n_jobs=1,
                random_state=42,
            )

    return models


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def build_feature_sets(feature_df: pd.DataFrame):
    """
    Build exactly the two feature sets required for this diagnostic.
    """

    feature_sets = {}

    fi_only = prepare_feature_set(
        feature_df,
        include_price_features=False,
        price_features=None,
    )

    fi_plus_volatility = prepare_feature_set(
        feature_df,
        include_price_features=True,
        price_features={"price_volatility_20d"},
    )

    feature_sets[BASE_FEATURES] = fi_only
    feature_sets[VOL_FEATURES] = fi_plus_volatility

    return feature_sets


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def select_model_name(results: dict) -> str:
    """
    Select the model with the highest validation score.
    """

    return max(
        results,
        key=lambda name: results[name]["validation_score"],
    )


# ---------------------------------------------------------------------------
# AUC helper
# ---------------------------------------------------------------------------


def safe_auc(y_true, scores):
    """
    Calculate ROC AUC when both classes are present.
    """

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, scores)


# ---------------------------------------------------------------------------
# Economic metrics
# ---------------------------------------------------------------------------


def calculate_top_metrics(
    frame: pd.DataFrame,
    score_column: str,
    fraction: float = TOP_FRACTION,
):
    """
    Calculate event rate and return statistics for the highest-scored
    observations.
    """

    if frame.empty:
        return {
            "n": 0,
            "event_rate": np.nan,
            "lift": np.nan,
            "mean_return": np.nan,
            "median_return": np.nan,
        }

    n = max(1, int(np.ceil(len(frame) * fraction)))

    ranked = frame.sort_values(
        score_column,
        ascending=False,
    ).head(n)

    event_rate = ranked["target"].mean()
    baseline_event_rate = frame["target"].mean()

    if baseline_event_rate > 0:
        lift = event_rate / baseline_event_rate
    else:
        lift = np.nan

    return {
        "n": len(ranked),
        "event_rate": event_rate,
        "lift": lift,
        "mean_return": ranked["target_return"].mean(),
        "median_return": ranked["target_return"].median(),
    }


# ---------------------------------------------------------------------------
# Walk-forward execution
# ---------------------------------------------------------------------------


def run_model(
    name: str,
    feature_set: pd.DataFrame,
    target_definition,
):
    """
    Run walk-forward evaluation and retain OOS predictions together with
    the volatility feature and target variables.
    """

    all_oos_rows = []
    window_results = []

    start_total = time.perf_counter()

    for window_index, window in enumerate(
        WALK_FORWARD_WINDOWS,
        start=1,
    ):
        window_start = time.perf_counter()

        result = walk_forward.run_walk_forward(
            feature_set,
            target_definition,
            window,
        )

        selected_model = select_model_name(result["models"])

        selected = result["models"][selected_model]

        validation_score = selected["validation_score"]

        oos = result["oos_predictions"].copy()

        if oos.empty:
            continue

        oos["window"] = window_index
        oos["model"] = name
        oos["selected_model"] = selected_model
        oos["validation_score"] = validation_score

        all_oos_rows.append(oos)

        window_results.append(
            {
                "window": window_index,
                "train_end": window.train_end,
                "validation_end": window.validation_end,
                "test_end": window.test_end,
                "validation_score": validation_score,
                "seconds": time.perf_counter() - window_start,
                "selected_model": selected_model,
            }
        )

    if all_oos_rows:
        combined = pd.concat(
            all_oos_rows,
            ignore_index=True,
        )
    else:
        combined = pd.DataFrame()

    return {
        "name": name,
        "oos": combined,
        "windows": window_results,
        "total_seconds": time.perf_counter() - start_total,
    }


# ---------------------------------------------------------------------------
# Attach source features to OOS predictions
# ---------------------------------------------------------------------------


def attach_source_features(
    oos: pd.DataFrame,
    source_features: pd.DataFrame,
):
    """
    Attach price volatility and the key FI features to OOS rows.

    The join is performed on snapshot_date + security_key.
    """

    required_columns = [
        "snapshot_date",
        "security_key",
        "price_volatility_20d",
        "short_interest_pct",
        "short_interest_delta_pp",
        "short_interest_acceleration_pp",
    ]

    available = [
        column
        for column in required_columns
        if column in source_features.columns
    ]

    source = source_features[available].copy()

    source = source.drop_duplicates(
        subset=["snapshot_date", "security_key"],
        keep="last",
    )

    result = oos.merge(
        source,
        on=["snapshot_date", "security_key"],
        how="left",
        suffixes=("", "_source"),
    )

    return result


# ---------------------------------------------------------------------------
# Volatility regimes
# ---------------------------------------------------------------------------


def add_volatility_regimes(frame: pd.DataFrame):
    """
    Assign volatility quintiles using the OOS population only.

    This prevents future feature data outside the evaluated OOS period
    from influencing the diagnostic thresholds.
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
# Regime report
# ---------------------------------------------------------------------------


def print_regime_analysis(
    fi_oos: pd.DataFrame,
    vol_oos: pd.DataFrame,
):
    """
    Compare FI-only and FI+volatility within each volatility regime.
    """

    merged = fi_oos.merge(
        vol_oos[
            [
                "snapshot_date",
                "security_key",
                "score",
                "selected_model",
            ]
        ].rename(
            columns={
                "score": "vol_score",
                "selected_model": "vol_selected_model",
            }
        ),
        on=["snapshot_date", "security_key"],
        how="inner",
    )

    merged = merged.rename(
        columns={
            "score": "fi_score",
        }
    )

    if "price_volatility_20d" not in merged.columns:
        raise RuntimeError(
            "price_volatility_20d missing from OOS data."
        )

    merged = add_volatility_regimes(merged)

    print()
    print("=" * 100)
    print("FI SIGNAL WITHIN VOLATILITY REGIMES")
    print("=" * 100)

    for regime in merged["volatility_regime"].dropna().unique():
        subset = merged[
            merged["volatility_regime"] == regime
        ].copy()

        if subset.empty:
            continue

        volatility_min = subset["price_volatility_20d"].min()
        volatility_max = subset["price_volatility_20d"].max()

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
        print(f"{regime}")
        print(
            f"  Rows: {len(subset):,}"
        )
        print(
            f"  Volatility range: "
            f"{volatility_min:.6f} -> {volatility_max:.6f}"
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
            f"    Top 1% event: "
            f"{fi_top['event_rate']:.4f} "
            f"lift={fi_top['lift']:.2f}x "
            f"mean={fi_top['mean_return']:.4%}"
        )

        print()
        print("  FI + volatility")
        print(
            f"    AUC: {vol_auc:.6f}"
        )
        print(
            f"    Top 1% event: "
            f"{vol_top['event_rate']:.4f} "
            f"lift={vol_top['lift']:.2f}x "
            f"mean={vol_top['mean_return']:.4%}"
        )

        print()
        print(
            f"  AUC difference: "
            f"{vol_auc - fi_auc:+.6f}"
        )

    return merged


# ---------------------------------------------------------------------------
# FI quintile inside volatility regimes
# ---------------------------------------------------------------------------


def print_fi_within_volatility_matrix(frame: pd.DataFrame):
    """
    Build a 5x5 matrix:

        rows    = volatility quintile
        columns = FI quintile

    The value is the observed downside-event rate.

    This directly tests whether FI still separates risk after volatility
    has already defined the regime.
    """

    data = frame.dropna(
        subset=[
            "price_volatility_20d",
            "short_interest_pct",
            "target",
        ]
    ).copy()

    data["vol_q"] = pd.qcut(
        data["price_volatility_20d"],
        q=5,
        labels=["V1", "V2", "V3", "V4", "V5"],
        duplicates="drop",
    )

    data["fi_q"] = pd.qcut(
        data["short_interest_pct"],
        q=5,
        labels=["F1", "F2", "F3", "F4", "F5"],
        duplicates="drop",
    )

    matrix = (
        data.pivot_table(
            index="vol_q",
            columns="fi_q",
            values="target",
            aggfunc="mean",
        )
        .reindex(
            index=["V1", "V2", "V3", "V4", "V5"]
        )
        .reindex(
            columns=["F1", "F2", "F3", "F4", "F5"]
        )
    )

    print()
    print("=" * 100)
    print("DOWNSIDE EVENT RATE: VOLATILITY × FI")
    print("=" * 100)
    print(
        "Rows = volatility quintile, "
        "columns = short-interest quintile"
    )
    print()

    print(matrix.to_string(float_format=lambda x: f"{x:.4f}"))


# ---------------------------------------------------------------------------
# Year-by-year analysis
# ---------------------------------------------------------------------------


def print_year_analysis(frame: pd.DataFrame):
    """
    Repeat the regime analysis separately for each OOS calendar year.
    """

    frame = frame.copy()

    frame["year"] = pd.to_datetime(
        frame["snapshot_date"]
    ).dt.year

    print()
    print("=" * 100)
    print("YEAR-BY-YEAR VOLATILITY REGIMES")
    print("=" * 100)

    for year in sorted(frame["year"].dropna().unique()):
        subset = frame[
            frame["year"] == year
        ].copy()

        print()
        print(f"YEAR {int(year)}")
        print("-" * 100)

        if subset.empty:
            continue

        for regime in subset["volatility_regime"].dropna().unique():
            regime_data = subset[
                subset["volatility_regime"] == regime
            ]

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
                f"event={regime_data['target'].mean():.4f} "
                f"FI_AUC={fi_auc:.4f} "
                f"FI+VOL_AUC={vol_auc:.4f} "
                f"delta={vol_auc - fi_auc:+.4f}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 100)
    print("BLANKDISS VOLATILITY REGIME DIAGNOSTIC")
    print("=" * 100)
    print(f"RF trees: {BENCHMARK_TREES}")
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
        f"Feature rows: {len(feature_df):,}"
    )

    target_definition = TARGETS[ECONOMIC_TARGET]

    print()
    print("Building feature sets...")

    feature_sets = build_feature_sets(
        feature_df
    )

    # Force benchmark model configuration.
    walk_forward.build_models = build_benchmark_models

    print()
    print("=" * 100)
    print("RUNNING FI-ONLY")
    print("=" * 100)

    fi_result = run_model(
        BASE_FEATURES,
        feature_sets[BASE_FEATURES],
        target_definition,
    )

    print()
    print("=" * 100)
    print("RUNNING FI + VOLATILITY")
    print("=" * 100)

    vol_result = run_model(
        VOL_FEATURES,
        feature_sets[VOL_FEATURES],
        target_definition,
    )

    print()
    print("=" * 100)
    print("ATTACHING SOURCE FEATURES")
    print("=" * 100)

    fi_oos = attach_source_features(
        fi_result["oos"],
        feature_sets[BASE_FEATURES],
    )

    vol_oos = attach_source_features(
        vol_result["oos"],
        feature_sets[VOL_FEATURES],
    )

    # The source feature is already part of the volatility feature set,
    # but explicitly make sure it exists in both OOS frames.
    if "price_volatility_20d" not in fi_oos.columns:
        fi_oos = attach_source_features(
            fi_oos,
            feature_df,
        )

    if "price_volatility_20d" not in vol_oos.columns:
        vol_oos = attach_source_features(
            vol_oos,
            feature_df,
        )

    print(
        f"FI-only OOS rows: "
        f"{len(fi_oos):,}"
    )
    print(
        f"FI+volatility OOS rows: "
        f"{len(vol_oos):,}"
    )

    print()
    print("=" * 100)
    print("BUILDING VOLATILITY REGIMES")
    print("=" * 100)

    combined = fi_oos.merge(
        vol_oos[
            [
                "snapshot_date",
                "security_key",
                "score",
            ]
        ].rename(
            columns={
                "score": "vol_score",
            }
        ),
        on=["snapshot_date", "security_key"],
        how="inner",
    )

    combined = combined.rename(
        columns={
            "score": "fi_score",
        }
    )

    combined = add_volatility_regimes(
        combined
    )

    print(
        f"Matched OOS rows: "
        f"{len(combined):,}"
    )

    print_regime_analysis(
        fi_oos,
        vol_oos,
    )

    print_fi_within_volatility_matrix(
        combined
    )

    print_year_analysis(
        combined
    )

    print()
    print("=" * 100)
    print("OVERALL CONCLUSION DATA")
    print("=" * 100)

    fi_auc = safe_auc(
        combined["target"],
        combined["fi_score"],
    )

    vol_auc = safe_auc(
        combined["target"],
        combined["vol_score"],
    )

    print(
        f"FI-only overall OOS AUC: "
        f"{fi_auc:.6f}"
    )

    print(
        f"FI + volatility overall OOS AUC: "
        f"{vol_auc:.6f}"
    )

    print(
        f"AUC difference: "
        f"{vol_auc - fi_auc:+.6f}"
    )

    print()
    print("=" * 100)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
