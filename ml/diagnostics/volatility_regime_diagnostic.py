"""
Blankdiss volatility regime diagnostic.
Tests whether FI short-interest information behaves differently
across volatility regimes.
The analysis uses the same walk-forward setup and economic target
as the main ML benchmark.
For each OOS observation we retain:
    - price_volatility_20d
    - short_interest_pct
    - FI-only model score
    - FI+volatility model score
    - target
    - target_return
The diagnostic compares:
    1. FI-only vs FI+volatility within volatility quintiles.
    2. Top-0.1%, 0.5%, 1%, 2% and 5% economic ranking.
    3. A volatility x FI event-rate matrix.
    4. The same analysis separately by OOS year.
    5. Low / middle / high volatility regimes.
    6. FI signal strength within each regime.
    7. Economic performance of FI within each regime.
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
# ============================================================================
# Configuration
# ============================================================================
BENCHMARK_TREES = 100
ECONOMIC_TARGET = "down_5pct_5d"
VOLATILITY_QUANTILES = 5
BASE_FEATURES = "fi_only"
VOL_FEATURES = "fi_plus_volatility_20d"
ECONOMIC_FRACTIONS = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
)
REGIMES = (
    "LOW",
    "MID",
    "HIGH",
)
# ============================================================================
# Model configuration
# ============================================================================
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
        random_forest = models.get("random_forest")
        if random_forest is not None:
            random_forest.set_params(
                model__n_estimators=BENCHMARK_TREES,
                model__n_jobs=1,
            )
    return models
# ============================================================================
# Feature construction
# ============================================================================
def build_feature_sets(
    feature_df: pd.DataFrame,
):
    """
    Build the two feature sets required by the diagnostic.
    """
    fi_only_data, fi_only_columns = prepare_feature_set(
        feature_df,
        include_price_features=False,
        price_features=None,
    )
    volatility_data, volatility_columns = prepare_feature_set(
        feature_df,
        include_price_features=True,
        price_features={
            "price_volatility_20d",
        },
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
# ============================================================================
# Walk-forward execution
# ============================================================================
def run_feature_set(
    name: str,
    feature_data: pd.DataFrame,
    feature_columns: list[str],
    target_definition,
):
    """
    Run the normal Blankdiss walk-forward engine for one feature set.
    Model selection is performed independently inside each
    walk-forward window.
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
        all_oos_rows.append(
            oos
        )
        window_results.append(
            {
                "window": window_index,
                "train_end": window.train_end,
                "validation_end": window.validation_end,
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
# ============================================================================
# Source feature attachment
# ============================================================================
def attach_source_features(
    oos: pd.DataFrame,
    source_features: pd.DataFrame,
):
    """
    Attach raw source features to OOS predictions.
    Only columns actually present in source_features are requested.
    This keeps the function robust when a feature set intentionally
    excludes a price feature.
    """
    desired_columns = [
        "snapshot_date",
        "security_key",
        "price_volatility_20d",
        "short_interest_pct",
        "short_interest_delta_pp",
        "short_interest_acceleration_pp",
    ]
    available_columns = [
        column
        for column in desired_columns
        if column in source_features.columns
    ]
    source = source_features[
        available_columns
    ].copy()
    source["snapshot_date"] = pd.to_datetime(
        source["snapshot_date"],
        errors="coerce",
    )
    source["security_key"] = (
        source["security_key"].astype(str)
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
    return result.merge(
        source,
        on=[
            "snapshot_date",
            "security_key",
        ],
        how="left",
        suffixes=(
            "",
            "_source",
        ),
    )
# ============================================================================
# AUC helpers
# ============================================================================
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
# ============================================================================
# Economic metrics
# ============================================================================
def calculate_top_metrics(
    frame: pd.DataFrame,
    score_column: str,
    fraction: float,
):
    """
    Calculate economic metrics for the highest-scored observations.
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
                * fraction
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
def print_economic_metrics(
    frame: pd.DataFrame,
    score_column: str,
    indent: str = "    ",
):
    """
    Print economic ranking metrics for all configured fractions.
    """
    for fraction in ECONOMIC_FRACTIONS:
        metrics = calculate_top_metrics(
            frame,
            score_column,
            fraction,
        )
        print(
            f"{indent}"
            f"Top {fraction:.1%}: "
            f"n={metrics['n']:,} "
            f"event={metrics['event_rate']:.4f} "
            f"lift={metrics['lift']:.2f}x "
            f"mean={metrics['mean_return']:.4%} "
            f"median={metrics['median_return']:.4%}"
        )
# ============================================================================
# Comparison frame
# ============================================================================
def build_comparison_frame(
    fi_oos: pd.DataFrame,
    volatility_oos: pd.DataFrame,
):
    """
    Match FI-only and FI+volatility predictions on identical OOS rows.
    Important:
        FI-only intentionally does not contain price_volatility_20d.
    Therefore raw volatility and FI source features are taken from
    volatility_oos, while the FI-only prediction remains separate.
    The target return is checked to ensure both model runs refer to
    the same economic target.
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
            "selected_model": "fi_selected_model",
        }
    )
    volatility = volatility_oos[
        [
            "snapshot_date",
            "security_key",
            "target_return",
            "price_volatility_20d",
            "short_interest_pct",
            "prediction",
            "score",
            "window_index",
            "selected_model",
        ]
    ].rename(
        columns={
            "target_return": "vol_target_return",
            "prediction": "vol_prediction",
            "score": "vol_score",
            "selected_model": "vol_selected_model",
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
    target_mismatch = (
        merged["target_return"]
        - merged["vol_target_return"]
    ).abs() > 1e-10
    mismatch_count = int(
        target_mismatch.sum()
    )
    if mismatch_count:
        raise ValueError(
            "Target return mismatch between "
            "FI-only and FI+volatility OOS rows: "
            f"{mismatch_count:,} rows"
        )
    merged["target"] = (
        merged["target_return"] <= -0.05
    ).astype(int)
    return merged
# ============================================================================
# Volatility quintile analysis
# ============================================================================
def add_volatility_quintiles(
    frame: pd.DataFrame,
):
    """
    Assign five volatility quintiles.
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
def print_regime_analysis(
    comparison: pd.DataFrame,
):
    """
    Compare FI-only and FI+volatility within each volatility quintile.
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
    data = add_volatility_quintiles(
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
        print_economic_metrics(
            subset,
            "fi_score",
        )
        print()
        print("  FI + volatility")
        print(
            f"    AUC: {vol_auc:.6f}"
        )
        print_economic_metrics(
            subset,
            "vol_score",
        )
        print()
        print(
            f"  AUC difference: "
            f"{vol_auc - fi_auc:+.6f}"
        )
# ============================================================================
# Three-regime analysis
# ============================================================================
def add_three_regimes(
    frame: pd.DataFrame,
):
    """
    Divide observations into three equally sized volatility regimes.
    LOW:
        bottom third
    MID:
        middle third
    HIGH:
        top third
    """
    result = frame.copy()
    result["three_vol_regime"] = pd.qcut(
        result["price_volatility_20d"],
        q=3,
        labels=[
            "LOW",
            "MID",
            "HIGH",
        ],
        duplicates="drop",
    )
    return result
def print_three_regime_analysis(
    comparison: pd.DataFrame,
):
    """
    Compare FI-only and FI+volatility in LOW/MID/HIGH volatility.
    """
    data = comparison.dropna(
        subset=[
            "price_volatility_20d",
            "short_interest_pct",
            "fi_score",
            "vol_score",
            "target",
            "target_return",
        ]
    ).copy()
    data = add_three_regimes(
        data
    )
    print()
    print("=" * 100)
    print("LOW / MID / HIGH VOLATILITY REGIME ANALYSIS")
    print("=" * 100)
    for regime in REGIMES:
        subset = data[
            data["three_vol_regime"]
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
        print()
        print(regime)
        print("-" * 100)
        print(
            f"Rows: {len(subset):,}"
        )
        print(
            f"Volatility: "
            f"{subset['price_volatility_20d'].min():.6f}"
            f" -> "
            f"{subset['price_volatility_20d'].max():.6f}"
        )
        print(
            f"Event rate: "
            f"{subset['target'].mean():.4f}"
        )
        print()
        print("FI-only")
        print(
            f"  AUC: {fi_auc:.6f}"
        )
        print_economic_metrics(
            subset,
            "fi_score",
            indent="  ",
        )
        print()
        print("FI + volatility")
        print(
            f"  AUC: {vol_auc:.6f}"
        )
        print_economic_metrics(
            subset,
            "vol_score",
            indent="  ",
        )
        print()
        print(
            f"AUC difference: "
            f"{vol_auc - fi_auc:+.6f}"
        )
# ============================================================================
# FI signal strength
# ============================================================================
def print_fi_signal_strength(
    comparison: pd.DataFrame,
):
    """
    Measure FI signal strength inside each broad volatility regime.
    Two measures are shown:
        1. FI model AUC.
        2. Raw short_interest_pct AUC.
    This separates the predictive information in the trained FI model
    from the raw short-interest level.
    """
    data = comparison.dropna(
        subset=[
            "price_volatility_20d",
            "short_interest_pct",
            "fi_score",
            "target",
        ]
    ).copy()
    data = add_three_regimes(
        data
    )
    print()
    print("=" * 100)
    print("FI SIGNAL STRENGTH BY VOLATILITY REGIME")
    print("=" * 100)
    for regime in REGIMES:
        subset = data[
            data["three_vol_regime"]
            == regime
        ]
        if subset.empty:
            continue
        fi_auc = safe_auc(
            subset["target"],
            subset["fi_score"],
        )
        fi_raw_auc = safe_auc(
            subset["target"],
            subset["short_interest_pct"],
        )
        print()
        print(regime)
        print(
            f"  Rows: {len(subset):,}"
        )
        print(
            f"  Event rate: "
            f"{subset['target'].mean():.4f}"
        )
        print(
            f"  FI model AUC: "
            f"{fi_auc:.6f}"
        )
        print(
            f"  Raw short_interest_pct AUC: "
            f"{fi_raw_auc:.6f}"
        )
# ============================================================================
# Volatility x FI matrix
# ============================================================================
def print_fi_within_volatility_matrix(
    comparison: pd.DataFrame,
):
    """
    Print a 5x5 downside-event matrix.
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
    matrix = pd.pivot_table(
        data,
        values="target",
        index="vol_q",
        columns="fi_q",
        aggfunc="mean",
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
            float_format=lambda x: f"{x:.4f}"
        )
    )
# ============================================================================
# Year-by-year analysis
# ============================================================================
def print_year_analysis(
    comparison: pd.DataFrame,
):
    """
    Repeat the volatility-quintile analysis separately for each OOS year.
    Quintiles are calculated within each year to show whether the
    regime relationship survives independently of changes in the
    overall volatility distribution.
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
        data["snapshot_date"],
        errors="coerce",
    ).dt.year
    print()
    print("=" * 100)
    print("YEAR-BY-YEAR VOLATILITY REGIMES")
    print("=" * 100)
    for year in sorted(
        data["year"].dropna().unique()
    ):
        year_data = data[
            data["year"] == year
        ].copy()
        if len(year_data) < 100:
            continue
        year_data["volatility_regime"] = (
            pd.qcut(
                year_data[
                    "price_volatility_20d"
                ],
                q=5,
                labels=[
                    "Q1_low",
                    "Q2",
                    "Q3",
                    "Q4",
                    "Q5_high",
                ],
                duplicates="drop",
            )
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
            subset = year_data[
                year_data[
                    "volatility_regime"
                ]
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
            print(
                f"{regime:<8}"
                f" n={len(subset):6,d}"
                f" event={subset['target'].mean():.4f}"
                f" FI_AUC={fi_auc:.4f}"
                f" FI+VOL_AUC={vol_auc:.4f}"
                f" delta={vol_auc - fi_auc:+.4f}"
            )
# ============================================================================
# Economic comparison by regime
# ============================================================================
def print_regime_economic_comparison(
    comparison: pd.DataFrame,
):
    """
    Compare the economic ranking performance of FI-only and FI+volatility
    within LOW/MID/HIGH volatility regimes.
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
    data = add_three_regimes(
        data
    )
    print()
    print("=" * 100)
    print("ECONOMIC PERFORMANCE BY VOLATILITY REGIME")
    print("=" * 100)
    for regime in REGIMES:
        subset = data[
            data["three_vol_regime"]
            == regime
        ]
        if subset.empty:
            continue
        print()
        print(regime)
        print("-" * 100)
        for fraction in ECONOMIC_FRACTIONS:
            fi_metrics = calculate_top_metrics(
                subset,
                "fi_score",
                fraction,
            )
            vol_metrics = calculate_top_metrics(
                subset,
                "vol_score",
                fraction,
            )
            print(
                f"Top {fraction:.1%}: "
                f"FI "
                f"event={fi_metrics['event_rate']:.4f} "
                f"lift={fi_metrics['lift']:.2f}x "
                f"mean={fi_metrics['mean_return']:.4%} "
                f"median={fi_metrics['median_return']:.4%}"
                f" | "
                f"FI+VOL "
                f"event={vol_metrics['event_rate']:.4f} "
                f"lift={vol_metrics['lift']:.2f}x "
                f"mean={vol_metrics['mean_return']:.4%} "
                f"median={vol_metrics['median_return']:.4%}"
            )
# ============================================================================
# Overall comparison
# ============================================================================
def print_overall_comparison(
    comparison: pd.DataFrame,
):
    """
    Print overall OOS AUC and economic comparison.
    """
    data = comparison.dropna(
        subset=[
            "fi_score",
            "vol_score",
            "target",
            "target_return",
        ]
    ).copy()
    fi_auc = safe_auc(
        data["target"],
        data["fi_score"],
    )
    vol_auc = safe_auc(
        data["target"],
        data["vol_score"],
    )
    print()
    print("=" * 100)
    print("OVERALL OOS COMPARISON")
    print("=" * 100)
    print(
        f"Rows: {len(data):,}"
    )
    print(
        f"Baseline event rate: "
        f"{data['target'].mean():.4f}"
    )
    print(
        f"Baseline mean return: "
        f"{data['target_return'].mean():.4%}"
    )
    print(
        f"FI-only OOS AUC: "
        f"{fi_auc:.6f}"
    )
    print(
        f"FI + volatility OOS AUC: "
        f"{vol_auc:.6f}"
    )
    print(
        f"AUC difference: "
        f"{vol_auc - fi_auc:+.6f}"
    )
    print()
    print("FI-only economic ranking")
    print_economic_metrics(
        data,
        "fi_score",
    )
    print()
    print("FI + volatility economic ranking")
    print_economic_metrics(
        data,
        "vol_score",
    )
# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 100)
    print("BLANKDISS VOLATILITY REGIME DIAGNOSTIC")
    print("=" * 100)
    print(
        f"RF trees: {BENCHMARK_TREES}"
    )
    print(
        "RF n_jobs: 1"
    )
    print(
        f"Walk-forward windows: "
        f"{len(WALK_FORWARD_WINDOWS)}"
    )
    print(
        f"Economic target: "
        f"{ECONOMIC_TARGET}"
    )
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print()
    print("Loading feature data...")
    feature_df = load_features()
    print(
        f"Feature rows: "
        f"{len(feature_df):,}"
    )
    # ------------------------------------------------------------------
    # Build feature sets
    # ------------------------------------------------------------------
    print()
    print("Building feature sets...")
    feature_sets = build_feature_sets(
        feature_df
    )
    target_definition = next(
        target
        for target in TARGETS
        if target.name == ECONOMIC_TARGET
    )
    # ------------------------------------------------------------------
    # Temporarily force benchmark RF configuration into the normal
    # walk-forward engine.
    # ------------------------------------------------------------------
    original_walk_forward_build_models = (
        walk_forward.build_models
    )
    walk_forward.build_models = (
        build_benchmark_models
    )
    try:
        # --------------------------------------------------------------
        # FI-only
        # --------------------------------------------------------------
        print()
        print("=" * 100)
        print("RUNNING FI-ONLY")
        print("=" * 100)
        fi_data, fi_columns = (
            feature_sets[BASE_FEATURES]
        )
        fi_result = run_feature_set(
            BASE_FEATURES,
            fi_data,
            fi_columns,
            target_definition,
        )
        # --------------------------------------------------------------
        # FI + volatility
        # --------------------------------------------------------------
        print()
        print("=" * 100)
        print("RUNNING FI + VOLATILITY")
        print("=" * 100)
        vol_data, vol_columns = (
            feature_sets[VOL_FEATURES]
        )
        vol_result = run_feature_set(
            VOL_FEATURES,
            vol_data,
            vol_columns,
            target_definition,
        )
    finally:
        walk_forward.build_models = (
            original_walk_forward_build_models
        )
    # ------------------------------------------------------------------
    # Print window results
    # ------------------------------------------------------------------
    for result in [
        fi_result,
        vol_result,
    ]:
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
    # ------------------------------------------------------------------
    # Verify OOS data exists
    # ------------------------------------------------------------------
    if fi_result["oos"].empty:
        raise RuntimeError(
            "FI-only produced no OOS predictions."
        )
    if vol_result["oos"].empty:
        raise RuntimeError(
            "FI+volatility produced no OOS predictions."
        )
    # ------------------------------------------------------------------
    # Attach raw source features.
    #
    # IMPORTANT:
    # fi_data intentionally does not contain price_volatility_20d.
    # vol_data does, so build_comparison_frame() takes volatility
    # from the FI+volatility source frame.
    # ------------------------------------------------------------------
    fi_oos = attach_source_features(
        fi_result["oos"],
        fi_data,
    )
    vol_oos = attach_source_features(
        vol_result["oos"],
        vol_data,
    )
    # ------------------------------------------------------------------
    # Match both models on exactly the same OOS observations.
    # ------------------------------------------------------------------
    comparison = build_comparison_frame(
        fi_oos,
        vol_oos,
    )
    print()
    print(
        f"Matched OOS rows: "
        f"{len(comparison):,}"
    )
    if comparison.empty:
        raise RuntimeError(
            "No matched OOS observations."
        )
    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    print_regime_analysis(
        comparison
    )
    print_three_regime_analysis(
        comparison
    )
    print_fi_signal_strength(
        comparison
    )
    print_fi_within_volatility_matrix(
        comparison
    )
    print_year_analysis(
        comparison
    )
    print_regime_economic_comparison(
        comparison
    )
    print_overall_comparison(
        comparison
    )
    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("RUNTIME")
    print("=" * 100)
    print(
        f"FI-only: "
        f"{fi_result['total_seconds']:.2f}s"
    )
    print(
        f"FI + volatility: "
        f"{vol_result['total_seconds']:.2f}s"
    )
    print()
    print("=" * 100)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 100)
if __name__ == "__main__":
    main()

Den viktiga skillnaden mot förra versionen är att build_comparison_frame() aldrig längre försöker hämta price_volatility_20d från fi_oos. Den tas från volatility_oos, där den faktiskt finns.

Kör den här. Nu bör vi komma hela vägen till regime-resultaten.
