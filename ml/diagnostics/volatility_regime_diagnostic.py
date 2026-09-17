"""
Blankdiss volatility regime gate diagnostic.
Tests whether a volatility-dependent model gate improves the
economic OOS ranking compared with:
    1. FI-only everywhere.
    2. FI + volatility everywhere.
    3. LOW -> FI, MID -> validation-selected, HIGH -> FI+VOL.
    4. LOW -> FI, MID -> FI+VOL, HIGH -> FI+VOL.
IMPORTANT:
    Regime selection is performed using validation data only.
    Test/OOS data is never used to decide which model to use.
The volatility regime boundaries are also learned from the training
period and then applied unchanged to validation and test data.
This avoids look-ahead leakage.
No repository files are modified by this script.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import ml.walk_forward as walk_forward
from ml.config import (
    RANDOM_STATE,
    TARGETS,
    WALK_FORWARD_WINDOWS,
)
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
VOLATILITY_FEATURE = "price_volatility_20d"
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
STRATEGIES = (
    "fi_only",
    "fi_plus_volatility",
    "gate_validation",
    "gate_vol_mid_high",
)
# ============================================================================
# Model configuration
# ============================================================================
def build_benchmark_models(
    random_state: int,
    task: str = "classification",
):
    """
    Build the normal Blankdiss model set while forcing RF benchmark
    configuration.
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
# ============================================================================
# Helpers
# ============================================================================
def safe_auc(
    y_true,
    scores,
):
    """
    Return ROC AUC or NaN when only one class exists.
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
def calculate_top_metrics(
    frame: pd.DataFrame,
    score_column: str,
    fraction: float,
):
    """
    Economic metrics for the highest-scored observations.
    """
    valid = frame[
        frame[score_column].notna()
        & frame["target"].notna()
        & frame["target_return"].notna()
    ].copy()
    if valid.empty:
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
                len(valid)
                * fraction
            )
        ),
    )
    ranked = (
        valid
        .sort_values(
            score_column,
            ascending=False,
        )
        .head(n)
    )
    event_rate = ranked[
        "target"
    ].mean()
    baseline_event_rate = valid[
        "target"
    ].mean()
    lift = (
        event_rate / baseline_event_rate
        if baseline_event_rate > 0
        else np.nan
    )
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
def print_top_metrics(
    frame: pd.DataFrame,
    score_column: str,
    indent: str = "    ",
):
    """
    Print economic ranking metrics.
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
def make_target(
    target_return: pd.Series,
):
    """
    Economic target matching down_5pct_5d.
    """
    return (
        target_return <= -0.05
    ).astype(int)
# ============================================================================
# Source features
# ============================================================================
def attach_source_features(
    frame: pd.DataFrame,
    source: pd.DataFrame,
):
    """
    Attach raw volatility and FI source features.
    The volatility feature is intentionally sourced from the
    FI+volatility dataframe because FI-only does not contain it.
    """
    columns = [
        "snapshot_date",
        "security_key",
        "price_volatility_20d",
        "short_interest_pct",
    ]
    missing = [
        column
        for column in columns
        if column not in source.columns
    ]
    if missing:
        raise KeyError(
            "Source feature frame is missing columns: "
            f"{missing}"
        )
    source_frame = source[
        columns
    ].copy()
    source_frame["snapshot_date"] = pd.to_datetime(
        source_frame["snapshot_date"],
        errors="coerce",
    )
    source_frame["security_key"] = (
        source_frame["security_key"].astype(str)
    )
    source_frame = source_frame.drop_duplicates(
        subset=[
            "snapshot_date",
            "security_key",
        ],
        keep="last",
    )
    result = frame.copy()
    result["snapshot_date"] = pd.to_datetime(
        result["snapshot_date"],
        errors="coerce",
    )
    result["security_key"] = (
        result["security_key"].astype(str)
    )
    return result.merge(
        source_frame,
        on=[
            "snapshot_date",
            "security_key",
        ],
        how="left",
    )
# ============================================================================
# Model training for one feature set
# ============================================================================
def train_feature_set_window(
    ml_data: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
    window,
):
    """
    Train all candidate models for one feature set and one
    walk-forward window.
    Returns:
        selected model
        validation predictions
        test predictions
        validation data
        test data
        validation score
        timing
    """
    (
        train_mask,
        validation_mask,
        test_mask,
    ) = walk_forward._split(
        ml_data,
        y,
        window,
    )
    train = ml_data.loc[
        train_mask,
        feature_columns,
    ]
    validation = ml_data.loc[
        validation_mask,
        feature_columns,
    ]
    test = ml_data.loc[
        test_mask,
        feature_columns,
    ]
    y_train = y.loc[train_mask]
    y_validation = y.loc[validation_mask]
    y_test = y.loc[test_mask]
    if (
        len(train) == 0
        or len(validation) == 0
        or len(test) == 0
    ):
        raise RuntimeError(
            "Empty train/validation/test split."
        )
    available_features = (
        walk_forward._features_available_in_training(
            train,
            feature_columns,
        )
    )
    if not available_features:
        raise RuntimeError(
            "No features available in training."
        )
    train = train.loc[
        :,
        available_features,
    ]
    validation = validation.loc[
        :,
        available_features,
    ]
    test = test.loc[
        :,
        available_features,
    ]
    models = build_benchmark_models(
        RANDOM_STATE,
        task="classification",
    )
    trained_models = []
    total_fit_seconds = 0.0
    total_validation_seconds = 0.0
    for name, model in models.items():
        fit_start = time.perf_counter()
        model.fit(
            train,
            y_train,
        )
        total_fit_seconds += (
            time.perf_counter()
            - fit_start
        )
        validation_start = time.perf_counter()
        validation_predictions = (
            model.predict_proba(
                validation
            )[:, 1]
        )
        total_validation_seconds += (
            time.perf_counter()
            - validation_start
        )
        validation_score = safe_auc(
            y_validation,
            validation_predictions,
        )
        trained_models.append(
            {
                "name": name,
                "model": model,
                "validation_score": (
                    validation_score
                ),
                "validation_predictions": (
                    validation_predictions
                ),
            }
        )
    trained_models.sort(
        key=lambda item: (
            item["validation_score"]
            if np.isfinite(
                item["validation_score"]
            )
            else -np.inf
        ),
        reverse=True,
    )
    selected = trained_models[0]
    test_predictions = (
        selected["model"].predict_proba(
            test
        )[:, 1]
    )
    validation_frame = ml_data.loc[
        validation_mask,
        [
            "snapshot_date",
            "security_key",
            "target_return",
        ],
    ].copy()
    validation_frame[
        "target"
    ] = make_target(
        validation_frame[
            "target_return"
        ]
    )
    validation_frame[
        "prediction"
    ] = selected[
        "validation_predictions"
    ]
    validation_frame[
        "price_volatility_20d"
    ] = ml_data.loc[
        validation_mask,
        VOLATILITY_FEATURE,
    ].to_numpy()
    test_frame = ml_data.loc[
        test_mask,
        [
            "snapshot_date",
            "security_key",
            "target_return",
        ],
    ].copy()
    test_frame[
        "target"
    ] = make_target(
        test_frame[
            "target_return"
        ]
    )
    test_frame[
        "prediction"
    ] = test_predictions
    test_frame[
        "price_volatility_20d"
    ] = ml_data.loc[
        test_mask,
        VOLATILITY_FEATURE,
    ].to_numpy()
    return {
        "selected_model": selected[
            "name"
        ],
        "validation_score": selected[
            "validation_score"
        ],
        "validation": validation_frame,
        "test": test_frame,
        "all_models": trained_models,
        "fit_seconds": total_fit_seconds,
        "validation_seconds": (
            total_validation_seconds
        ),
    }
# ============================================================================
# Regime boundaries
# ============================================================================
def calculate_training_regime_boundaries(
    train_volatility: pd.Series,
):
    """
    Calculate LOW/MID/HIGH boundaries from training data only.
    This is critical: validation/test data must not influence
    the regime boundaries.
    """
    values = pd.to_numeric(
        train_volatility,
        errors="coerce",
    ).dropna()
    if values.empty:
        raise RuntimeError(
            "No training volatility values available."
        )
    q1 = float(
        values.quantile(
            1.0 / 3.0
        )
    )
    q2 = float(
        values.quantile(
            2.0 / 3.0
        )
    )
    return q1, q2
def assign_regime(
    volatility,
    q1: float,
    q2: float,
):
    """
    Apply training-derived volatility boundaries.
    """
    if pd.isna(volatility):
        return None
    if volatility <= q1:
        return "LOW"
    if volatility <= q2:
        return "MID"
    return "HIGH"
# ============================================================================
# Validation gate selection
# ============================================================================
def select_regime_gate(
    fi_validation: pd.DataFrame,
    vol_validation: pd.DataFrame,
    q1: float,
    q2: float,
):
    """
    Select FI or FI+VOL separately for each regime using
    validation AUC only.
    The returned gate is then frozen for OOS.
    """
    fi = fi_validation[
        [
            "snapshot_date",
            "security_key",
            "target_return",
            "target",
            "price_volatility_20d",
            "prediction",
        ]
    ].rename(
        columns={
            "prediction": "fi_prediction",
        }
    )
    vol = vol_validation[
        [
            "snapshot_date",
            "security_key",
            "target_return",
            "target",
            "price_volatility_20d",
            "prediction",
        ]
    ].rename(
        columns={
            "prediction": "vol_prediction",
        }
    )
    merged = fi.merge(
        vol,
        on=[
            "snapshot_date",
            "security_key",
        ],
        suffixes=(
            "_fi",
            "_vol",
        ),
    )
    merged["regime"] = merged[
        "price_volatility_fi"
    ].apply(
        lambda value: assign_regime(
            value,
            q1,
            q2,
        )
    )
    gate = {}
    print()
    print(
        "Validation regime selection:"
    )
    for regime in REGIMES:
        subset = merged[
            merged["regime"]
            == regime
        ]
        fi_auc = safe_auc(
            subset["target_fi"],
            subset["fi_prediction"],
        )
        vol_auc = safe_auc(
            subset["target_vol"],
            subset["vol_prediction"],
        )
        if (
            np.isfinite(fi_auc)
            and np.isfinite(vol_auc)
        ):
            selected = (
                "fi"
                if fi_auc >= vol_auc
                else "vol"
            )
        elif np.isfinite(fi_auc):
            selected = "fi"
        elif np.isfinite(vol_auc):
            selected = "vol"
        else:
            selected = "fi"
        gate[regime] = selected
        print(
            f"  {regime}: "
            f"FI AUC={fi_auc:.6f} | "
            f"FI+VOL AUC={vol_auc:.6f} | "
            f"selected={selected}"
        )
    return gate
# ============================================================================
# Apply strategy
# ============================================================================
def apply_strategy(
    fi_test: pd.DataFrame,
    vol_test: pd.DataFrame,
    strategy: str,
    gate: dict[str, str],
    q1: float,
    q2: float,
):
    """
    Apply a frozen strategy to OOS/test data.
    """
    fi = fi_test[
        [
            "snapshot_date",
            "security_key",
            "target_return",
            "target",
            "price_volatility_20d",
            "prediction",
        ]
    ].rename(
        columns={
            "prediction": "fi_prediction",
        }
    )
    vol = vol_test[
        [
            "snapshot_date",
            "security_key",
            "target_return",
            "target",
            "price_volatility_20d",
            "prediction",
        ]
    ].rename(
        columns={
            "prediction": "vol_prediction",
        }
    )
    merged = fi.merge(
        vol,
        on=[
            "snapshot_date",
            "security_key",
        ],
        suffixes=(
            "_fi",
            "_vol",
        ),
    )
    target_difference = (
        merged["target_return_fi"]
        - merged["target_return_vol"]
    ).abs()
    if (
        target_difference
        > 1e-10
    ).any():
        raise ValueError(
            "Target return mismatch between "
            "FI and FI+VOL test rows."
        )
    merged["target"] = merged[
        "target_fi"
    ]
    merged["target_return"] = merged[
        "target_return_fi"
    ]
    merged["regime"] = merged[
        "price_volatility_fi"
    ].apply(
        lambda value: assign_regime(
            value,
            q1,
            q2,
        )
    )
    if strategy == "fi_only":
        merged["score"] = (
            merged["fi_prediction"]
        )
        merged["chosen_model"] = "FI"
    elif strategy == "fi_plus_volatility":
        merged["score"] = (
            merged["vol_prediction"]
        )
        merged["chosen_model"] = (
            "FI+VOL"
        )
    elif strategy == "gate_validation":
        def choose_validation(
            regime
        ):
            return gate.get(
                regime,
                "fi",
            )
        selected = merged[
            "regime"
        ].map(
            choose_validation
        )
        merged["score"] = np.where(
            selected == "vol",
            merged["vol_prediction"],
            merged["fi_prediction"],
        )
        merged["chosen_model"] = np.where(
            selected == "vol",
            "FI+VOL",
            "FI",
        )
    elif strategy == "gate_vol_mid_high":
        selected = np.where(
            merged["regime"]
            == "LOW",
            "fi",
            "vol",
        )
        merged["score"] = np.where(
            selected == "vol",
            merged["vol_prediction"],
            merged["fi_prediction"],
        )
        merged["chosen_model"] = np.where(
            selected == "vol",
            "FI+VOL",
            "FI",
        )
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )
    return merged[
        [
            "snapshot_date",
            "security_key",
            "target_return",
            "target",
            "price_volatility_fi",
            "regime",
            "score",
            "chosen_model",
        ]
    ].copy()
# ============================================================================
# Strategy summary
# ============================================================================
def print_strategy_metrics(
    frame: pd.DataFrame,
    strategy: str,
):
    """
    Print AUC and economic ranking metrics.
    """
    auc = safe_auc(
        frame["target"],
        frame["score"],
    )
    print()
    print(
        f"{strategy}"
    )
    print(
        f"  Rows: {len(frame):,}"
    )
    print(
        f"  AUC: {auc:.6f}"
    )
    print(
        f"  Baseline event rate: "
        f"{frame['target'].mean():.4f}"
    )
    print(
        f"  Baseline mean return: "
        f"{frame['target_return'].mean():.4%}"
    )
    print_top_metrics(
        frame,
        "score",
    )
def print_regime_usage(
    frame: pd.DataFrame,
):
    """
    Show how many OOS rows each gated model handled.
    """
    counts = (
        frame[
            "chosen_model"
        ]
        .value_counts()
    )
    print(
        "  Model usage:"
    )
    for model_name in [
        "FI",
        "FI+VOL",
    ]:
        count = int(
            counts.get(
                model_name,
                0,
            )
        )
        fraction = (
            count / len(frame)
            if len(frame)
            else np.nan
        )
        print(
            f"    {model_name}: "
            f"{count:,} "
            f"({fraction:.1%})"
        )
def print_regime_strategy_metrics(
    frame: pd.DataFrame,
):
    """
    Show economics separately for LOW/MID/HIGH.
    """
    print()
    for regime in REGIMES:
        subset = frame[
            frame["regime"]
            == regime
        ]
        if subset.empty:
            continue
        print(
            f"  {regime}: "
            f"n={len(subset):,} "
            f"event={subset['target'].mean():.4f} "
            f"mean={subset['target_return'].mean():.4%}"
        )
        print_top_metrics(
            subset,
            "score",
            indent="    ",
        )
# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 100)
    print("BLANKDISS VOLATILITY REGIME GATE DIAGNOSTIC")
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
    print()
    print(
        "Strategies:"
    )
    print(
        "  1. FI-only everywhere"
    )
    print(
        "  2. FI+VOL everywhere"
    )
    print(
        "  3. LOW=FI, MID=validation-selected, HIGH=VOL"
    )
    print(
        "  4. LOW=FI, MID=VOL, HIGH=VOL"
    )
    # ------------------------------------------------------------------
    # Load features
    # ------------------------------------------------------------------
    print()
    print(
        "Loading feature data..."
    )
    feature_df = load_features()
    print(
        f"Feature rows: "
        f"{len(feature_df):,}"
    )
    # ------------------------------------------------------------------
    # Build exact feature sets
    # ------------------------------------------------------------------
    print()
    print(
        "Building feature sets..."
    )
    fi_data, fi_columns = (
        prepare_feature_set(
            feature_df,
            include_price_features=False,
            price_features=None,
        )
    )
    vol_data, vol_columns = (
        prepare_feature_set(
            feature_df,
            include_price_features=True,
            price_features={
                VOLATILITY_FEATURE,
            },
        )
    )
    # ------------------------------------------------------------------
    # Target
    # ------------------------------------------------------------------
    target_definition = next(
        target
        for target in TARGETS
        if target.name
        == ECONOMIC_TARGET
    )
    (
        fi_ml_data,
        fi_y,
        fi_feature_columns,
    ) = prepare_ml_data_from_feature_set(
        fi_data,
        fi_columns,
        target_definition,
    )
    (
        vol_ml_data,
        vol_y,
        vol_feature_columns,
    ) = prepare_ml_data_from_feature_set(
        vol_data,
        vol_columns,
        target_definition,
    )
    # ------------------------------------------------------------------
    # Run each walk-forward window
    # ------------------------------------------------------------------
    all_strategy_results = {
        strategy: []
        for strategy in STRATEGIES
    }
    total_start = time.perf_counter()
    for window_index, window in enumerate(
        WALK_FORWARD_WINDOWS,
        start=1,
    ):
        print()
        print("=" * 100)
        print(
            f"WALK-FORWARD WINDOW {window_index}"
        )
        print("=" * 100)
        print(
            f"Train <= {window.train_end}"
        )
        print(
            f"Validation <= "
            f"{window.validation_end}"
        )
        print(
            f"Test <= {window.test_end}"
        )
        window_start = time.perf_counter()
        # --------------------------------------------------------------
        # Training volatility boundaries.
        #
        # Boundaries come from training only.
        # --------------------------------------------------------------
        (
            train_mask,
            validation_mask,
            test_mask,
        ) = walk_forward._split(
            vol_ml_data,
            vol_y,
            window,
        )
        train_volatility = vol_ml_data.loc[
            train_mask,
            VOLATILITY_FEATURE,
        ]
        q1, q2 = (
            calculate_training_regime_boundaries(
                train_volatility
            )
        )
        print()
        print(
            "Training-derived volatility boundaries:"
        )
        print(
            f"  LOW <= {q1:.6f}"
        )
        print(
            f"  MID <= {q2:.6f}"
        )
        print(
            f"  HIGH > {q2:.6f}"
        )
        # --------------------------------------------------------------
        # FI model
        # --------------------------------------------------------------
        fi_start = time.perf_counter()
        fi_result = train_feature_set_window(
            fi_ml_data,
            fi_y,
            fi_feature_columns,
            window,
        )
        fi_seconds = (
            time.perf_counter()
            - fi_start
        )
        print()
        print(
            "FI-only:"
        )
        print(
            f"  Selected model: "
            f"{fi_result['selected_model']}"
        )
        print(
            f"  Validation AUC: "
            f"{fi_result['validation_score']:.6f}"
        )
        print(
            f"  Time: "
            f"{fi_seconds:.2f}s"
        )
        # --------------------------------------------------------------
        # FI + volatility model
        # --------------------------------------------------------------
        vol_start = time.perf_counter()
        vol_result = train_feature_set_window(
            vol_ml_data,
            vol_y,
            vol_feature_columns,
            window,
        )
        vol_seconds = (
            time.perf_counter()
            - vol_start
        )
        print()
        print(
            "FI + volatility:"
        )
        print(
            f"  Selected model: "
            f"{vol_result['selected_model']}"
        )
        print(
            f"  Validation AUC: "
            f"{vol_result['validation_score']:.6f}"
        )
        print(
            f"  Time: "
            f"{vol_seconds:.2f}s"
        )
        # --------------------------------------------------------------
        # Validation gate
        # --------------------------------------------------------------
        gate = select_regime_gate(
            fi_result["validation"],
            vol_result["validation"],
            q1,
            q2,
        )
        # --------------------------------------------------------------
        # Apply four strategies to this window's OOS test.
        # --------------------------------------------------------------
        for strategy in STRATEGIES:
            result = apply_strategy(
                fi_result["test"],
                vol_result["test"],
                strategy,
                gate,
                q1,
                q2,
            )
            result[
                "window_index"
            ] = window_index
            result[
                "window_train_end"
            ] = window.train_end
            result[
                "window_validation_end"
            ] = window.validation_end
            result[
                "window_test_end"
            ] = window.test_end
            all_strategy_results[
                strategy
            ].append(
                result
            )
        print()
        print(
            "Window strategy results:"
        )
        for strategy in STRATEGIES:
            result = all_strategy_results[
                strategy
            ][-1]
            print()
            print(
                f"  {strategy}: "
                f"AUC="
                f"{safe_auc(result['target'], result['score']):.6f} "
                f"top1="
                f"{calculate_top_metrics(result, 'score', 0.01)['event_rate']:.4f} "
                f"mean="
                f"{calculate_top_metrics(result, 'score', 0.01)['mean_return']:.4%}"
            )
        print()
        print(
            f"Window runtime: "
            f"{time.perf_counter() - window_start:.2f}s"
        )
    # ------------------------------------------------------------------
    # Combine OOS windows
    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("COMBINED OOS RESULTS")
    print("=" * 100)
    combined = {}
    for strategy in STRATEGIES:
        combined[strategy] = pd.concat(
            all_strategy_results[
                strategy
            ],
            ignore_index=True,
        )
    # ------------------------------------------------------------------
    # Overall comparison
    # ------------------------------------------------------------------
    for strategy in STRATEGIES:
        frame = combined[
            strategy
        ]
        print()
        print("=" * 100)
        print(
            strategy.upper()
        )
        print("=" * 100)
        print_strategy_metrics(
            frame,
            strategy,
        )
        if strategy in (
            "gate_validation",
            "gate_vol_mid_high",
        ):
            print()
            print_regime_usage(
                frame
            )
            print_regime_strategy_metrics(
                frame
            )
    # ------------------------------------------------------------------
    # Direct comparison table
    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("STRATEGY COMPARISON")
    print("=" * 100)
    rows = []
    for strategy in STRATEGIES:
        frame = combined[
            strategy
        ]
        top_metrics = calculate_top_metrics(
            frame,
            "score",
            0.01,
        )
        rows.append(
            {
                "strategy": strategy,
                "rows": len(frame),
                "auc": safe_auc(
                    frame["target"],
                    frame["score"],
                ),
                "top_0.1_event": (
                    calculate_top_metrics(
                        frame,
                        "score",
                        0.001,
                    )["event_rate"]
                ),
                "top_0.5_event": (
                    calculate_top_metrics(
                        frame,
                        "score",
                        0.005,
                    )["event_rate"]
                ),
                "top_1_event": (
                    top_metrics[
                        "event_rate"
                    ]
                ),
                "top_1_lift": (
                    top_metrics[
                        "lift"
                    ]
                ),
                "top_1_mean_return": (
                    top_metrics[
                        "mean_return"
                    ]
                ),
                "top_1_median_return": (
                    top_metrics[
                        "median_return"
                    ]
                ),
                "top_2_mean_return": (
                    calculate_top_metrics(
                        frame,
                        "score",
                        0.02,
                    )["mean_return"]
                ),
                "top_5_mean_return": (
                    calculate_top_metrics(
                        frame,
                        "score",
                        0.05,
                    )["mean_return"]
                ),
            }
        )
    comparison = pd.DataFrame(
        rows
    )
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )
    # ------------------------------------------------------------------
    # Gate selection consistency
    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("REGIME GATE MODEL USAGE")
    print("=" * 100)
    for strategy in (
        "gate_validation",
        "gate_vol_mid_high",
    ):
        frame = combined[
            strategy
        ]
        print()
        print(
            strategy
        )
        usage = (
            frame.groupby(
                [
                    "regime",
                    "chosen_model",
                ]
            )
            .size()
            .reset_index(
                name="rows"
            )
        )
        print(
            usage.to_string(
                index=False
            )
        )
    # ------------------------------------------------------------------
    # Per-window strategy comparison
    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("PER-WINDOW STRATEGY COMPARISON")
    print("=" * 100)
    window_rows = []
    for window_index in range(
        1,
        len(WALK_FORWARD_WINDOWS) + 1,
    ):
        row = {
            "window": window_index,
        }
        for strategy in STRATEGIES:
            frame = combined[
                strategy
            ]
            subset = frame[
                frame[
                    "window_index"
                ]
                == window_index
            ]
            row[
                f"{strategy}_auc"
            ] = safe_auc(
                subset["target"],
                subset["score"],
            )
            row[
                f"{strategy}_top1_mean"
            ] = calculate_top_metrics(
                subset,
                "score",
                0.01,
            )[
                "mean_return"
            ]
        window_rows.append(
            row
        )
    window_comparison = pd.DataFrame(
        window_rows
    )
    print(
        window_comparison.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )
    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    print()
    print("=" * 100)
    print("RUNTIME")
    print("=" * 100)
    print(
        f"Total: "
        f"{time.perf_counter() - total_start:.2f}s"
    )
    print()
    print("=" * 100)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 100)
if __name__ == "__main__":
    main()
