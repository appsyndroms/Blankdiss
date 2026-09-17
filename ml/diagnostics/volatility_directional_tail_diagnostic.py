"""
Volatility directional / tail diagnostic.
The model is trained only on:
    down_5pct_5d
The resulting OOS ranking is evaluated against:
    down_5pct_5d
    up_5pct_5d
    abs_5pct_5d
    abs_10pct_5d
The purpose is to determine whether the signal is primarily:
    1. Directional downside information
    2. A general large-move / volatility signal
    3. A downside-asymmetric large-move signal
The diagnostic also checks:
    - VOL60 versus VOL20 + VOL60
    - volatility term structure
    - volatility ratio
    - recent price returns
    - volatility quintiles
    - score deciles
    - calendar-year stability
    - walk-forward-window stability
    - top-tail return distributions
The repository's existing walk-forward implementation is used.
Only logistic regression is exposed during model selection.
"""
from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
import ml.walk_forward as walk_forward
from analysis.feature_config import PRICE_DIR
from analysis.feature_prices import find_price_files, load_prices
from ml.config import TARGETS, WALK_FORWARD_WINDOWS
from ml.dataset import build_target, load_features
from ml.models import build_models
from ml.walk_forward import train_window
TOP_FRACTIONS = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
)
TRAIN_TARGET = "down_5pct_5d"
FEATURE_SETS = {
    "volatility_60d": [
        "volatility_60d",
    ],
    "volatility_20d_plus_60d": [
        "price_volatility_20d",
        "volatility_60d",
    ],
    "volatility_60d_plus_term_structure": [
        "volatility_60d",
        "volatility_20d_minus_60d",
    ],
    "volatility_20d_plus_60d_plus_term_structure": [
        "price_volatility_20d",
        "volatility_60d",
        "volatility_20d_minus_60d",
    ],
    "volatility_20d_plus_60d_plus_ratio": [
        "price_volatility_20d",
        "volatility_60d",
        "volatility_20d_div_60d",
    ],
    "volatility_plus_returns": [
        "volatility_60d",
        "price_return_5d",
        "price_return_20d",
        "price_return_60d",
    ],
    "volatility_20d_plus_60d_plus_returns": [
        "price_volatility_20d",
        "volatility_60d",
        "price_return_5d",
        "price_return_20d",
        "price_return_60d",
    ],
}
ANALYSIS_TARGETS = {
    "down_5pct_5d": "target_down_5pct_5d",
    "up_5pct_5d": "target_up_5pct_5d",
    "abs_5pct_5d": "target_abs_5pct_5d",
    "abs_10pct_5d": "target_abs_10pct_5d",
}
def find_target(target_name: str):
    for target in TARGETS:
        if target.name == target_name:
            return target
    available = ", ".join(
        target.name
        for target in TARGETS
    )
    raise ValueError(
        f"Unknown target: {target_name}. "
        f"Available targets: {available}"
    )
def load_price_data() -> pd.DataFrame:
    print(
        "Loading raw price data for 60d volatility..."
    )
    prices = load_prices(
        find_price_files(PRICE_DIR)
    )
    if prices.empty:
        raise ValueError(
            "No raw price data found."
        )
    required = {
        "yahoo_symbol",
        "date",
        "close",
    }
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(
            "Raw price data is missing required columns: "
            f"{sorted(missing)}"
        )
    prices = prices.copy()
    prices["date"] = pd.to_datetime(
        prices["date"],
        errors="coerce",
    )
    prices["close"] = pd.to_numeric(
        prices["close"],
        errors="coerce",
    )
    prices = prices.dropna(
        subset=[
            "yahoo_symbol",
            "date",
            "close",
        ]
    )
    prices = prices.loc[
        prices["close"] > 0
    ].copy()
    prices = prices.sort_values(
        [
            "yahoo_symbol",
            "date",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )
    print(
        f"Price rows: {len(prices):,}"
    )
    return prices
def add_volatility_features(
    features: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    prices = prices[
        [
            "yahoo_symbol",
            "date",
            "close",
        ]
    ].copy()
    prices = prices.sort_values(
        [
            "yahoo_symbol",
            "date",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )
    prices["daily_return"] = (
        prices
        .groupby(
            "yahoo_symbol",
            sort=False,
        )["close"]
        .pct_change()
    )
    prices["volatility_60d"] = (
        prices
        .groupby(
            "yahoo_symbol",
            sort=False,
        )["daily_return"]
        .transform(
            lambda series: series.rolling(
                window=59,
                min_periods=59,
            ).std()
        )
    )
    prices = prices.rename(
        columns={
            "date": "price_date",
        }
    )
    lookup = prices[
        [
            "yahoo_symbol",
            "price_date",
            "volatility_60d",
        ]
    ]
    result = features.copy()
    result["price_date"] = pd.to_datetime(
        result["price_date"],
        errors="coerce",
    )
    result = result.merge(
        lookup,
        on=[
            "yahoo_symbol",
            "price_date",
        ],
        how="left",
        validate="many_to_one",
    )
    result["volatility_20d_minus_60d"] = (
        result["price_volatility_20d"]
        - result["volatility_60d"]
    )
    result["volatility_20d_div_60d"] = (
        result["price_volatility_20d"]
        / result["volatility_60d"].replace(
            0,
            np.nan,
        )
    )
    return result
def add_analysis_targets(
    data: pd.DataFrame,
) -> pd.DataFrame:
    data = data.copy()
    if "forward_return_5d" not in data.columns:
        raise ValueError(
            "Feature dataset is missing forward_return_5d."
        )
    returns = pd.to_numeric(
        data["forward_return_5d"],
        errors="coerce",
    )
    data["target_down_5pct_5d"] = (
        returns <= -0.05
    )
    data["target_up_5pct_5d"] = (
        returns >= 0.05
    )
    data["target_abs_5pct_5d"] = (
        returns.abs() >= 0.05
    )
    data["target_abs_10pct_5d"] = (
        returns.abs() >= 0.10
    )
    return data
def get_feature_columns(
    data: pd.DataFrame,
    feature_set: str,
) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set: {feature_set}"
        )
    columns = FEATURE_SETS[
        feature_set
    ]
    missing = [
        column
        for column in columns
        if column not in data.columns
    ]
    if missing:
        raise ValueError(
            f"Feature set {feature_set} is missing "
            f"columns: {missing}"
        )
    return list(columns)
def build_ml_dataset(
    data: pd.DataFrame,
    feature_columns: list[str],
    target,
):
    required_columns = [
        "snapshot_date",
        "security_key",
        "forward_return_5d",
        *feature_columns,
    ]
    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]
    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{missing}"
        )
    feature_data = data.dropna(
        subset=feature_columns
    ).copy()
    target_values = build_target(
        feature_data,
        target,
    )
    valid = target_values.notna()
    ml_data = feature_data.loc[
        valid,
        required_columns,
    ].copy()
    y = target_values.loc[
        valid
    ].copy()
    ml_data["target_return"] = pd.to_numeric(
        ml_data["forward_return_5d"],
        errors="coerce",
    )
    ml_data = ml_data.drop(
        columns=[
            "forward_return_5d",
        ]
    )
    valid_returns = (
        ml_data["target_return"].notna()
    )
    ml_data = ml_data.loc[
        valid_returns
    ].copy()
    y = y.loc[
        ml_data.index
    ].copy()
    ml_data = ml_data.reset_index(
        drop=True
    )
    y = y.reset_index(
        drop=True
    )
    return (
        ml_data,
        y,
    )
def logistic_only_models(
    random_state,
    task="classification",
):
    models = build_models(
        random_state=random_state,
        task=task,
    )
    if "logistic_regression" not in models:
        raise ValueError(
            "logistic_regression not found "
            "in build_models()."
        )
    return {
        "logistic_regression": models[
            "logistic_regression"
        ],
    }
def run_walk_forward(
    data: pd.DataFrame,
    y: pd.Series,
    feature_columns: list[str],
):
    original_build_models = (
        walk_forward.build_models
    )
    def controlled_build_models(
        random_state,
        task="classification",
    ):
        return logistic_only_models(
            random_state=random_state,
            task=task,
        )
    walk_forward.build_models = (
        controlled_build_models
    )
    try:
        oos_rows = []
        for (
            window_number,
            window,
        ) in enumerate(
            WALK_FORWARD_WINDOWS,
            start=1,
        ):
            (
                model_results,
                window_oos_rows,
                timing,
            ) = train_window(
                data=data,
                y=y,
                feature_columns=feature_columns,
                window=window,
                task="classification",
                direction="below",
            )
            if not model_results:
                raise ValueError(
                    "No model result for "
                    f"window {window_number}."
                )
            selected = next(
                (
                    result
                    for result in model_results
                    if result[
                        "selected_for_oos"
                    ]
                ),
                None,
            )
            if selected is None:
                raise ValueError(
                    "No selected OOS model for "
                    f"window {window_number}."
                )
            print()
            print(
                f"Window {window_number}"
            )
            print(
                f"  Train <= {window.train_end}"
            )
            print(
                "  Validation <= "
                f"{window.validation_end}"
            )
            print(
                f"  Test <= {window.test_end}"
            )
            print(
                "  Model: "
                f"{selected['model']}"
            )
            print(
                "  Validation AUC: "
                f"{selected['validation_score']:.6f}"
            )
            print(
                "  Fit: "
                f"{timing['fit_seconds']:.2f}s"
            )
            print(
                "  OOS rows: "
                f"{len(window_oos_rows):,}"
            )
            oos_rows.extend(
                window_oos_rows
            )
        if not oos_rows:
            raise ValueError(
                "No OOS rows available."
            )
        return pd.DataFrame(
            oos_rows
        )
    finally:
        walk_forward.build_models = (
            original_build_models
        )
def attach_analysis_targets_to_oos(
    oos: pd.DataFrame,
    data: pd.DataFrame,
) -> pd.DataFrame:
    required_oos = {
        "snapshot_date",
        "security_key",
        "prediction",
        "score",
    }
    missing_oos = (
        required_oos
        - set(oos.columns)
    )
    if missing_oos:
        raise ValueError(
            "OOS predictions are missing columns: "
            f"{sorted(missing_oos)}"
        )
    diagnostic_columns = [
        "snapshot_date",
        "security_key",
        "forward_return_5d",
        "target_down_5pct_5d",
        "target_up_5pct_5d",
        "target_abs_5pct_5d",
        "target_abs_10pct_5d",
        "price_volatility_20d",
        "volatility_60d",
        "volatility_20d_minus_60d",
        "volatility_20d_div_60d",
        "price_return_5d",
        "price_return_20d",
        "price_return_60d",
    ]
    missing_data = [
        column
        for column in diagnostic_columns
        if column not in data.columns
    ]
    if missing_data:
        raise ValueError(
            "Diagnostic data is missing columns: "
            f"{missing_data}"
        )
    oos = oos.copy()
    lookup = data[
        diagnostic_columns
    ].copy()
    oos["snapshot_date"] = (
        pd.to_datetime(
            oos["snapshot_date"],
            errors="coerce",
        ).dt.strftime(
            "%Y-%m-%d"
        )
    )
    lookup["snapshot_date"] = (
        pd.to_datetime(
            lookup["snapshot_date"],
            errors="coerce",
        ).dt.strftime(
            "%Y-%m-%d"
        )
    )
    lookup = lookup.drop_duplicates(
        subset=[
            "snapshot_date",
            "security_key",
        ],
        keep="last",
    )
    return oos.merge(
        lookup,
        on=[
            "snapshot_date",
            "security_key",
        ],
        how="left",
        validate="many_to_one",
        suffixes=(
            "",
            "_data",
        ),
    )
def safe_auc(
    score: pd.Series,
    target: pd.Series,
) -> float:
    from sklearn.metrics import roc_auc_score
    valid = (
        score.notna()
        & target.notna()
    )
    score = score.loc[
        valid
    ]
    target = target.loc[
        valid
    ]
    if target.nunique() < 2:
        return float("nan")
    return float(
        roc_auc_score(
            target,
            score,
        )
    )
def describe_top_fraction(
    frame: pd.DataFrame,
    target_column: str,
    fraction: float,
):
    frame = frame.dropna(
        subset=[
            "score",
            "forward_return_5d",
            target_column,
        ]
    )
    if frame.empty:
        return {
            "event_rate": float("nan"),
            "lift": float("nan"),
            "mean_return": float("nan"),
            "median_return": float("nan"),
            "positive_return_fraction": float("nan"),
            "negative_return_fraction": float("nan"),
            "n": 0,
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
    top = frame.nlargest(
        n,
        "score",
    )
    baseline_event = (
        frame[target_column].mean()
    )
    event_rate = (
        top[target_column].mean()
    )
    lift = (
        event_rate / baseline_event
        if baseline_event > 0
        else float("nan")
    )
    returns = top[
        "forward_return_5d"
    ]
    return {
        "event_rate": float(
            event_rate
        ),
        "lift": float(
            lift
        ),
        "mean_return": float(
            returns.mean()
        ),
        "median_return": float(
            returns.median()
        ),
        "positive_return_fraction": float(
            (returns > 0).mean()
        ),
        "negative_return_fraction": float(
            (returns < 0).mean()
        ),
        "n": int(
            len(top)
        ),
    }
def print_tail_analysis(
    oos: pd.DataFrame,
    label: str,
):
    print()
    print(
        "=" * 100
    )
    print(label)
    print(
        "=" * 100
    )
    for (
        target_name,
        target_column,
    ) in ANALYSIS_TARGETS.items():
        auc = safe_auc(
            oos["score"],
            oos[target_column],
        )
        baseline = (
            oos[target_column].mean()
        )
        print(
            f"{target_name:20s} "
            f"AUC={auc:.6f} "
            f"baseline={baseline:.4f}"
        )
    print()
    print(
        "Top-tail analysis "
        "(ranking trained only on down_5pct_5d)"
    )
    for (
        target_name,
        target_column,
    ) in ANALYSIS_TARGETS.items():
        print()
        print(
            f"Target: {target_name}"
        )
        for fraction in TOP_FRACTIONS:
            result = describe_top_fraction(
                oos,
                target_column,
                fraction,
            )
            print(
                f"  top {fraction * 100:5.1f}% "
                f"event={result['event_rate']:.4f} "
                f"lift={result['lift']:.2f}x "
                f"mean={result['mean_return']:+.4%} "
                f"median={result['median_return']:+.4%} "
                f"pos={result['positive_return_fraction']:.4f} "
                f"neg={result['negative_return_fraction']:.4f} "
                f"n={result['n']}"
            )
def print_direction_given_move(
    oos: pd.DataFrame,
):
    print()
    print(
        "=" * 100
    )
    print(
        "DIRECTION GIVEN LARGE MOVE"
    )
    print(
        "=" * 100
    )
    returns = oos[
        "forward_return_5d"
    ]
    for threshold in (
        0.05,
        0.10,
    ):
        large = (
            returns.abs()
            >= threshold
        )
        if not large.any():
            continue
        large_returns = returns.loc[
            large
        ]
        down_fraction = (
            large_returns
            <= -threshold
        ).mean()
        up_fraction = (
            large_returns
            >= threshold
        ).mean()
        print(
            f"|return| >= {threshold:.0%}: "
            f"n={len(large_returns)} "
            f"down={down_fraction:.4f} "
            f"up={up_fraction:.4f}"
        )
    print()
    print(
        "Same calculation restricted "
        "to the model's top 1%:"
    )
    top_n = max(
        1,
        int(
            np.ceil(
                len(oos)
                * 0.01
            )
        ),
    )
    top = oos.nlargest(
        top_n,
        "score",
    )
    for threshold in (
        0.05,
        0.10,
    ):
        large_returns = top.loc[
            top[
                "forward_return_5d"
            ].abs()
            >= threshold,
            "forward_return_5d",
        ]
        if large_returns.empty:
            continue
        down_fraction = (
            large_returns
            <= -threshold
        ).mean()
        up_fraction = (
            large_returns
            >= threshold
        ).mean()
        print(
            f"top 1%, "
            f"|return| >= {threshold:.0%}: "
            f"n={len(large_returns)} "
            f"down={down_fraction:.4f} "
            f"up={up_fraction:.4f}"
        )
def print_return_buckets(
    oos: pd.DataFrame,
):
    print()
    print(
        "=" * 100
    )
    print(
        "ACTUAL 5-DAY RETURN DISTRIBUTION"
    )
    print(
        "=" * 100
    )
    buckets = (
        (
            "<= -10%",
            -np.inf,
            -0.10,
        ),
        (
            "-10% to -5%",
            -0.10,
            -0.05,
        ),
        (
            "-5% to 0%",
            -0.05,
            0.0,
        ),
        (
            "0% to 5%",
            0.0,
            0.05,
        ),
        (
            "5% to 10%",
            0.05,
            0.10,
        ),
        (
            ">= 10%",
            0.10,
            np.inf,
        ),
    )
    for fraction in TOP_FRACTIONS:
        n = max(
            1,
            int(
                np.ceil(
                    len(oos)
                    * fraction
                )
            ),
        )
        top = oos.nlargest(
            n,
            "score",
        )
        print()
        print(
            f"Top {fraction * 100:.1f}% "
            f"(n={len(top)})"
        )
        for (
            label,
            lower,
            upper,
        ) in buckets:
            mask = (
                (
                    top[
                        "forward_return_5d"
                    ]
                    >= lower
                )
                & (
                    top[
                        "forward_return_5d"
                    ]
                    < upper
                )
            )
            print(
                f"  {label:12s} "
                f"{mask.mean():.4f}"
            )
def print_volatility_quintiles(
    oos: pd.DataFrame,
):
    print()
    print(
        "=" * 100
    )
    print(
        "VOLATILITY QUINTILES"
    )
    print(
        "=" * 100
    )
    for column in (
        "price_volatility_20d",
        "volatility_60d",
    ):
        frame = oos.dropna(
            subset=[
                column,
                "target_down_5pct_5d",
                "target_abs_5pct_5d",
            ]
        ).copy()
        if frame.empty:
            continue
        frame["quintile"] = pd.qcut(
            frame[column],
            5,
            labels=False,
            duplicates="drop",
        )
        print()
        print(column)
        for (
            quintile,
            group,
        ) in frame.groupby(
            "quintile",
            observed=True,
        ):
            print(
                f"  Q{int(quintile) + 1}: "
                f"n={len(group):6d} "
                f"median_vol="
                f"{group[column].median():.6f} "
                f"down="
                f"{group['target_down_5pct_5d'].mean():.4f} "
                f"abs5="
                f"{group['target_abs_5pct_5d'].mean():.4f}"
            )
def print_calendar_year_stability(
    oos: pd.DataFrame,
):
    print()
    print(
        "=" * 100
    )
    print(
        "CALENDAR-YEAR OOS STABILITY"
    )
    print(
        "=" * 100
    )
    frame = oos.copy()
    frame["year"] = pd.to_datetime(
        frame["snapshot_date"]
    ).dt.year
    for (
        year,
        group,
    ) in frame.groupby(
        "year"
    ):
        auc_down = safe_auc(
            group["score"],
            group[
                "target_down_5pct_5d"
            ],
        )
        auc_up = safe_auc(
            group["score"],
            group[
                "target_up_5pct_5d"
            ],
        )
        auc_abs5 = safe_auc(
            group["score"],
            group[
                "target_abs_5pct_5d"
            ],
        )
        print(
            f"{year}: "
            f"n={len(group):6d} "
            f"down_auc={auc_down:.6f} "
            f"up_auc={auc_up:.6f} "
            f"abs5_auc={auc_abs5:.6f} "
            f"down_rate="
            f"{group['target_down_5pct_5d'].mean():.4f} "
            f"abs5_rate="
            f"{group['target_abs_5pct_5d'].mean():.4f}"
        )
def print_score_deciles(
    oos: pd.DataFrame,
):
    print()
    print(
        "=" * 100
    )
    print(
        "SCORE DECILES"
    )
    print(
        "=" * 100
    )
    frame = oos.dropna(
        subset=[
            "score",
            "forward_return_5d",
        ]
    ).copy()
    if frame.empty:
        return
    frame["score_decile"] = pd.qcut(
        frame["score"],
        10,
        labels=False,
        duplicates="drop",
    )
    for (
        decile,
        group,
    ) in frame.groupby(
        "score_decile",
        observed=True,
    ):
        print(
            f"D{int(decile) + 1:02d}: "
            f"n={len(group):6d} "
            f"score_med="
            f"{group['score'].median():.5f} "
            f"down="
            f"{group['target_down_5pct_5d'].mean():.4f} "
            f"up="
            f"{group['target_up_5pct_5d'].mean():.4f} "
            f"abs5="
            f"{group['target_abs_5pct_5d'].mean():.4f} "
            f"abs10="
            f"{group['target_abs_10pct_5d'].mean():.4f} "
            f"mean_ret="
            f"{group['forward_return_5d'].mean():+.4%} "
            f"median_ret="
            f"{group['forward_return_5d'].median():+.4%}"
        )
def make_hashable_window_label(value) -> str:
    """
    Convert the walk-forward window metadata to a stable,
    human-readable string.
    train_window() currently returns window metadata as a dict
    in the OOS rows. Dicts cannot be used directly as pandas
    groupby keys because they are unhashable.
    """
    if isinstance(value, dict):
        return json.dumps(
            value,
            sort_keys=True,
            default=str,
        )
    if isinstance(value, (list, tuple)):
        return json.dumps(
            value,
            default=str,
        )
    return str(value)
def print_top1_by_window(
    oos: pd.DataFrame,
):
    print()
    print(
        "=" * 100
    )
    print(
        "TOP 1% BY WALK-FORWARD WINDOW"
    )
    print(
        "=" * 100
    )
    frame = oos.copy()
    if "window" not in frame.columns:
        raise ValueError(
            "OOS predictions are missing the "
            "'window' column."
        )
    # The walk-forward implementation stores window metadata
    # as a dict. pandas cannot group by dicts, so create a
    # separate hashable representation and leave the original
    # column untouched.
    frame["window_label"] = (
        frame["window"].map(
            make_hashable_window_label
        )
    )
    for (
        window,
        group,
    ) in frame.groupby(
        "window_label",
        dropna=False,
    ):
        top = describe_top_fraction(
            group,
            "target_down_5pct_5d",
            0.01,
        )
        auc_down = safe_auc(
            group["score"],
            group[
                "target_down_5pct_5d"
            ],
        )
        auc_up = safe_auc(
            group["score"],
            group[
                "target_up_5pct_5d"
            ],
        )
        auc_abs5 = safe_auc(
            group["score"],
            group[
                "target_abs_5pct_5d"
            ],
        )
        print(
            f"window {window}: "
            f"n={len(group):6d} "
            f"down_auc={auc_down:.6f} "
            f"up_auc={auc_up:.6f} "
            f"abs5_auc={auc_abs5:.6f} "
            f"top1_down="
            f"{top['event_rate']:.4f} "
            f"lift={top['lift']:.2f}x "
            f"mean={top['mean_return']:+.4%} "
            f"median={top['median_return']:+.4%}"
        )
def print_top1_volatility_profile(
    oos: pd.DataFrame,
):
    print()
    print(
        "=" * 100
    )
    print(
        "TOP 1% VOLATILITY / RETURN PROFILE"
    )
    print(
        "=" * 100
    )
    top = oos.nlargest(
        max(
            1,
            int(
                np.ceil(
                    len(oos)
                    * 0.01
                )
            ),
        ),
        "score",
    )
    for column in (
        "price_volatility_20d",
        "volatility_60d",
        "volatility_20d_minus_60d",
        "volatility_20d_div_60d",
        "price_return_5d",
        "price_return_20d",
        "price_return_60d",
    ):
        all_values = oos[
            column
        ].dropna()
        top_values = top[
            column
        ].dropna()
        if (
            all_values.empty
            or top_values.empty
        ):
            continue
        print(
            f"{column:32s} "
            f"all_median={all_values.median():+.6f} "
            f"top1_median={top_values.median():+.6f} "
            f"all_mean={all_values.mean():+.6f} "
            f"top1_mean={top_values.mean():+.6f}"
        )
def run_feature_set(
    data: pd.DataFrame,
    feature_set: str,
):
    columns = get_feature_columns(
        data,
        feature_set,
    )
    print()
    print(
        "#" * 100
    )
    print(
        f"RUNNING FEATURE SET: "
        f"{feature_set}"
    )
    print(
        f"Features ({len(columns)}): "
        f"{columns}"
    )
    print(
        "#" * 100
    )
    target = find_target(
        TRAIN_TARGET
    )
    (
        ml_data,
        y,
    ) = build_ml_dataset(
        data,
        columns,
        target,
    )
    print(
        f"ML rows: "
        f"{len(ml_data):,}"
    )
    started = time.perf_counter()
    oos = run_walk_forward(
        ml_data,
        y,
        columns,
    )
    elapsed = (
        time.perf_counter()
        - started
    )
    oos = attach_analysis_targets_to_oos(
        oos,
        data,
    )
    print(
        f"OOS rows: "
        f"{len(oos):,}"
    )
    print(
        f"Runtime: "
        f"{elapsed:.2f}s"
    )
    print_tail_analysis(
        oos,
        feature_set,
    )
    print_direction_given_move(
        oos
    )
    print_return_buckets(
        oos
    )
    print_volatility_quintiles(
        oos
    )
    print_calendar_year_stability(
        oos
    )
    print_score_deciles(
        oos
    )
    print_top1_by_window(
        oos
    )
    print_top1_volatility_profile(
        oos
    )
    return oos
def print_feature_set_comparison(
    outputs: dict[str, pd.DataFrame],
):
    print()
    print(
        "=" * 100
    )
    print(
        "FEATURE-SET COMPARISON"
    )
    print(
        "=" * 100
    )
    rows = []
    for (
        feature_set,
        oos,
    ) in outputs.items():
        auc_down = safe_auc(
            oos["score"],
            oos[
                "target_down_5pct_5d"
            ],
        )
        auc_up = safe_auc(
            oos["score"],
            oos[
                "target_up_5pct_5d"
            ],
        )
        auc_abs5 = safe_auc(
            oos["score"],
            oos[
                "target_abs_5pct_5d"
            ],
        )
        auc_abs10 = safe_auc(
            oos["score"],
            oos[
                "target_abs_10pct_5d"
            ],
        )
        top = describe_top_fraction(
            oos,
            "target_down_5pct_5d",
            0.01,
        )
        rows.append(
            {
                "feature_set": feature_set,
                "features": len(
                    FEATURE_SETS[
                        feature_set
                    ]
                ),
                "down_auc": auc_down,
                "up_auc": auc_up,
                "abs5_auc": auc_abs5,
                "abs10_auc": auc_abs10,
                "top1_down": (
                    top["event_rate"]
                ),
                "top1_lift": (
                    top["lift"]
                ),
                "top1_mean_return": (
                    top["mean_return"]
                ),
                "oos_rows": len(oos),
            }
        )
    comparison = pd.DataFrame(
        rows
    )
    print(
        comparison.to_string(
            index=False,
            float_format=(
                lambda value:
                f"{value:.6f}"
            ),
        )
    )
def print_qc(
    data: pd.DataFrame,
    prices: pd.DataFrame,
):
    print()
    print(
        "=" * 100
    )
    print(
        "DIAGNOSTIC QC"
    )
    print(
        "=" * 100
    )
    print(
        f"Feature rows: "
        f"{len(data):,}"
    )
    print(
        f"Raw price rows: "
        f"{len(prices):,}"
    )
    valid_20 = (
        data[
            "price_volatility_20d"
        ].notna()
    )
    valid_60 = (
        data[
            "volatility_60d"
        ].notna()
    )
    print(
        "Rows with 20d volatility: "
        f"{valid_20.sum():,}"
    )
    print(
        "Rows with 60d volatility: "
        f"{valid_60.sum():,}"
    )
    print(
        "Rows without 60d volatility: "
        f"{(~valid_60).sum():,}"
    )
    print(
        f"Target: {TRAIN_TARGET}"
    )
    print()
    print(
        "Volatility statistics:"
    )
    for column in (
        "price_volatility_20d",
        "volatility_60d",
        "volatility_20d_minus_60d",
        "volatility_20d_div_60d",
    ):
        series = data[
            column
        ].dropna()
        if series.empty:
            continue
        print(
            f"  {column:32s} "
            f"median={series.median():.6f} "
            f"p20={series.quantile(.20):.6f} "
            f"p80={series.quantile(.80):.6f}"
        )
def main():
    print(
        "=" * 100
    )
    print(
        "VOLATILITY DIRECTIONAL / "
        "TAIL DIAGNOSTIC"
    )
    print(
        "=" * 100
    )
    started = time.perf_counter()
    features = load_features()
    if not isinstance(
        features,
        pd.DataFrame,
    ):
        raise TypeError(
            "load_features() did not "
            "return a DataFrame."
        )
    print(
        f"Loaded feature rows: "
        f"{len(features):,}"
    )
    prices = load_price_data()
    data = add_volatility_features(
        features,
        prices,
    )
    data = add_analysis_targets(
        data
    )
    print_qc(
        data,
        prices,
    )
    outputs = {}
    for feature_set in (
        "volatility_60d",
        "volatility_20d_plus_60d",
        "volatility_60d_plus_term_structure",
        "volatility_20d_plus_60d_plus_term_structure",
        "volatility_20d_plus_60d_plus_ratio",
        "volatility_plus_returns",
        "volatility_20d_plus_60d_plus_returns",
    ):
        outputs[
            feature_set
        ] = run_feature_set(
            data,
            feature_set,
        )
    print_feature_set_comparison(
        outputs
    )
    print()
    print(
        "=" * 100
    )
    print(
        "FINAL SUMMARY"
    )
    print(
        "=" * 100
    )
    for (
        feature_set,
        oos,
    ) in outputs.items():
        down_auc = safe_auc(
            oos["score"],
            oos[
                "target_down_5pct_5d"
            ],
        )
        up_auc = safe_auc(
            oos["score"],
            oos[
                "target_up_5pct_5d"
            ],
        )
        abs5_auc = safe_auc(
            oos["score"],
            oos[
                "target_abs_5pct_5d"
            ],
        )
        abs10_auc = safe_auc(
            oos["score"],
            oos[
                "target_abs_10pct_5d"
            ],
        )
        top = describe_top_fraction(
            oos,
            "target_down_5pct_5d",
            0.01,
        )
        print(
            f"{feature_set:48s} "
            f"down={down_auc:.6f} "
            f"up={up_auc:.6f} "
            f"abs5={abs5_auc:.6f} "
            f"abs10={abs10_auc:.6f} "
            f"top1_down={top['event_rate']:.4f} "
            f"lift={top['lift']:.2f}x "
            f"mean={top['mean_return']:+.4%}"
        )
    elapsed = (
        time.perf_counter()
        - started
    )
    print()
    print(
        f"Total runtime: "
        f"{elapsed:.2f}s"
    )
if __name__ == "__main__":
    main()
