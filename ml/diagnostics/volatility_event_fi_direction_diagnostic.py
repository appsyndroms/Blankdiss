"""
Volatility event + FI direction diagnostic.
Purpose
-------
Separate two different questions:
1. EVENT:
   Can volatility identify stocks where a large move is likely?
2. DIRECTION:
   Given that a large move occurs, can FI short-interest data
   distinguish DOWN from UP?
The diagnostic deliberately separates these questions.
EVENT MODEL
-----------
- abs_10pct_5d
- volatility_20d
- volatility_60d
- volatility term structure
- volatility ratio
DIRECTION MODEL
---------------
- FI only
- event-risk score only
- FI + event-risk score
- FI + event-risk interaction terms
MATCHED TAIL
------------
Direction models are also trained only on historical large-move
events that belonged to the same event-risk tail being evaluated.
Tails:
    1%
    2%
    5%
FI groups
---------
- FI levels
- FI changes
- FI acceleration
- concentration
- threshold / state features
- all FI features
PLACEBO
-------
FI values are shuffled within the training period only.
This checks whether apparent FI direction signal survives a
simple label-preserving placebo test.
HORIZONS
--------
The diagnostic also reports the realized direction / magnitude
for 1d, 5d and 20d returns inside the 5d event-risk tail.
WALK-FORWARD
------------
Uses the project's existing chronological windows.
No repository files are modified by this diagnostic.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from analysis.feature_config import PRICE_DIR
from analysis.feature_prices import find_price_files, load_prices
from ml.config import WALK_FORWARD_WINDOWS
from ml.dataset import get_feature_columns, load_features
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)
TOP_FRACTIONS = (
    0.01,
    0.02,
    0.05,
)
PLACEBO_SEEDS = (
    101,
    202,
    303,
    404,
    505,
)
EVENT_FEATURE_SETS = {
    "volatility_60d": [
        "volatility_60d",
    ],
    "volatility_20d_plus_60d": [
        "price_volatility_20d",
        "volatility_60d",
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
}
FI_GROUPS = {
    "fi_levels": [
        "short_interest_pct",
        "active_holders",
        "max_individual_position_pct",
        "max_position_share_pct",
        "previous_short_interest_pct",
        "previous_active_holders",
        "previous_max_individual_position_pct",
        "previous_max_position_share_pct",
    ],
    "fi_changes": [
        "short_interest_delta_pp",
        "holder_delta",
        "max_position_delta_pp",
        "concentration_delta_pp",
        "short_interest_relative_change",
    ],
    "fi_acceleration": [
        "short_interest_acceleration_pp",
    ],
    "fi_state": [
        "above_1_0pct",
        "entered_above_1_0pct",
        "exited_below_1_0pct",
        "above_2_0pct",
        "entered_above_2_0pct",
        "exited_below_2_0pct",
        "above_3_0pct",
        "entered_above_3_0pct",
        "exited_below_3_0pct",
        "above_5_0pct",
        "entered_above_5_0pct",
        "exited_below_5_0pct",
        "new_visible_observation",
    ],
}
FI_CORE_COLUMNS = [
    "short_interest_pct",
    "active_holders",
    "max_individual_position_pct",
    "max_position_share_pct",
    "previous_short_interest_pct",
    "previous_active_holders",
    "previous_max_individual_position_pct",
    "previous_max_position_share_pct",
    "short_interest_delta_pp",
    "holder_delta",
    "max_position_delta_pp",
    "concentration_delta_pp",
    "short_interest_relative_change",
    "short_interest_acceleration_pp",
]
VOLATILITY_COLUMNS = [
    "price_volatility_20d",
    "volatility_60d",
    "volatility_20d_minus_60d",
    "volatility_20d_div_60d",
]
def load_price_data() -> pd.DataFrame:
    print()
    print("=" * 80)
    print("LOADING RAW PRICE DATA")
    print("=" * 80)
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
    returns = pd.to_numeric(
        data["forward_return_5d"],
        errors="coerce",
    )
    data["target_abs_10pct_5d"] = (
        returns.abs() >= 0.10
    )
    data["target_down_10pct_5d"] = (
        returns <= -0.10
    )
    data["target_up_10pct_5d"] = (
        returns >= 0.10
    )
    return data
def make_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
def safe_auc(
    y_true: pd.Series,
    score: pd.Series,
) -> float:
    valid = (
        y_true.notna()
        & score.notna()
    )
    y_true = y_true.loc[valid]
    score = score.loc[valid]
    if len(y_true) == 0:
        return float("nan")
    if y_true.nunique() < 2:
        return float("nan")
    return float(
        roc_auc_score(
            y_true,
            score,
        )
    )
def fit_event_model(
    train: pd.DataFrame,
    feature_columns: list[str],
) -> Pipeline:
    model = make_model()
    model.fit(
        train[feature_columns],
        train["target_abs_10pct_5d"].astype(int),
    )
    return model
def event_scores(
    model: Pipeline,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    return pd.Series(
        model.predict_proba(
            frame[feature_columns]
        )[:, 1],
        index=frame.index,
        dtype=float,
    )
def top_tail(
    frame: pd.DataFrame,
    score_column: str,
    fraction: float,
) -> pd.DataFrame:
    frame = frame.dropna(
        subset=[score_column]
    ).copy()
    if frame.empty:
        return frame
    count = max(
        1,
        int(
            np.ceil(
                len(frame) * fraction
            )
        ),
    )
    return (
        frame
        .nlargest(
            count,
            score_column,
        )
        .copy()
    )
def direction_target(
    frame: pd.DataFrame,
) -> pd.Series:
    returns = pd.to_numeric(
        frame["forward_return_5d"],
        errors="coerce",
    )
    result = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )
    result.loc[
        returns <= -0.10
    ] = 1.0
    result.loc[
        returns >= 0.10
    ] = 0.0
    return result
def large_move_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    returns = pd.to_numeric(
        frame["forward_return_5d"],
        errors="coerce",
    )
    return frame.loc[
        returns.abs() >= 0.10
    ].copy()
def fit_direction_model(
    train: pd.DataFrame,
    feature_columns: list[str],
) -> Pipeline | None:
    if not feature_columns:
        return None
    train = train.copy()
    train["direction_target"] = (
        direction_target(train)
    )
    train = train.dropna(
        subset=[
            "direction_target",
        ]
    )
    if len(train) < 50:
        return None
    if (
        train["direction_target"]
        .nunique()
        < 2
    ):
        return None
    model = make_model()
    model.fit(
        train[feature_columns],
        train["direction_target"].astype(int),
    )
    return model
def direction_score(
    model: Pipeline | None,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    if model is None:
        return pd.Series(
            np.nan,
            index=frame.index,
            dtype=float,
        )
    return pd.Series(
        model.predict_proba(
            frame[feature_columns]
        )[:, 1],
        index=frame.index,
        dtype=float,
    )
def describe_direction_tail(
    frame: pd.DataFrame,
    score_column: str,
    fraction: float,
    label: str,
) -> dict[str, float]:
    tail = top_tail(
        frame,
        score_column,
        fraction,
    )
    if tail.empty:
        return {
            "count": 0,
            "down_rate": float("nan"),
            "mean_return": float("nan"),
            "median_return": float("nan"),
            "positive_rate": float("nan"),
            "negative_rate": float("nan"),
        }
    returns = pd.to_numeric(
        tail["forward_return_5d"],
        errors="coerce",
    ).dropna()
    down_rate = (
        returns <= -0.10
    ).mean()
    return {
        "count": int(len(tail)),
        "down_rate": float(down_rate),
        "mean_return": float(
            returns.mean()
        ),
        "median_return": float(
            returns.median()
        ),
        "positive_rate": float(
            (returns > 0).mean()
        ),
        "negative_rate": float(
            (returns < 0).mean()
        ),
    }
def print_tail_stats(
    frame: pd.DataFrame,
    score_column: str,
    fraction: float,
    label: str,
) -> None:
    stats = describe_direction_tail(
        frame,
        score_column,
        fraction,
        label,
    )
    print(
        f"    {label:<28}"
        f" n={stats['count']:>6,}"
        f" down={stats['down_rate']:.4f}"
        f" mean={stats['mean_return']:+.4%}"
        f" median={stats['median_return']:+.4%}"
    )
def print_horizon_stats(
    tail: pd.DataFrame,
) -> None:
    print()
    print(
        "  Realized direction inside tail:"
    )
    for days, column in (
        (1, "forward_return_1d"),
        (5, "forward_return_5d"),
        (20, "forward_return_20d"),
    ):
        if column not in tail.columns:
            continue
        returns = pd.to_numeric(
            tail[column],
            errors="coerce",
        ).dropna()
        if returns.empty:
            continue
        print(
            f"    {days:>2}d:"
            f" mean={returns.mean():+.4%}"
            f" median={returns.median():+.4%}"
            f" positive={(returns > 0).mean():.4f}"
            f" negative={(returns < 0).mean():.4f}"
        )
def print_volatility_profile(
    tail: pd.DataFrame,
    all_rows: pd.DataFrame,
) -> None:
    print()
    print(
        "  Volatility profile:"
    )
    for column in (
        "price_volatility_20d",
        "volatility_60d",
        "volatility_20d_minus_60d",
        "volatility_20d_div_60d",
    ):
        if column not in tail.columns:
            continue
        tail_value = pd.to_numeric(
            tail[column],
            errors="coerce",
        ).median()
        all_value = pd.to_numeric(
            all_rows[column],
            errors="coerce",
        ).median()
        print(
            f"    {column:<34}"
            f" tail={tail_value:.6f}"
            f" all={all_value:.6f}"
        )
def print_fi_profile(
    tail: pd.DataFrame,
) -> None:
    print()
    print(
        "  FI profile:"
    )
    for column in (
        "short_interest_pct",
        "short_interest_delta_pp",
        "short_interest_acceleration_pp",
        "concentration_delta_pp",
        "active_holders",
        "max_position_delta_pp",
    ):
        if column not in tail.columns:
            continue
        value = pd.to_numeric(
            tail[column],
            errors="coerce",
        ).median()
        print(
            f"    {column:<38}"
            f" median={value:.6f}"
        )
def fi_columns_from_original(
    original: pd.DataFrame,
) -> list[str]:
    columns = get_feature_columns(
        original,
        include_price_features=False,
    )
    excluded = {
        "snapshot_date",
        "security_key",
    }
    return [
        column
        for column in columns
        if column not in excluded
        and column in original.columns
    ]
def available_fi_groups(
    fi_columns: list[str],
) -> dict[str, list[str]]:
    available = {}
    for group_name, columns in FI_GROUPS.items():
        selected = [
            column
            for column in columns
            if column in fi_columns
        ]
        if selected:
            available[group_name] = selected
    available["fi_all"] = list(
        fi_columns
    )
    return available
def print_fi_group_screen(
    train_events: pd.DataFrame,
    test_events: pd.DataFrame,
    test_tail: pd.DataFrame,
    groups: dict[str, list[str]],
) -> None:
    print()
    print(
        "  FI directional feature-group screen:"
    )
    for group_name, columns in groups.items():
        model = fit_direction_model(
            train_events,
            columns,
        )
        if model is None:
            print(
                f"    {group_name:<24}"
                " unavailable"
            )
            continue
        score = direction_score(
            model,
            test_tail,
            columns,
        )
        valid = (
            test_tail["direction_target"]
            .notna()
            & score.notna()
        )
        if valid.sum() == 0:
            continue
        auc = safe_auc(
            test_tail.loc[
                valid,
                "direction_target",
            ],
            score.loc[valid],
        )
        predictions = (
            score.loc[valid] >= 0.5
        ).astype(int)
        accuracy = accuracy_score(
            test_tail.loc[
                valid,
                "direction_target",
            ],
            predictions,
        )
        print(
            f"    {group_name:<24}"
            f"AUC={auc:.6f}"
            f" accuracy={accuracy:.4f}"
            f" n={valid.sum():,}"
        )
def add_direction_target(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    result["direction_target"] = (
        direction_target(result)
    )
    return result
def add_event_interactions(
    frame: pd.DataFrame,
    event_score_column: str,
    fi_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    selected = [
        column
        for column in (
            "short_interest_pct",
            "short_interest_delta_pp",
            "short_interest_acceleration_pp",
            "concentration_delta_pp",
            "max_position_delta_pp",
        )
        if column in fi_columns
    ]
    interaction_columns = []
    for column in selected:
        name = (
            f"event_x_{column}"
        )
        result[name] = (
            pd.to_numeric(
                result[event_score_column],
                errors="coerce",
            )
            * pd.to_numeric(
                result[column],
                errors="coerce",
            )
        )
        interaction_columns.append(
            name
        )
    return (
        result,
        interaction_columns,
    )
def run_placebo(
    train_events: pd.DataFrame,
    test_tail: pd.DataFrame,
    fi_columns: list[str],
    seed: int,
) -> float:
    rng = np.random.default_rng(
        seed
    )
    shuffled = train_events.copy()
    for column in fi_columns:
        values = (
            shuffled[column]
            .to_numpy(
                copy=True
            )
        )
        rng.shuffle(values)
        shuffled[column] = values
    model = fit_direction_model(
        shuffled,
        fi_columns,
    )
    if model is None:
        return float("nan")
    score = direction_score(
        model,
        test_tail,
        fi_columns,
    )
    valid = (
        test_tail["direction_target"]
        .notna()
        & score.notna()
    )
    if valid.sum() == 0:
        return float("nan")
    return safe_auc(
        test_tail.loc[
            valid,
            "direction_target",
        ],
        score.loc[valid],
    )
def print_placebo(
    train_events: pd.DataFrame,
    test_tail: pd.DataFrame,
    fi_columns: list[str],
) -> None:
    values = []
    for seed in PLACEBO_SEEDS:
        auc = run_placebo(
            train_events,
            test_tail,
            fi_columns,
            seed,
        )
        if np.isfinite(auc):
            values.append(auc)
    if not values:
        print(
            "    Placebo: unavailable"
        )
        return
    print(
        "    Placebo FI AUC:"
        f" mean={np.mean(values):.6f}"
        f" min={np.min(values):.6f}"
        f" max={np.max(values):.6f}"
    )
def print_top_fi_direction(
    train_events: pd.DataFrame,
    test_tail: pd.DataFrame,
    fi_columns: list[str],
) -> None:
    rows = []
    for column in fi_columns:
        raw_values = train_events[column]
        # A boolean feature cannot be passed directly to
        # Series.quantile() because NumPy may attempt boolean
        # subtraction during interpolation.
        #
        # For binary/state features we therefore test the
        # positive state directly instead of inventing a
        # percentile threshold.
        numeric_values = pd.to_numeric(
            raw_values,
            errors="coerce",
        )
        if numeric_values.notna().sum() < 100:
            continue
        unique_values = (
            numeric_values
            .dropna()
            .unique()
        )
        unique_values = np.sort(
            unique_values
        )
        if len(unique_values) <= 2:
            if len(unique_values) == 1:
                positive_value = unique_values[0]
            else:
                positive_value = unique_values[-1]
            selected_train = train_events.loc[
                numeric_values == positive_value
            ].copy()
        else:
            # Quantile is now calculated on a real numeric
            # Series, never on bool/object values.
            threshold = numeric_values.quantile(
                0.95
            )
            selected_train = train_events.loc[
                numeric_values >= threshold
            ].copy()
        if len(selected_train) < 50:
            continue
        model = fit_direction_model(
            selected_train,
            [column],
        )
        if model is None:
            continue
        score = direction_score(
            model,
            test_tail,
            [column],
        )
        valid = (
            test_tail["direction_target"]
            .notna()
            & score.notna()
        )
        if valid.sum() == 0:
            continue
        auc = safe_auc(
            test_tail.loc[
                valid,
                "direction_target",
            ],
            score.loc[valid],
        )
        rows.append(
            (
                column,
                auc,
                len(selected_train),
            )
        )
    rows.sort(
        key=lambda item: (
            -item[1]
            if np.isfinite(item[1])
            else np.inf
        )
    )
    print()
    print(
        "  Individual FI feature direction screen:"
    )
    for column, auc, train_count in rows[:15]:
        print(
            f"    {column:<40}"
            f"AUC={auc:.6f}"
            f" train={train_count:,}"
        )
def evaluate_window(
    original: pd.DataFrame,
    data: pd.DataFrame,
    fi_columns: list[str],
    window_number: int,
    window,
) -> None:
    train = data.loc[
        data["snapshot_date"]
        <= pd.Timestamp(
            window.train_end
        )
    ].copy()
    validation = data.loc[
        (
            data["snapshot_date"]
            > pd.Timestamp(
                window.train_end
            )
        )
        & (
            data["snapshot_date"]
            <= pd.Timestamp(
                window.validation_end
            )
        )
    ].copy()
    test = data.loc[
        (
            data["snapshot_date"]
            > pd.Timestamp(
                window.validation_end
            )
        )
        & (
            data["snapshot_date"]
            <= pd.Timestamp(
                window.test_end
            )
        )
    ].copy()
    print()
    print("=" * 80)
    print(
        f"WINDOW {window_number}"
    )
    print("=" * 80)
    print(
        f"Train:      {len(train):,}"
    )
    print(
        f"Validation: {len(validation):,}"
    )
    print(
        f"Test:       {len(test):,}"
    )
    event_train = train.dropna(
        subset=[
            *set().union(
                *EVENT_FEATURE_SETS.values()
            ),
            "forward_return_5d",
        ]
    ).copy()
    event_validation = validation.dropna(
        subset=[
            *set().union(
                *EVENT_FEATURE_SETS.values()
            ),
            "forward_return_5d",
        ]
    ).copy()
    event_test = test.dropna(
        subset=[
            *set().union(
                *EVENT_FEATURE_SETS.values()
            ),
            "forward_return_5d",
        ]
    ).copy()
    if len(event_train) < 100:
        print(
            "Not enough training rows."
        )
        return
    event_results = []
    for name, columns in EVENT_FEATURE_SETS.items():
        if any(
            column not in event_train.columns
            for column in columns
        ):
            continue
        train_valid = event_train.dropna(
            subset=columns
        ).copy()
        validation_valid = event_validation.dropna(
            subset=columns
        ).copy()
        if (
            len(train_valid) < 100
            or len(validation_valid) < 100
        ):
            continue
        model = fit_event_model(
            train_valid,
            columns,
        )
        score = event_scores(
            model,
            validation_valid,
            columns,
        )
        auc = safe_auc(
            validation_valid[
                "target_abs_10pct_5d"
            ],
            score,
        )
        event_results.append(
            (
                name,
                columns,
                model,
                auc,
            )
        )
    event_results.sort(
        key=lambda item: (
            -item[3]
            if np.isfinite(item[3])
            else np.inf
        )
    )
    print()
    print(
        "EVENT MODEL:"
    )
    for (
        name,
        _,
        _,
        auc,
    ) in event_results:
        print(
            f"  {name:<46}"
            f"AUC={auc:.6f}"
        )
    if not event_results:
        print(
            "No event model available."
        )
        return
    (
        selected_name,
        selected_columns,
        _,
        selected_auc,
    ) = event_results[0]
    print()
    print(
        f"Selected event model: "
        f"{selected_name}"
        f" (validation AUC={selected_auc:.6f})"
    )
    event_train_valid = event_train.dropna(
        subset=selected_columns
    ).copy()
    event_validation_valid = (
        event_validation.dropna(
            subset=selected_columns
        ).copy()
    )
    event_test_valid = event_test.dropna(
        subset=selected_columns
    ).copy()
    selected_model = fit_event_model(
        event_train_valid,
        selected_columns,
    )
    event_train_valid["event_score"] = (
        event_scores(
            selected_model,
            event_train_valid,
            selected_columns,
        )
    )
    event_validation_valid[
        "event_score"
    ] = event_scores(
        selected_model,
        event_validation_valid,
        selected_columns,
    )
    event_test_valid["event_score"] = (
        event_scores(
            selected_model,
            event_test_valid,
            selected_columns,
        )
    )
    print()
    print(
        "TEST EVENT MODEL:"
    )
    test_event_auc = safe_auc(
        event_test_valid[
            "target_abs_10pct_5d"
        ],
        event_test_valid["event_score"],
    )
    print(
        f"  AUC abs10 = {test_event_auc:.6f}"
    )
    test_all = event_test_valid.copy()
    print()
    print(
        "EVENT-RISK TAILS:"
    )
    for fraction in TOP_FRACTIONS:
        tail = top_tail(
            test_all,
            "event_score",
            fraction,
        )
        print()
        print(
            f"  TOP {fraction:.1%}"
            f" n={len(tail):,}"
        )
        if tail.empty:
            continue
        returns = pd.to_numeric(
            tail["forward_return_5d"],
            errors="coerce",
        ).dropna()
        base_event_rate = (
            test_all[
                "target_abs_10pct_5d"
            ].mean()
        )
        tail_event_rate = (
            tail[
                "target_abs_10pct_5d"
            ].mean()
        )
        print(
            f"    abs10 event rate="
            f"{tail_event_rate:.4f}"
            f"  baseline={base_event_rate:.4f}"
            f"  lift="
            f"{tail_event_rate / base_event_rate:.2f}x"
        )
        print(
            f"    down among abs10="
            f"{(returns <= -0.10).mean():.4f}"
        )
        print_horizon_stats(
            tail
        )
        print_volatility_profile(
            tail,
            test_all,
        )
    print()
    print(
        "DIRECTIONAL FI TESTS:"
    )
    for fraction in TOP_FRACTIONS:
        train_tail = top_tail(
            event_train_valid,
            "event_score",
            fraction,
        )
        test_tail = top_tail(
            event_test_valid,
            "event_score",
            fraction,
        )
        train_events = large_move_rows(
            train_tail
        )
        test_events = large_move_rows(
            test_tail
        )
        train_events = add_direction_target(
            train_events
        )
        test_events = add_direction_target(
            test_events
        )
        if len(train_events) < 50:
            continue
        print()
        print(
            "-" * 80
        )
        print(
            f"TAIL {fraction:.1%}"
        )
        print(
            "-" * 80
        )
        print(
            f"  Historical matched events:"
            f" {len(train_events):,}"
        )
        print(
            f"  Test matched events:"
            f" {len(test_events):,}"
        )
        if test_events.empty:
            continue
        test_down_rate = (
            test_events[
                "direction_target"
            ].mean()
        )
        print(
            f"  Actual test DOWN rate:"
            f" {test_down_rate:.4f}"
        )
        print_fi_group_screen(
            train_events,
            test_events,
            test_events,
            available_fi_groups(
                fi_columns
            ),
        )
        print_top_fi_direction(
            train_events,
            test_events,
            fi_columns,
        )
        fi_model = fit_direction_model(
            train_events,
            fi_columns,
        )
        if fi_model is not None:
            fi_score = direction_score(
                fi_model,
                test_events,
                fi_columns,
            )
            fi_auc = safe_auc(
                test_events[
                    "direction_target"
                ],
                fi_score,
            )
            print()
            print(
                f"  MATCHED FI direction AUC:"
                f" {fi_auc:.6f}"
            )
        event_only_model = fit_direction_model(
            train_events,
            ["event_score"],
        )
        if event_only_model is not None:
            event_score = direction_score(
                event_only_model,
                test_events,
                ["event_score"],
            )
            event_direction_auc = safe_auc(
                test_events[
                    "direction_target"
                ],
                event_score,
            )
            print(
                f"  EVENT SCORE direction AUC:"
                f" {event_direction_auc:.6f}"
            )
        combined_train = train_events.copy()
        combined_test = test_events.copy()
        combined_columns = [
            "event_score",
            *fi_columns,
        ]
        combined_model = fit_direction_model(
            combined_train,
            combined_columns,
        )
        if combined_model is not None:
            combined_score = direction_score(
                combined_model,
                combined_test,
                combined_columns,
            )
            combined_auc = safe_auc(
                combined_test[
                    "direction_target"
                ],
                combined_score,
            )
            print(
                f"  EVENT + FI direction AUC:"
                f" {combined_auc:.6f}"
            )
        interaction_train, interaction_columns = (
            add_event_interactions(
                combined_train,
                "event_score",
                fi_columns,
            )
        )
        interaction_test, _ = (
            add_event_interactions(
                combined_test,
                "event_score",
                fi_columns,
            )
        )
        interaction_features = [
            "event_score",
            *fi_columns,
            *interaction_columns,
        ]
        interaction_model = fit_direction_model(
            interaction_train,
            interaction_features,
        )
        if interaction_model is not None:
            interaction_score = direction_score(
                interaction_model,
                interaction_test,
                interaction_features,
            )
            interaction_auc = safe_auc(
                interaction_test[
                    "direction_target"
                ],
                interaction_score,
            )
            print(
                f"  EVENT + FI + interactions:"
                f" AUC={interaction_auc:.6f}"
            )
        print()
        print(
            "  PLACEBO:"
        )
        print_placebo(
            train_events,
            test_events,
            fi_columns,
        )
        print()
        print(
            "  Tail economic profile:"
        )
        returns = pd.to_numeric(
            test_events[
                "forward_return_5d"
            ],
            errors="coerce",
        ).dropna()
        print(
            f"    mean={returns.mean():+.4%}"
            f" median={returns.median():+.4%}"
            f" positive={(returns > 0).mean():.4f}"
            f" negative={(returns < 0).mean():.4f}"
        )
        print_fi_profile(
            test_events
        )
    print()
    print(
        "DIRECTIONAL INTERPRETATION:"
    )
    print(
        "  If FI-only and EVENT+FI have similar AUC:"
    )
    print(
        "    FI adds little beyond the existing event-risk ranking."
    )
    print(
        "  If EVENT+FI materially exceeds EVENT SCORE:"
    )
    print(
        "    FI contains directional information conditional"
        " on large-move risk."
    )
    print(
        "  If matched-tail FI survives while the placebo"
        " remains near 0.50:"
    )
    print(
        "    that is substantially stronger evidence for a"
        " genuine FI directional signal."
    )
    print(
        "  If FI helps only at 1% but not 5%:"
    )
    print(
        "    the signal is concentrated in the extreme tail."
    )
    print(
        "  If FI helps at 5% but disappears at 1%:"
    )
    print(
        "    the signal is broader rather than extreme-tail specific."
    )
def print_data_summary(
    data: pd.DataFrame,
    fi_columns: list[str],
) -> None:
    print()
    print("=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(
        f"Rows: {len(data):,}"
    )
    print(
        f"Date range: "
        f"{data['snapshot_date'].min().date()}"
        f" -> "
        f"{data['snapshot_date'].max().date()}"
    )
    print(
        f"FI features: {len(fi_columns)}"
    )
    print(
        f"20d volatility available:"
        f" {data['price_volatility_20d'].notna().sum():,}"
    )
    print(
        f"60d volatility available:"
        f" {data['volatility_60d'].notna().sum():,}"
    )
    print(
        f"abs10 base rate:"
        f" {data['target_abs_10pct_5d'].mean():.4f}"
    )
    print(
        f"down10 base rate:"
        f" {data['target_down_10pct_5d'].mean():.4f}"
    )
    print(
        f"up10 base rate:"
        f" {data['target_up_10pct_5d'].mean():.4f}"
    )
def normalize_windows(
    windows,
) -> list:
    return list(windows)
def main() -> None:
    print(
        "=" * 80
    )
    print(
        "BLANKDISS — VOLATILITY EVENT + FI DIRECTION DIAGNOSTIC"
    )
    print(
        "=" * 80
    )
    original = load_features()
    duplicate_columns = (
        original.columns[
            original.columns.duplicated()
        ]
        .tolist()
    )
    if duplicate_columns:
        print()
        print(
            "Duplicate feature columns detected;"
            " keeping first occurrence:"
        )
        print(
            f"  {duplicate_columns}"
        )
        original = original.loc[
            :,
            ~original.columns.duplicated(
                keep="first"
            ),
        ].copy()
    print(
        "Loading raw prices..."
    )
    prices = load_price_data()
    data = add_volatility_features(
        original,
        prices,
    )
    data = add_analysis_targets(
        data
    )
    data["snapshot_date"] = pd.to_datetime(
        data["snapshot_date"],
        errors="coerce",
    )
    fi_columns = fi_columns_from_original(
        original
    )
    available_groups = available_fi_groups(
        fi_columns
    )
    print_data_summary(
        data,
        fi_columns,
    )
    print()
    print(
        "FI groups available:"
    )
    for name, columns in available_groups.items():
        print(
            f"  {name:<24}"
            f"{len(columns):>3} features"
        )
    for (
        window_number,
        window,
    ) in enumerate(
        normalize_windows(
            WALK_FORWARD_WINDOWS
        ),
        start=1,
    ):
        evaluate_window(
            original=original,
            data=data,
            fi_columns=fi_columns,
            window_number=window_number,
            window=window,
        )
    print()
    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
if __name__ == "__main__":
    main()
