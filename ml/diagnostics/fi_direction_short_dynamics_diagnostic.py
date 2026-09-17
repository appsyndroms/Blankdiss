"""
Conditional FI short-interest dynamics diagnostic.
Question:
    Within an already identified event-risk tail, does short interest
    distinguish DOWN from UP?
FI variables:
    1. short_interest_pct
    2. short_interest_pct_change
    3. short_interest_pct_change_pct
The diagnostic deliberately separates:
1. Event risk:
       Which observations are likely to experience a large absolute
       5-day move?
2. Event direction:
       Among those high-risk observations, is short interest associated
       with DOWN versus UP?
The event-risk population is defined by an event-score tail.
All thresholds are calculated ONLY from the training period and then
applied unchanged to the OOS test period.
For short_interest_pct:
    - LOW / HIGH split at the training median.
For short_interest_pct_change and short_interest_pct_change_pct:
    A. Change direction:
        DECREASE
        NO_CHANGE
        INCREASE
       The primary comparison is:
           INCREASE_DOWN_RATE - DECREASE_DOWN_RATE
    B. Change magnitude:
        Among non-zero changes, split absolute change at the training
        median.
       The comparison is:
           HIGH_MAGNITUDE_DOWN_RATE - LOW_MAGNITUDE_DOWN_RATE
This avoids the previous problem where a median of zero effectively
turned the change variables into "any increase vs everything else".
The absolute and relative change variables are evaluated separately.
No repository files are modified by this script.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from analysis.feature_config import PRICE_DIR
from analysis.feature_prices import find_price_files, load_prices
from ml.config import WALK_FORWARD_WINDOWS
from ml.dataset import load_features
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BOOTSTRAP_ITERATIONS = 2_000
RANDOM_STATE = 42
EVENT_THRESHOLD = 0.10
EVENT_HORIZON_DAYS = 5
TOP_FRACTIONS = (
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
)
FI_COLUMNS = (
    "short_interest_pct",
    "short_interest_pct_change",
    "short_interest_pct_change_pct",
)
BASE_FI_COLUMN = "short_interest_pct"
OUTPUT_FILE = (
    "ml/diagnostics/fi_direction_short_dynamics_results.csv"
)
# Candidate features for the event model.
EVENT_FEATURE_SETS = {
    "volatility_20d": (
        "price_volatility_20d",
    ),
    "volatility_60d": (
        "volatility_60d",
    ),
    "volatility_20d_plus_60d": (
        "price_volatility_20d",
        "volatility_60d",
    ),
    "volatility_20d_plus_60d_plus_term_structure": (
        "price_volatility_20d",
        "volatility_60d",
        "volatility_term_structure",
    ),
}
# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DiagnosticResult:
    window: str
    fi_variable: str
    analysis: str
    tail: float
    training_tail_events: int
    test_tail_events: int
    usable_test_events: int
    fi_threshold: float
    low_events: int
    high_events: int
    low_down_events: int
    low_up_events: int
    high_down_events: int
    high_up_events: int
    low_down_rate: float
    high_down_rate: float
    observed_delta_down_rate: float
    bootstrap_mean_delta_down_rate: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    bootstrap_probability_positive: float
    bootstrap_probability_non_positive: float
    auc_fi: float
@dataclass(frozen=True)
class ChangeDirectionResult:
    window: str
    fi_variable: str
    tail: float
    training_tail_events: int
    test_tail_events: int
    decrease_events: int
    decrease_down_events: int
    decrease_up_events: int
    decrease_down_rate: float
    no_change_events: int
    no_change_down_events: int
    no_change_up_events: int
    no_change_down_rate: float
    increase_events: int
    increase_down_events: int
    increase_up_events: int
    increase_down_rate: float
    observed_increase_minus_decrease: float
    bootstrap_mean_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    bootstrap_probability_positive: float
    bootstrap_probability_non_positive: float
    auc_fi: float
# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def first_existing_column(
    data: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    """Return the first candidate column present in data."""
    for column in candidates:
        if column in data.columns:
            return column
    return None
def normalize_date_column(
    data: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Normalize dates to YYYY-MM-DD strings."""
    return pd.to_datetime(
        data[column],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
def deduplicate_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep the first occurrence of duplicate column names.
    Duplicate names are dangerous because:
        data[column]
    returns a DataFrame instead of a Series.
    """
    duplicated = data.columns.duplicated(
        keep="first"
    )
    if duplicated.any():
        duplicate_names = (
            data.columns[duplicated].tolist()
        )
        print()
        print(
            "Duplicate feature columns detected; "
            "keeping first occurrence:"
        )
        print(f"  {duplicate_names}")
        data = data.loc[
            :,
            ~duplicated,
        ].copy()
    return data
def numeric_series(
    data: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Convert one column to numeric safely.
    Boolean columns are explicitly converted to float.
    """
    values = data[column]
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    is_boolean = pd.api.types.is_bool_dtype(
        values
    )
    values = pd.to_numeric(
        values,
        errors="coerce",
    )
    if is_boolean:
        values = values.astype(float)
    return values
def safe_auc(
    y_true: pd.Series,
    scores: pd.Series,
) -> float:
    """Calculate AUC or return NaN when undefined."""
    frame = pd.DataFrame(
        {
            "y": y_true,
            "score": scores,
        }
    ).dropna()
    if len(frame) < 2:
        return float("nan")
    if frame["y"].nunique() < 2:
        return float("nan")
    if frame["score"].nunique() < 2:
        return 0.5
    return float(
        roc_auc_score(
            frame["y"],
            frame["score"],
        )
    )
# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------
def load_price_data() -> pd.DataFrame:
    """Load raw price data."""
    price_files = find_price_files(
        PRICE_DIR
    )
    if not price_files:
        raise RuntimeError(
            "No price files found by "
            "analysis.feature_prices.find_price_files()."
        )
    frames: list[pd.DataFrame] = []
    for path in price_files:
        prices = load_prices(path)
        if prices is None or prices.empty:
            continue
        prices = prices.copy()
        if "yahoo_symbol" not in prices.columns:
            if "symbol" in prices.columns:
                prices["yahoo_symbol"] = (
                    prices["symbol"]
                )
            else:
                continue
        date_column = first_existing_column(
            prices,
            (
                "price_date",
                "date",
                "Date",
            ),
        )
        if date_column is None:
            continue
        if date_column != "price_date":
            prices["price_date"] = prices[
                date_column
            ]
        frames.append(prices)
    if not frames:
        raise RuntimeError(
            "Price files were found, but no usable "
            "price data could be loaded."
        )
    prices = pd.concat(
        frames,
        ignore_index=True,
    )
    prices = deduplicate_columns(
        prices
    )
    prices["yahoo_symbol"] = (
        prices["yahoo_symbol"]
        .astype(str)
        .str.strip()
    )
    prices["price_date"] = pd.to_datetime(
        prices["price_date"],
        errors="coerce",
    )
    prices = prices.dropna(
        subset=[
            "yahoo_symbol",
            "price_date",
        ],
    )
    return prices
def find_close_column(
    prices: pd.DataFrame,
) -> str:
    """Find the project's close-price column."""
    column = first_existing_column(
        prices,
        (
            "adj_close",
            "Adj Close",
            "adjusted_close",
            "close",
            "Close",
        ),
    )
    if column is None:
        raise RuntimeError(
            "Could not find a close-price column "
            "in price data."
        )
    return column
def build_volatility_60d(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate 60-day realized volatility per instrument."""
    close_column = find_close_column(
        prices
    )
    prices = prices.copy()
    prices[close_column] = pd.to_numeric(
        prices[close_column],
        errors="coerce",
    )
    prices = prices.dropna(
        subset=[
            "yahoo_symbol",
            "price_date",
            close_column,
        ],
    )
    prices = prices.sort_values(
        [
            "yahoo_symbol",
            "price_date",
        ],
    )
    prices["daily_return"] = (
        prices
        .groupby("yahoo_symbol")[close_column]
        .pct_change()
    )
    prices["volatility_60d"] = (
        prices
        .groupby("yahoo_symbol")[
            "daily_return"
        ]
        .transform(
            lambda values: (
                values
                .rolling(
                    60,
                    min_periods=20,
                )
                .std()
                * np.sqrt(252)
            )
        )
    )
    return prices[
        [
            "yahoo_symbol",
            "price_date",
            "volatility_60d",
        ]
    ].copy()
# ---------------------------------------------------------------------------
# FI dynamics
# ---------------------------------------------------------------------------
def add_short_interest_dynamics(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add absolute and relative changes in short interest.
    The change is calculated per instrument after sorting chronologically.
    Repeated FI values naturally produce zero change.
    Absolute change:
        current - previous
    Relative change:
        (current - previous) / abs(previous)
    The relative and absolute changes are deliberately kept separate.
    """
    data = data.copy()
    data = data.sort_values(
        [
            "yahoo_symbol",
            "price_date",
        ]
    )
    base = numeric_series(
        data,
        BASE_FI_COLUMN,
    )
    data[BASE_FI_COLUMN] = base
    previous = (
        data
        .groupby("yahoo_symbol")[
            BASE_FI_COLUMN
        ]
        .shift(1)
    )
    data["short_interest_pct_change"] = (
        base - previous
    )
    denominator = previous.abs()
    data[
        "short_interest_pct_change_pct"
    ] = np.where(
        denominator > 0,
        (base - previous) / denominator,
        np.nan,
    )
    data[
        "short_interest_pct_change_pct"
    ] = pd.to_numeric(
        data[
            "short_interest_pct_change_pct"
        ],
        errors="coerce",
    )
    return data
# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def prepare_data() -> pd.DataFrame:
    """
    Load features and attach 60-day volatility
    and short-interest dynamics.
    """
    print(
        "Loading feature data..."
    )
    data = load_features()
    if data is None or data.empty:
        raise RuntimeError(
            "load_features() returned no feature data."
        )
    data = data.copy()
    data = deduplicate_columns(
        data
    )
    print(
        f"Loaded {len(data):,} feature rows."
    )
    required = [
        "yahoo_symbol",
        "forward_return_5d",
        BASE_FI_COLUMN,
    ]
    missing = [
        column
        for column in required
        if column not in data.columns
    ]
    if missing:
        raise RuntimeError(
            "Missing required feature columns: "
            f"{missing}"
        )
    date_column = first_existing_column(
        data,
        (
            "price_date",
            "snapshot_date",
            "date",
        ),
    )
    if date_column is None:
        raise RuntimeError(
            "Could not find a date column "
            "in feature data."
        )
    if date_column != "price_date":
        data["price_date"] = data[
            date_column
        ]
    data["price_date"] = (
        normalize_date_column(
            data,
            "price_date",
        )
    )
    data["yahoo_symbol"] = (
        data["yahoo_symbol"]
        .astype(str)
        .str.strip()
    )
    data["forward_return_5d"] = (
        pd.to_numeric(
            data["forward_return_5d"],
            errors="coerce",
        )
    )
    data[BASE_FI_COLUMN] = (
        numeric_series(
            data,
            BASE_FI_COLUMN,
        )
    )
    data = data.dropna(
        subset=[
            "yahoo_symbol",
            "price_date",
            "forward_return_5d",
        ],
    )
    print(
        f"{BASE_FI_COLUMN} usable values: "
        f"{data[BASE_FI_COLUMN].notna().sum():,}"
    )
    data = add_short_interest_dynamics(
        data
    )
    print(
        "Short-interest dynamics calculated."
    )
    for column in FI_COLUMNS:
        print(
            f"  {column}: "
            f"{data[column].notna().sum():,} "
            f"usable values"
        )
    print(
        "Loading prices for volatility_60d..."
    )
    prices = load_price_data()
    volatility = build_volatility_60d(
        prices
    )
    volatility["price_date"] = (
        normalize_date_column(
            volatility,
            "price_date",
        )
    )
    volatility = volatility.drop_duplicates(
        subset=[
            "yahoo_symbol",
            "price_date",
        ],
        keep="last",
    )
    data = data.merge(
        volatility,
        on=[
            "yahoo_symbol",
            "price_date",
        ],
        how="left",
    )
    data["event"] = (
        data["forward_return_5d"]
        .abs()
        >= EVENT_THRESHOLD
    ).astype(int)
    # Direction:
    #   1 = DOWN
    #   0 = UP
    #
    # Non-events remain NaN.
    data["direction"] = np.where(
        data["forward_return_5d"]
        <= -EVENT_THRESHOLD,
        1,
        np.where(
            data["forward_return_5d"]
            >= EVENT_THRESHOLD,
            0,
            np.nan,
        ),
    )
    return data
# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_logistic_model() -> Pipeline:
    """Create the regularized logistic model."""
    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2_000,
                    C=1.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
def fit_model(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> Pipeline | None:
    """Fit a logistic model after cleaning numeric inputs."""
    if not feature_columns:
        return None
    frame = data[
        feature_columns + [target_column]
    ].copy()
    for column in feature_columns:
        frame[column] = numeric_series(
            frame,
            column,
        )
    frame[target_column] = pd.to_numeric(
        frame[target_column],
        errors="coerce",
    )
    frame = frame.dropna()
    if len(frame) < 30:
        return None
    if frame[target_column].nunique() < 2:
        return None
    model = build_logistic_model()
    model.fit(
        frame[feature_columns],
        frame[target_column],
    )
    return model
def predict_probability(
    model: Pipeline | None,
    data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    """Predict class-1 probability while preserving the index."""
    result = pd.Series(
        np.nan,
        index=data.index,
        dtype=float,
    )
    if model is None or not feature_columns:
        return result
    frame = data[
        feature_columns
    ].copy()
    for column in feature_columns:
        frame[column] = numeric_series(
            frame,
            column,
        )
    valid = frame.notna().all(
        axis=1
    )
    if not valid.any():
        return result
    result.loc[valid] = (
        model.predict_proba(
            frame.loc[
                valid,
                feature_columns,
            ]
        )[:, 1]
    )
    return result
# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------
def available_feature_set(
    data: pd.DataFrame,
    columns: Iterable[str],
) -> list[str]:
    """Return candidate columns that exist and contain data."""
    result: list[str] = []
    for column in columns:
        if column not in data.columns:
            continue
        values = numeric_series(
            data,
            column,
        )
        if values.notna().sum() == 0:
            continue
        result.append(column)
    return result
def select_event_feature_set(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[str, list[str], float]:
    """Select the event model using validation AUC."""
    best_name: str | None = None
    best_columns: list[str] = []
    best_auc = -np.inf
    for name, candidates in (
        EVENT_FEATURE_SETS.items()
    ):
        columns = available_feature_set(
            train,
            candidates,
        )
        if not columns:
            continue
        model = fit_model(
            train,
            columns,
            "event",
        )
        if model is None:
            continue
        probabilities = predict_probability(
            model,
            validation,
            columns,
        )
        valid = probabilities.notna()
        if valid.sum() < 30:
            continue
        y_true = validation.loc[
            valid,
            "event",
        ]
        auc = safe_auc(
            y_true,
            probabilities.loc[valid],
        )
        if np.isnan(auc):
            continue
        print(
            f"    event feature set "
            f"{name}: validation AUC={auc:.6f}"
        )
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_columns = columns
    if best_name is None:
        raise RuntimeError(
            "Could not select an event feature set."
        )
    return (
        best_name,
        best_columns,
        best_auc,
    )
def build_event_score(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[
    str,
    list[str],
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Fit event model using train + validation selection
    and generate scores for train, validation and test.
    """
    (
        name,
        columns,
        validation_auc,
    ) = select_event_feature_set(
        train,
        validation,
    )
    print()
    print(
        f"Selected event feature set: {name}"
    )
    print(
        f"Validation AUC: {validation_auc:.6f}"
    )
    model = fit_model(
        train,
        columns,
        "event",
    )
    train_score = predict_probability(
        model,
        train,
        columns,
    )
    validation_score = predict_probability(
        model,
        validation,
        columns,
    )
    test_score = predict_probability(
        model,
        test,
        columns,
    )
    return (
        name,
        columns,
        train_score,
        validation_score,
        test_score,
    )
# ---------------------------------------------------------------------------
# Event-risk tails
# ---------------------------------------------------------------------------
def select_tail(
    data: pd.DataFrame,
    score: pd.Series,
    fraction: float,
) -> tuple[pd.DataFrame, float]:
    """
    Select the highest-risk fraction.
    The threshold is learned from the supplied training data.
    """
    frame = data.copy()
    frame["_event_score"] = score
    frame = frame.dropna(
        subset=[
            "_event_score",
        ]
    )
    if frame.empty:
        return (
            frame,
            float("nan"),
        )
    threshold = (
        frame["_event_score"]
        .quantile(
            1.0 - fraction
        )
    )
    tail = frame[
        frame["_event_score"]
        >= threshold
    ].copy()
    return (
        tail,
        float(threshold),
    )
# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------
def bootstrap_delta_down_rate(
    low_group: pd.DataFrame,
    high_group: pd.DataFrame,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
]:
    """
    Bootstrap:
        HIGH_DOWN_RATE - LOW_DOWN_RATE
    Resampling is performed independently inside LOW and HIGH groups.
    Returns:
        mean,
        CI low,
        CI high,
        P(delta > 0),
        P(delta <= 0)
    """
    low = (
        pd.to_numeric(
            low_group["direction"],
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )
    high = (
        pd.to_numeric(
            high_group["direction"],
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )
    if len(low) == 0 or len(high) == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )
    observed = float(
        high.mean() - low.mean()
    )
    # If either group contains no outcome variation,
    # the observed difference is exact for that group.
    if (
        np.all(low == low[0])
        and np.all(high == high[0])
    ):
        return (
            observed,
            observed,
            observed,
            1.0 if observed > 0 else 0.0,
            1.0 if observed <= 0 else 0.0,
        )
    rng = np.random.default_rng(
        RANDOM_STATE
    )
    bootstrap_values = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=float,
    )
    for index in range(
        BOOTSTRAP_ITERATIONS
    ):
        low_sample = rng.choice(
            low,
            size=len(low),
            replace=True,
        )
        high_sample = rng.choice(
            high,
            size=len(high),
            replace=True,
        )
        bootstrap_values[index] = (
            high_sample.mean()
            - low_sample.mean()
        )
    return (
        float(
            bootstrap_values.mean()
        ),
        float(
            np.quantile(
                bootstrap_values,
                0.025,
            )
        ),
        float(
            np.quantile(
                bootstrap_values,
                0.975,
            )
        ),
        float(
            np.mean(
                bootstrap_values > 0
            )
        ),
        float(
            np.mean(
                bootstrap_values <= 0
            )
        ),
    )
# ---------------------------------------------------------------------------
# FI conditional level / magnitude test
# ---------------------------------------------------------------------------
def run_low_high_test(
    train_tail: pd.DataFrame,
    test_tail: pd.DataFrame,
    fi_variable: str,
    threshold: float,
    analysis_name: str,
    transform: str = "raw",
) -> DiagnosticResult | None:
    """
    Run a LOW/HIGH test.
    transform:
        raw
            Use the FI value directly.
        abs
            Use absolute FI value.
    The threshold is calculated from training data before calling this
    function and is then applied unchanged to the test period.
    """
    train_values = numeric_series(
        train_tail,
        fi_variable,
    )
    test_values = numeric_series(
        test_tail,
        fi_variable,
    )
    if transform == "abs":
        train_values = train_values.abs()
        test_values = test_values.abs()
    train_valid = train_tail[
        train_values.notna()
        & train_tail["direction"].notna()
    ].copy()
    test_valid = test_tail[
        test_values.notna()
        & test_tail["direction"].notna()
    ].copy()
    if train_valid.empty or test_valid.empty:
        return None
    # Recalculate values after filtering so indices line up exactly.
    train_values = numeric_series(
        train_valid,
        fi_variable,
    )
    test_values = numeric_series(
        test_valid,
        fi_variable,
    )
    if transform == "abs":
        train_values = train_values.abs()
        test_values = test_values.abs()
    low_mask = (
        test_values <= threshold
    )
    high_mask = (
        test_values > threshold
    )
    low = test_valid.loc[
        low_mask
    ].copy()
    high = test_valid.loc[
        high_mask
    ].copy()
    if low.empty or high.empty:
        return None
    low_down_rate = float(
        low["direction"].mean()
    )
    high_down_rate = float(
        high["direction"].mean()
    )
    observed_delta = (
        high_down_rate
        - low_down_rate
    )
    (
        bootstrap_mean,
        ci_low,
        ci_high,
        probability_positive,
        probability_non_positive,
    ) = bootstrap_delta_down_rate(
        low,
        high,
    )
    auc = safe_auc(
        test_valid["direction"],
        test_values,
    )
    return DiagnosticResult(
        window="",
        fi_variable=fi_variable,
        analysis=analysis_name,
        tail=0.0,
        training_tail_events=len(
            train_tail
        ),
        test_tail_events=len(
            test_tail
        ),
        usable_test_events=len(
            test_valid
        ),
        fi_threshold=float(
            threshold
        ),
        low_events=len(low),
        high_events=len(high),
        low_down_events=int(
            low["direction"].sum()
        ),
        low_up_events=int(
            len(low)
            - low["direction"].sum()
        ),
        high_down_events=int(
            high["direction"].sum()
        ),
        high_up_events=int(
            len(high)
            - high["direction"].sum()
        ),
        low_down_rate=low_down_rate,
        high_down_rate=high_down_rate,
        observed_delta_down_rate=observed_delta,
        bootstrap_mean_delta_down_rate=(
            bootstrap_mean
        ),
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        bootstrap_probability_positive=(
            probability_positive
        ),
        bootstrap_probability_non_positive=(
            probability_non_positive
        ),
        auc_fi=auc,
    )
# ---------------------------------------------------------------------------
# Change direction test
# ---------------------------------------------------------------------------
def run_change_direction_test(
    train_tail: pd.DataFrame,
    test_tail: pd.DataFrame,
    fi_variable: str,
) -> ChangeDirectionResult | None:
    """
    Split FI change into:
        DECREASE
        NO_CHANGE
        INCREASE
    Thresholds are not learned from OOS data.
    The primary directional comparison is:
        INCREASE_DOWN_RATE - DECREASE_DOWN_RATE
    NO_CHANGE is retained as a separate descriptive group.
    This directly tests whether recent increases in short interest
    behave differently from decreases, rather than treating every
    increase as a generic "high" value.
    """
    train_values = numeric_series(
        train_tail,
        fi_variable,
    )
    test_values = numeric_series(
        test_tail,
        fi_variable,
    )
    train_valid = train_tail[
        train_values.notna()
        & train_tail["direction"].notna()
    ].copy()
    test_valid = test_tail[
        test_values.notna()
        & test_tail["direction"].notna()
    ].copy()
    if train_valid.empty or test_valid.empty:
        return None
    # We deliberately do not derive a median threshold here.
    # Zero is the economically meaningful boundary between
    # decrease / unchanged / increase.
    test_values = numeric_series(
        test_valid,
        fi_variable,
    )
    decrease = test_valid.loc[
        test_values < 0
    ].copy()
    no_change = test_valid.loc[
        test_values == 0
    ].copy()
    increase = test_valid.loc[
        test_values > 0
    ].copy()
    # Primary comparison requires both directions to exist.
    if decrease.empty or increase.empty:
        return None
    decrease_down_rate = float(
        decrease["direction"].mean()
    )
    increase_down_rate = float(
        increase["direction"].mean()
    )
    no_change_down_rate = (
        float(
            no_change["direction"].mean()
        )
        if not no_change.empty
        else float("nan")
    )
    observed_delta = (
        increase_down_rate
        - decrease_down_rate
    )
    (
        bootstrap_mean,
        ci_low,
        ci_high,
        probability_positive,
        probability_non_positive,
    ) = bootstrap_delta_down_rate(
        decrease,
        increase,
    )
    auc = safe_auc(
        test_valid["direction"],
        test_values,
    )
    return ChangeDirectionResult(
        window="",
        fi_variable=fi_variable,
        tail=0.0,
        training_tail_events=len(
            train_tail
        ),
        test_tail_events=len(
            test_tail
        ),
        decrease_events=len(
            decrease
        ),
        decrease_down_events=int(
            decrease["direction"].sum()
        ),
        decrease_up_events=int(
            len(decrease)
            - decrease["direction"].sum()
        ),
        decrease_down_rate=(
            decrease_down_rate
        ),
        no_change_events=len(
            no_change
        ),
        no_change_down_events=(
            int(
                no_change["direction"].sum()
            )
            if not no_change.empty
            else 0
        ),
        no_change_up_events=(
            int(
                len(no_change)
                - no_change["direction"].sum()
            )
            if not no_change.empty
            else 0
        ),
        no_change_down_rate=(
            no_change_down_rate
        ),
        increase_events=len(
            increase
        ),
        increase_down_events=int(
            increase["direction"].sum()
        ),
        increase_up_events=int(
            len(increase)
            - increase["direction"].sum()
        ),
        increase_down_rate=(
            increase_down_rate
        ),
        observed_increase_minus_decrease=(
            observed_delta
        ),
        bootstrap_mean_delta=(
            bootstrap_mean
        ),
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        bootstrap_probability_positive=(
            probability_positive
        ),
        bootstrap_probability_non_positive=(
            probability_non_positive
        ),
        auc_fi=auc,
    )
# ---------------------------------------------------------------------------
# Change magnitude test
# ---------------------------------------------------------------------------
def run_change_magnitude_test(
    train_tail: pd.DataFrame,
    test_tail: pd.DataFrame,
    fi_variable: str,
) -> DiagnosticResult | None:
    """
    Test whether the magnitude of a non-zero FI change matters.
    The threshold is the median of the absolute non-zero changes
    in the TRAINING tail only.
    OOS observations are classified as:
        LOW magnitude
        HIGH magnitude
    Only non-zero changes participate in this test.
    """
    train_values = numeric_series(
        train_tail,
        fi_variable,
    )
    test_values = numeric_series(
        test_tail,
        fi_variable,
    )
    train_valid = train_tail[
        train_values.notna()
        & train_tail["direction"].notna()
    ].copy()
    test_valid = test_tail[
        test_values.notna()
        & test_tail["direction"].notna()
    ].copy()
    if train_valid.empty or test_valid.empty:
        return None
    train_values = numeric_series(
        train_valid,
        fi_variable,
    )
    test_values = numeric_series(
        test_valid,
        fi_variable,
    )
    train_non_zero = (
        train_values[
            train_values != 0
        ]
        .abs()
        .dropna()
    )
    if train_non_zero.empty:
        return None
    threshold = float(
        train_non_zero.median()
    )
    test_non_zero_mask = (
        test_values != 0
    )
    test_valid = test_valid.loc[
        test_non_zero_mask
    ].copy()
    test_values = numeric_series(
        test_valid,
        fi_variable,
    ).abs()
    if test_valid.empty:
        return None
    low_mask = (
        test_values <= threshold
    )
    high_mask = (
        test_values > threshold
    )
    low = test_valid.loc[
        low_mask
    ].copy()
    high = test_valid.loc[
        high_mask
    ].copy()
    if low.empty or high.empty:
        return None
    low_down_rate = float(
        low["direction"].mean()
    )
    high_down_rate = float(
        high["direction"].mean()
    )
    observed_delta = (
        high_down_rate
        - low_down_rate
    )
    (
        bootstrap_mean,
        ci_low,
        ci_high,
        probability_positive,
        probability_non_positive,
    ) = bootstrap_delta_down_rate(
        low,
        high,
    )
    # AUC is based on the signed original change,
    # because sign itself may contain directional information.
    original_test_values = numeric_series(
        test_valid,
        fi_variable,
    )
    auc = safe_auc(
        test_valid["direction"],
        original_test_values,
    )
    return DiagnosticResult(
        window="",
        fi_variable=fi_variable,
        analysis="change_magnitude",
        tail=0.0,
        training_tail_events=len(
            train_tail
        ),
        test_tail_events=len(
            test_tail
        ),
        usable_test_events=len(
            test_valid
        ),
        fi_threshold=threshold,
        low_events=len(low),
        high_events=len(high),
        low_down_events=int(
            low["direction"].sum()
        ),
        low_up_events=int(
            len(low)
            - low["direction"].sum()
        ),
        high_down_events=int(
            high["direction"].sum()
        ),
        high_up_events=int(
            len(high)
            - high["direction"].sum()
        ),
        low_down_rate=low_down_rate,
        high_down_rate=high_down_rate,
        observed_delta_down_rate=observed_delta,
        bootstrap_mean_delta_down_rate=(
            bootstrap_mean
        ),
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        bootstrap_probability_positive=(
            probability_positive
        ),
        bootstrap_probability_non_positive=(
            probability_non_positive
        ),
        auc_fi=auc,
    )
# ---------------------------------------------------------------------------
# Window handling
# ---------------------------------------------------------------------------
def normalize_window_value(
    value: object,
) -> str:
    """Convert window boundaries to comparable date strings."""
    return pd.Timestamp(value).strftime(
        "%Y-%m-%d"
    )
def split_window(
    data: pd.DataFrame,
    train_end: object,
    validation_end: object,
    test_end: object,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Split data into train, validation and test periods."""
    train_end_string = (
        normalize_window_value(
            train_end
        )
    )
    validation_end_string = (
        normalize_window_value(
            validation_end
        )
    )
    test_end_string = (
        normalize_window_value(
            test_end
        )
    )
    dates = pd.to_datetime(
        data["price_date"],
        errors="coerce",
    )
    train = data[
        dates <= pd.Timestamp(
            train_end_string
        )
    ].copy()
    validation = data[
        (
            dates
            > pd.Timestamp(
                train_end_string
            )
        )
        & (
            dates
            <= pd.Timestamp(
                validation_end_string
            )
        )
    ].copy()
    test = data[
        (
            dates
            > pd.Timestamp(
                validation_end_string
            )
        )
        & (
            dates
            <= pd.Timestamp(
                test_end_string
            )
        )
    ].copy()
    return (
        train,
        validation,
        test,
    )
# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_low_high_result(
    result: DiagnosticResult,
) -> None:
    """Print one LOW/HIGH diagnostic result."""
    print()
    print(
        f"  {result.fi_variable}"
    )
    print(
        f"    Analysis:             "
        f"{result.analysis}"
    )
    print(
        f"    Training tail events: "
        f"{result.training_tail_events:,}"
    )
    print(
        f"    Test tail events:     "
        f"{result.test_tail_events:,}"
    )
    print(
        f"    Usable test events:   "
        f"{result.usable_test_events:,}"
    )
    print(
        f"    Training threshold:   "
        f"{result.fi_threshold:.6f}"
    )
    print()
    print("    LOW:")
    print(
        f"      events: "
        f"{result.low_events:,}"
    )
    print(
        f"      DOWN:   "
        f"{result.low_down_events:,}"
    )
    print(
        f"      UP:     "
        f"{result.low_up_events:,}"
    )
    print(
        f"      DOWN rate: "
        f"{result.low_down_rate:.4f}"
    )
    print()
    print("    HIGH:")
    print(
        f"      events: "
        f"{result.high_events:,}"
    )
    print(
        f"      DOWN:   "
        f"{result.high_down_events:,}"
    )
    print(
        f"      UP:     "
        f"{result.high_up_events:,}"
    )
    print(
        f"      DOWN rate: "
        f"{result.high_down_rate:.4f}"
    )
    print()
    print(
        f"    Observed delta:     "
        f"{result.observed_delta_down_rate:+.4f}"
    )
    print(
        f"    FI-only AUC:        "
        f"{result.auc_fi:.6f}"
    )
    if np.isnan(
        result.bootstrap_mean_delta_down_rate
    ):
        print(
            "    Bootstrap:          "
            "not estimable"
        )
        return
    print(
        f"    Bootstrap mean:     "
        f"{result.bootstrap_mean_delta_down_rate:+.4f}"
    )
    print(
        f"    Bootstrap 95% CI:   "
        f"["
        f"{result.bootstrap_ci_low:+.4f}, "
        f"{result.bootstrap_ci_high:+.4f}"
        f"]"
    )
    print(
        f"    P(delta > 0):       "
        f"{result.bootstrap_probability_positive:.4f}"
    )
    print(
        f"    P(delta <= 0):      "
        f"{result.bootstrap_probability_non_positive:.4f}"
    )
def print_change_direction_result(
    result: ChangeDirectionResult,
) -> None:
    """Print the three-state change-direction result."""
    print()
    print(
        f"  {result.fi_variable}"
    )
    print(
        "    Analysis: change_direction"
    )
    print()
    print("    DECREASE:")
    print(
        f"      events: "
        f"{result.decrease_events:,}"
    )
    print(
        f"      DOWN:   "
        f"{result.decrease_down_events:,}"
    )
    print(
        f"      UP:     "
        f"{result.decrease_up_events:,}"
    )
    print(
        f"      DOWN rate: "
        f"{result.decrease_down_rate:.4f}"
    )
    print()
    print("    NO_CHANGE:")
    print(
        f"      events: "
        f"{result.no_change_events:,}"
    )
    print(
        f"      DOWN:   "
        f"{result.no_change_down_events:,}"
    )
    print(
        f"      UP:     "
        f"{result.no_change_up_events:,}"
    )
    if np.isnan(
        result.no_change_down_rate
    ):
        print(
            "      DOWN rate: n/a"
        )
    else:
        print(
            f"      DOWN rate: "
            f"{result.no_change_down_rate:.4f}"
        )
    print()
    print("    INCREASE:")
    print(
        f"      events: "
        f"{result.increase_events:,}"
    )
    print(
        f"      DOWN:   "
        f"{result.increase_down_events:,}"
    )
    print(
        f"      UP:     "
        f"{result.increase_up_events:,}"
    )
    print(
        f"      DOWN rate: "
        f"{result.increase_down_rate:.4f}"
    )
    print()
    print(
        "    INCREASE - DECREASE:"
    )
    print(
        f"      observed delta: "
        f"{result.observed_increase_minus_decrease:+.4f}"
    )
    print(
        f"      FI-only AUC: "
        f"{result.auc_fi:.6f}"
    )
    print(
        f"      Bootstrap mean: "
        f"{result.bootstrap_mean_delta:+.4f}"
    )
    print(
        f"      Bootstrap 95% CI: "
        f"["
        f"{result.bootstrap_ci_low:+.4f}, "
        f"{result.bootstrap_ci_high:+.4f}"
        f"]"
    )
    print(
        f"      P(delta > 0): "
        f"{result.bootstrap_probability_positive:.4f}"
    )
    print(
        f"      P(delta <= 0): "
        f"{result.bootstrap_probability_non_positive:.4f}"
    )
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 80)
    print(
        "Blankdiss Short-Interest Dynamics "
        "Conditional FI Direction Diagnostic"
    )
    print("=" * 80)
    print()
    print(
        "Question:"
    )
    print(
        "Within an already identified event-risk tail, "
        "does short-interest level or change distinguish "
        "DOWN from UP?"
    )
    print()
    print(
        f"Event threshold: "
        f"{EVENT_THRESHOLD:.0%}"
    )
    print(
        f"Bootstrap iterations: "
        f"{BOOTSTRAP_ITERATIONS:,}"
    )
    print(
        f"Random state: "
        f"{RANDOM_STATE}"
    )
    print()
    print(
        "Change analysis:"
    )
    print(
        "  - DECREASE / NO_CHANGE / INCREASE"
    )
    print(
        "  - non-zero change magnitude split"
    )
    print(
        "  - absolute and relative change evaluated separately"
    )
    print()
    print(
        "FI variables:"
    )
    for column in FI_COLUMNS:
        print(
            f"  - {column}"
        )
    print(
        f"Tails: "
        f"{', '.join(f'{fraction:.0%}' for fraction in TOP_FRACTIONS)}"
    )
    print()
    data = prepare_data()
    print()
    print(
        f"Prepared {len(data):,} rows."
    )
    print(
        f"Date range: "
        f"{data['price_date'].min()} -> "
        f"{data['price_date'].max()}"
    )
    print(
        f"10% absolute-return events: "
        f"{data['event'].sum():,}"
    )
    low_high_results: list[
        DiagnosticResult
    ] = []
    magnitude_results: list[
        DiagnosticResult
    ] = []
    direction_results: list[
        ChangeDirectionResult
    ] = []
    for window in WALK_FORWARD_WINDOWS:
        train_end = window.train_end
        validation_end = (
            window.validation_end
        )
        test_end = window.test_end
        window_label = (
            f"{normalize_window_value(train_end)}"
            f" -> "
            f"{normalize_window_value(validation_end)}"
            f" -> "
            f"{normalize_window_value(test_end)}"
        )
        print()
        print("=" * 80)
        print(
            f"WINDOW {window_label}"
        )
        print("=" * 80)
        train, validation, test = (
            split_window(
                data,
                train_end,
                validation_end,
                test_end,
            )
        )
        print(
            f"Train:      {len(train):,}"
        )
        print(
            f"Validation: {len(validation):,}"
        )
        print(
            f"Test:       {len(test):,}"
        )
        (
            event_feature_name,
            event_feature_columns,
            train_score,
            validation_score,
            test_score,
        ) = build_event_score(
            train,
            validation,
            test,
        )
        _ = (
            event_feature_name,
            event_feature_columns,
            validation_score,
        )
        print()
        for fraction in TOP_FRACTIONS:
            print("-" * 80)
            print(
                f"TAIL {fraction:.0%}"
            )
            print("-" * 80)
            train_events = train[
                train["event"] == 1
            ].copy()
            test_events = test[
                test["event"] == 1
            ].copy()
            train_event_score = (
                train_score.loc[
                    train_events.index
                ]
            )
            test_event_score = (
                test_score.loc[
                    test_events.index
                ]
            )
            train_tail, train_threshold = (
                select_tail(
                    train_events,
                    train_event_score,
                    fraction,
                )
            )
            print(
                f"Training tail events: "
                f"{len(train_tail):,}"
            )
            # IMPORTANT:
            # The OOS threshold comes exclusively from training.
            test_tail = test_events.copy()
            test_tail["_event_score"] = (
                test_event_score
            )
            test_tail = test_tail[
                test_tail["_event_score"]
                >= train_threshold
            ].copy()
            print(
                f"Test tail events:     "
                f"{len(test_tail):,}"
            )
            # ---------------------------------------------------------------
            # Level
            # ---------------------------------------------------------------
            level_train_values = (
                numeric_series(
                    train_tail,
                    BASE_FI_COLUMN,
                )
            )
            level_train_values = (
                level_train_values.dropna()
            )
            if not level_train_values.empty:
                level_threshold = float(
                    level_train_values.median()
                )
                result = run_low_high_test(
                    train_tail=train_tail,
                    test_tail=test_tail,
                    fi_variable=BASE_FI_COLUMN,
                    threshold=level_threshold,
                    analysis_name="level",
                )
                if result is not None:
                    result = DiagnosticResult(
                        window=window_label,
                        fi_variable=(
                            result.fi_variable
                        ),
                        analysis=result.analysis,
                        tail=fraction,
                        training_tail_events=(
                            result.training_tail_events
                        ),
                        test_tail_events=(
                            result.test_tail_events
                        ),
                        usable_test_events=(
                            result.usable_test_events
                        ),
                        fi_threshold=(
                            result.fi_threshold
                        ),
                        low_events=(
                            result.low_events
                        ),
                        high_events=(
                            result.high_events
                        ),
                        low_down_events=(
                            result.low_down_events
                        ),
                        low_up_events=(
                            result.low_up_events
                        ),
                        high_down_events=(
                            result.high_down_events
                        ),
                        high_up_events=(
                            result.high_up_events
                        ),
                        low_down_rate=(
                            result.low_down_rate
                        ),
                        high_down_rate=(
                            result.high_down_rate
                        ),
                        observed_delta_down_rate=(
                            result.observed_delta_down_rate
                        ),
                        bootstrap_mean_delta_down_rate=(
                            result.bootstrap_mean_delta_down_rate
                        ),
                        bootstrap_ci_low=(
                            result.bootstrap_ci_low
                        ),
                        bootstrap_ci_high=(
                            result.bootstrap_ci_high
                        ),
                        bootstrap_probability_positive=(
                            result.bootstrap_probability_positive
                        ),
                        bootstrap_probability_non_positive=(
                            result.bootstrap_probability_non_positive
                        ),
                        auc_fi=result.auc_fi,
                    )
                    print(
                        "\n  SHORT-INTEREST LEVEL"
                    )
                    print_low_high_result(
                        result
                    )
                    low_high_results.append(
                        result
                    )
            # ---------------------------------------------------------------
            # Change variables
            # ---------------------------------------------------------------
            for fi_variable in (
                "short_interest_pct_change",
                "short_interest_pct_change_pct",
            ):
                # -----------------------------------------------------------
                # A. Change direction:
                #    decrease / unchanged / increase
                # -----------------------------------------------------------
                direction_result = (
                    run_change_direction_test(
                        train_tail=train_tail,
                        test_tail=test_tail,
                        fi_variable=fi_variable,
                    )
                )
                if direction_result is not None:
                    direction_result = (
                        ChangeDirectionResult(
                            window=window_label,
                            fi_variable=(
                                direction_result.fi_variable
                            ),
                            tail=fraction,
                            training_tail_events=(
                                direction_result.training_tail_events
                            ),
                            test_tail_events=(
                                direction_result.test_tail_events
                            ),
                            decrease_events=(
                                direction_result.decrease_events
                            ),
                            decrease_down_events=(
                                direction_result.decrease_down_events
                            ),
                            decrease_up_events=(
                                direction_result.decrease_up_events
                            ),
                            decrease_down_rate=(
                                direction_result.decrease_down_rate
                            ),
                            no_change_events=(
                                direction_result.no_change_events
                            ),
                            no_change_down_events=(
                                direction_result.no_change_down_events
                            ),
                            no_change_up_events=(
                                direction_result.no_change_up_events
                            ),
                            no_change_down_rate=(
                                direction_result.no_change_down_rate
                            ),
                            increase_events=(
                                direction_result.increase_events
                            ),
                            increase_down_events=(
                                direction_result.increase_down_events
                            ),
                            increase_up_events=(
                                direction_result.increase_up_events
                            ),
                            increase_down_rate=(
                                direction_result.increase_down_rate
                            ),
                            observed_increase_minus_decrease=(
                                direction_result.observed_increase_minus_decrease
                            ),
                            bootstrap_mean_delta=(
                                direction_result.bootstrap_mean_delta
                            ),
                            bootstrap_ci_low=(
                                direction_result.bootstrap_ci_low
                            ),
                            bootstrap_ci_high=(
                                direction_result.bootstrap_ci_high
                            ),
                            bootstrap_probability_positive=(
                                direction_result.bootstrap_probability_positive
                            ),
                            bootstrap_probability_non_positive=(
                                direction_result.bootstrap_probability_non_positive
                            ),
                            auc_fi=(
                                direction_result.auc_fi
                            ),
                        )
                    )
                    print(
                        f"\n  {fi_variable.upper()} "
                        f"CHANGE DIRECTION"
                    )
                    print_change_direction_result(
                        direction_result
                    )
                    direction_results.append(
                        direction_result
                    )
                else:
                    print()
                    print(
                        f"  {fi_variable}: "
                        "change-direction test "
                        "not estimable"
                    )
                # -----------------------------------------------------------
                # B. Change magnitude
                # -----------------------------------------------------------
                magnitude_result = (
                    run_change_magnitude_test(
                        train_tail=train_tail,
                        test_tail=test_tail,
                        fi_variable=fi_variable,
                    )
                )
                if magnitude_result is not None:
                    magnitude_result = (
                        DiagnosticResult(
                            window=window_label,
                            fi_variable=(
                                magnitude_result.fi_variable
                            ),
                            analysis=(
                                magnitude_result.analysis
                            ),
                            tail=fraction,
                            training_tail_events=(
                                magnitude_result.training_tail_events
                            ),
                            test_tail_events=(
                                magnitude_result.test_tail_events
                            ),
                            usable_test_events=(
                                magnitude_result.usable_test_events
                            ),
                            fi_threshold=(
                                magnitude_result.fi_threshold
                            ),
                            low_events=(
                                magnitude_result.low_events
                            ),
                            high_events=(
                                magnitude_result.high_events
                            ),
                            low_down_events=(
                                magnitude_result.low_down_events
                            ),
                            low_up_events=(
                                magnitude_result.low_up_events
                            ),
                            high_down_events=(
                                magnitude_result.high_down_events
                            ),
                            high_up_events=(
                                magnitude_result.high_up_events
                            ),
                            low_down_rate=(
                                magnitude_result.low_down_rate
                            ),
                            high_down_rate=(
                                magnitude_result.high_down_rate
                            ),
                            observed_delta_down_rate=(
                                magnitude_result.observed_delta_down_rate
                            ),
                            bootstrap_mean_delta_down_rate=(
                                magnitude_result.bootstrap_mean_delta_down_rate
                            ),
                            bootstrap_ci_low=(
                                magnitude_result.bootstrap_ci_low
                            ),
                            bootstrap_ci_high=(
                                magnitude_result.bootstrap_ci_high
                            ),
                            bootstrap_probability_positive=(
                                magnitude_result.bootstrap_probability_positive
                            ),
                            bootstrap_probability_non_positive=(
                                magnitude_result.bootstrap_probability_non_positive
                            ),
                            auc_fi=(
                                magnitude_result.auc_fi
                            ),
                        )
                    )
                    print(
                        f"\n  {fi_variable.upper()} "
                        f"CHANGE MAGNITUDE"
                    )
                    print_low_high_result(
                        magnitude_result
                    )
                    magnitude_results.append(
                        magnitude_result
                    )
                else:
                    print()
                    print(
                        f"  {fi_variable}: "
                        "change-magnitude test "
                        "not estimable"
                    )
    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print(
        "SHORT-INTEREST DYNAMICS SUMMARY"
    )
    print("=" * 80)
    if direction_results:
        print()
        print(
            "CHANGE DIRECTION SUMMARY"
        )
        direction_summary = pd.DataFrame(
            [
                result.__dict__
                for result in direction_results
            ]
        )
        print(
            direction_summary.to_string(
                index=False
            )
        )
    else:
        direction_summary = pd.DataFrame()
    if low_high_results:
        print()
        print(
            "LEVEL SUMMARY"
        )
        level_summary = pd.DataFrame(
            [
                result.__dict__
                for result in low_high_results
            ]
        )
        print(
            level_summary.to_string(
                index=False
            )
        )
    else:
        level_summary = pd.DataFrame()
    if magnitude_results:
        print()
        print(
            "CHANGE MAGNITUDE SUMMARY"
        )
        magnitude_summary = pd.DataFrame(
            [
                result.__dict__
                for result in magnitude_results
            ]
        )
        print(
            magnitude_summary.to_string(
                index=False
            )
        )
    else:
        magnitude_summary = pd.DataFrame()
    # Store all result types in one CSV.
    output_frames: list[
        pd.DataFrame
    ] = []
    if not direction_summary.empty:
        output_frames.append(
            direction_summary.assign(
                result_type="change_direction"
            )
        )
    if not level_summary.empty:
        output_frames.append(
            level_summary.assign(
                result_type="level"
            )
        )
    if not magnitude_summary.empty:
        output_frames.append(
            magnitude_summary.assign(
                result_type="change_magnitude"
            )
        )
    if not output_frames:
        print()
        print(
            "No diagnostic results were produced."
        )
        return
    summary = pd.concat(
        output_frames,
        ignore_index=True,
        sort=False,
    )
    summary.to_csv(
        OUTPUT_FILE,
        index=False,
    )
    print()
    print(
        "Interpretation:"
    )
    print(
        "  LEVEL:"
    )
    print(
        "    positive delta means HIGH short-interest "
        "level had a higher DOWN rate."
    )
    print(
        "  CHANGE DIRECTION:"
    )
    print(
        "    positive INCREASE-DECREASE delta means "
        "increasing short interest had a higher DOWN rate "
        "than decreasing short interest."
    )
    print(
        "    NO_CHANGE is reported separately."
    )
    print(
        "  CHANGE MAGNITUDE:"
    )
    print(
        "    positive delta means larger absolute changes "
        "in short interest had a higher DOWN rate."
    )
    print(
        "    The magnitude threshold is learned only "
        "from non-zero training changes."
    )
    print(
        "  FI-only AUC:"
    )
    print(
        "    > 0.5 means higher FI values are associated "
        "with DOWN within the OOS event tail."
    )
    print(
        "  Bootstrap:"
    )
    print(
        "    all thresholds are learned from training data."
    )
    print(
        "    OOS observations are never used to determine "
        "the threshold."
    )
    print()
    print(
        f"Results saved to: "
        f"{OUTPUT_FILE}"
    )
    print()
    print(
        "Done."
    )
if __name__ == "__main__":
    main()
