"""
Conditional FI short-interest dynamics diagnostic.
Question:
    Within an already identified event-risk tail, does the level or
    recent change in short interest distinguish DOWN from UP?
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
For each FI variable and OOS tail we report:
    - LOW/HIGH FI observation counts
    - DOWN/UP counts
    - DOWN rate in LOW FI
    - DOWN rate in HIGH FI
    - observed difference:
          HIGH_FI_DOWN_RATE - LOW_FI_DOWN_RATE
    - paired-by-group bootstrap confidence interval
    - probability that the difference is > 0
    - continuous FI-only AUC inside the same OOS tail
The FI change variables are calculated per instrument from consecutive
available FI observations.
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
@dataclass(frozen=True)
class DiagnosticResult:
    window: str
    fi_variable: str
    tail: float
    training_tail_events: int
    test_tail_events: int
    fi_threshold: float
    low_fi_events: int
    high_fi_events: int
    low_fi_down_events: int
    low_fi_up_events: int
    high_fi_down_events: int
    high_fi_up_events: int
    low_fi_down_rate: float
    high_fi_down_rate: float
    observed_delta_down_rate: float
    bootstrap_mean_delta_down_rate: float
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
    duplicated = data.columns.duplicated(keep="first")
    if duplicated.any():
        duplicate_names = data.columns[duplicated].tolist()
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
    is_boolean = pd.api.types.is_bool_dtype(values)
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
    price_files = find_price_files(PRICE_DIR)
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
                prices["yahoo_symbol"] = prices["symbol"]
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
            prices["price_date"] = prices[date_column]
        frames.append(prices)
    if not frames:
        raise RuntimeError(
            "Price files were found, but no usable price data "
            "could be loaded."
        )
    prices = pd.concat(
        frames,
        ignore_index=True,
    )
    prices = deduplicate_columns(prices)
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
            "Could not find a close-price column in price data."
        )
    return column
def build_volatility_60d(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate 60-day realized volatility per instrument."""
    close_column = find_close_column(prices)
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
        .groupby("yahoo_symbol")["daily_return"]
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
    Since FI values can remain unchanged across multiple feature snapshots,
    a repeated value naturally produces a zero change. When the FI value
    changes, the change is measured against the previous available
    observation.
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
        .groupby("yahoo_symbol")[BASE_FI_COLUMN]
        .shift(1)
    )
    data["short_interest_pct_change"] = (
        base - previous
    )
    denominator = previous.abs()
    data["short_interest_pct_change_pct"] = np.where(
        denominator > 0,
        (
            (base - previous)
            / denominator
        ),
        np.nan,
    )
    data["short_interest_pct_change_pct"] = (
        pd.to_numeric(
            data["short_interest_pct_change_pct"],
            errors="coerce",
        )
    )
    return data
# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def prepare_data() -> pd.DataFrame:
    """
    Load features and attach 60-day volatility and FI dynamics.
    """
    print("Loading feature data...")
    data = load_features()
    if data is None or data.empty:
        raise RuntimeError(
            "load_features() returned no feature data."
        )
    data = data.copy()
    data = deduplicate_columns(data)
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
            "Could not find a date column in feature data."
        )
    if date_column != "price_date":
        data["price_date"] = data[date_column]
    data["price_date"] = normalize_date_column(
        data,
        "price_date",
    )
    data["yahoo_symbol"] = (
        data["yahoo_symbol"]
        .astype(str)
        .str.strip()
    )
    data["forward_return_5d"] = pd.to_numeric(
        data["forward_return_5d"],
        errors="coerce",
    )
    data[BASE_FI_COLUMN] = numeric_series(
        data,
        BASE_FI_COLUMN,
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
            f"{data[column].notna().sum():,} usable values"
        )
    print(
        "Loading prices for volatility_60d..."
    )
    prices = load_price_data()
    volatility = build_volatility_60d(
        prices
    )
    volatility["price_date"] = normalize_date_column(
        volatility,
        "price_date",
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
        data["forward_return_5d"].abs()
        >= EVENT_THRESHOLD
    ).astype(int)
    # Direction:
    #   1 = DOWN
    #   0 = UP
    #
    # Non-events remain NaN.
    data["direction"] = np.where(
        data["forward_return_5d"] <= -EVENT_THRESHOLD,
        1,
        np.where(
            data["forward_return_5d"] >= EVENT_THRESHOLD,
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
    valid = frame.notna().all(axis=1)
    if not valid.any():
        return result
    result.loc[valid] = model.predict_proba(
        frame.loc[
            valid,
            feature_columns,
        ]
    )[:, 1]
    return result
# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------
def available_feature_set(
    data: pd.DataFrame,
    columns: Iterable[str],
) -> list[str]:
    """Return candidate columns that exist and contain data."""
    result = []
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
    for name, candidates in EVENT_FEATURE_SETS.items():
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
    """Fit event model and generate scores for all periods."""
    name, columns, validation_auc = (
        select_event_feature_set(
            train,
            validation,
        )
    )
    print()
    print(
        f"Selected event feature set: {name}"
    )
    print(
        f"Validation AUC: {validation_auc:.6f}"
    )
    combined_train = train.copy()
    model = fit_model(
        combined_train,
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
        subset=["_event_score"]
    )
    if frame.empty:
        return frame, float("nan")
    threshold = frame["_event_score"].quantile(
        1.0 - fraction
    )
    tail = frame[
        frame["_event_score"] >= threshold
    ].copy()
    return tail, float(threshold)
# ---------------------------------------------------------------------------
# Bootstrap
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
    Bootstrap HIGH_DOWN_RATE - LOW_DOWN_RATE.
    Resampling is performed independently inside LOW and HIGH groups,
    preserving the group structure.
    """
    low = pd.to_numeric(
        low_group["direction"],
        errors="coerce",
    ).dropna().to_numpy(dtype=float)
    high = pd.to_numeric(
        high_group["direction"],
        errors="coerce",
    ).dropna().to_numpy(dtype=float)
    if len(low) == 0 or len(high) == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )
    if np.all(low == low[0]) and np.all(high == high[0]):
        # A bootstrap is still mathematically possible here, but
        # there is no within-group variation. Return the exact
        # observed difference instead of fabricating uncertainty.
        observed = float(
            high.mean() - low.mean()
        )
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
    observed = float(
        high.mean() - low.mean()
    )
    return (
        float(bootstrap_values.mean()),
        float(np.quantile(bootstrap_values, 0.025)),
        float(np.quantile(bootstrap_values, 0.975)),
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
# FI conditional diagnostic
# ---------------------------------------------------------------------------
def run_conditional_test(
    train_tail: pd.DataFrame,
    test_tail: pd.DataFrame,
    fi_variable: str,
) -> tuple[float, DiagnosticResult | None]:
    """Run LOW/HIGH FI conditional test for one tail."""
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
        return (
            float("nan"),
            None,
        )
    train_values = numeric_series(
        train_valid,
        fi_variable,
    )
    test_values = numeric_series(
        test_valid,
        fi_variable,
    )
    threshold = float(
        train_values.median()
    )
    low_mask = test_values <= threshold
    high_mask = test_values > threshold
    low = test_valid.loc[
        low_mask
    ].copy()
    high = test_valid.loc[
        high_mask
    ].copy()
    if low.empty or high.empty:
        return (
            threshold,
            None,
        )
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
    result = DiagnosticResult(
        window="",
        fi_variable=fi_variable,
        tail=0.0,
        training_tail_events=len(train_valid),
        test_tail_events=len(test_valid),
        fi_threshold=threshold,
        low_fi_events=len(low),
        high_fi_events=len(high),
        low_fi_down_events=int(
            low["direction"].sum()
        ),
        low_fi_up_events=int(
            len(low) - low["direction"].sum()
        ),
        high_fi_down_events=int(
            high["direction"].sum()
        ),
        high_fi_up_events=int(
            len(high) - high["direction"].sum()
        ),
        low_fi_down_rate=low_down_rate,
        high_fi_down_rate=high_down_rate,
        observed_delta_down_rate=observed_delta,
        bootstrap_mean_delta_down_rate=bootstrap_mean,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        bootstrap_probability_positive=probability_positive,
        bootstrap_probability_non_positive=probability_non_positive,
        auc_fi=auc,
    )
    return (
        threshold,
        result,
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
    train_end_string = normalize_window_value(
        train_end
    )
    validation_end_string = normalize_window_value(
        validation_end
    )
    test_end_string = normalize_window_value(
        test_end
    )
    dates = pd.to_datetime(
        data["price_date"],
        errors="coerce",
    )
    train = data[
        dates <= pd.Timestamp(train_end_string)
    ].copy()
    validation = data[
        (dates > pd.Timestamp(train_end_string))
        & (
            dates
            <= pd.Timestamp(validation_end_string)
        )
    ].copy()
    test = data[
        (dates > pd.Timestamp(validation_end_string))
        & (
            dates
            <= pd.Timestamp(test_end_string)
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
def print_result(
    result: DiagnosticResult,
) -> None:
    """Print one diagnostic result."""
    print(
        f"  {result.fi_variable}"
    )
    print(
        f"    Training FI events: "
        f"{result.training_tail_events:,}"
    )
    print(
        f"    Test FI events:     "
        f"{result.test_tail_events:,}"
    )
    print(
        f"    Training threshold: "
        f"{result.fi_threshold:.6f}"
    )
    print()
    print("    LOW FI:")
    print(
        f"      events: "
        f"{result.low_fi_events:,}"
    )
    print(
        f"      DOWN:   "
        f"{result.low_fi_down_events:,}"
    )
    print(
        f"      UP:     "
        f"{result.low_fi_up_events:,}"
    )
    print()
    print("    HIGH FI:")
    print(
        f"      events: "
        f"{result.high_fi_events:,}"
    )
    print(
        f"      DOWN:   "
        f"{result.high_fi_down_events:,}"
    )
    print(
        f"      UP:     "
        f"{result.high_fi_up_events:,}"
    )
    print()
    print(
        f"    LOW FI DOWN rate:  "
        f"{result.low_fi_down_rate:.4f}"
    )
    print(
        f"    HIGH FI DOWN rate: "
        f"{result.high_fi_down_rate:.4f}"
    )
    print(
        f"    Observed delta:     "
        f"{result.observed_delta_down_rate:+.4f}"
    )
    print()
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
    print(
        f"    Bootstrap n:        "
        f"{BOOTSTRAP_ITERATIONS:,}"
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
        f"Event threshold: {EVENT_THRESHOLD:.0%}"
    )
    print(
        f"Bootstrap iterations: "
        f"{BOOTSTRAP_ITERATIONS:,}"
    )
    print(
        f"Random state: {RANDOM_STATE}"
    )
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
    results: list[DiagnosticResult] = []
    for window in WALK_FORWARD_WINDOWS:
        train_end = window.train_end
        validation_end = window.validation_end
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
        train, validation, test = split_window(
            data,
            train_end,
            validation_end,
            test_end,
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
        # Silence unused-variable lint concerns while retaining the
        # explicit scores for diagnostics/debugging.
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
            train_event_score = train_score.loc[
                train_events.index
            ]
            test_event_score = test_score.loc[
                test_events.index
            ]
            train_tail, train_threshold = (
                select_tail(
                    train_events,
                    train_event_score,
                    fraction,
                )
            )
            test_tail, _ = select_tail(
                test_events,
                test_event_score,
                fraction,
            )
            # The test threshold MUST come from training.
            # Reapply the training threshold explicitly.
            test_tail = test_events.copy()
            test_tail["_event_score"] = (
                test_event_score
            )
            test_tail = test_tail[
                test_tail["_event_score"]
                >= train_threshold
            ].copy()
            print(
                f"Training tail events: "
                f"{len(train_tail):,}"
            )
            print(
                f"Test tail events:     "
                f"{len(test_tail):,}"
            )
            for fi_variable in FI_COLUMNS:
                (
                    threshold,
                    result,
                ) = run_conditional_test(
                    train_tail,
                    test_tail,
                    fi_variable,
                )
                if result is None:
                    print()
                    print(
                        f"  {fi_variable}: "
                        "insufficient usable data"
                    )
                    continue
                result = DiagnosticResult(
                    window=window_label,
                    fi_variable=fi_variable,
                    tail=fraction,
                    training_tail_events=(
                        result.training_tail_events
                    ),
                    test_tail_events=(
                        result.test_tail_events
                    ),
                    fi_threshold=threshold,
                    low_fi_events=result.low_fi_events,
                    high_fi_events=result.high_fi_events,
                    low_fi_down_events=(
                        result.low_fi_down_events
                    ),
                    low_fi_up_events=(
                        result.low_fi_up_events
                    ),
                    high_fi_down_events=(
                        result.high_fi_down_events
                    ),
                    high_fi_up_events=(
                        result.high_fi_up_events
                    ),
                    low_fi_down_rate=(
                        result.low_fi_down_rate
                    ),
                    high_fi_down_rate=(
                        result.high_fi_down_rate
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
                print_result(result)
                results.append(result)
    print()
    print("=" * 80)
    print(
        "SHORT-INTEREST DYNAMICS SUMMARY"
    )
    print("=" * 80)
    summary = pd.DataFrame(
        [
            result.__dict__
            for result in results
        ]
    )
    if summary.empty:
        print(
            "No diagnostic results were produced."
        )
        return
    print(
        summary.to_string(
            index=False
        )
    )
    output_path = OUTPUT_FILE
    summary.to_csv(
        output_path,
        index=False,
    )
    print()
    print(
        "Interpretation:"
    )
    print(
        "  delta_down_rate > 0 means HIGH FI had a "
        "higher OOS DOWN rate than LOW FI."
    )
    print(
        "  delta_down_rate < 0 means HIGH FI had a "
        "lower OOS DOWN rate than LOW FI."
    )
    print(
        "  For short_interest_pct_change, HIGH FI means "
        "larger recent absolute increases in short interest."
    )
    print(
        "  For short_interest_pct_change_pct, HIGH FI means "
        "larger recent relative increases in short interest."
    )
    print(
        "  A CI entirely above zero supports a positive "
        "conditional FI association."
    )
    print(
        "  A CI entirely below zero supports a negative "
        "conditional FI association."
    )
    print(
        "  FI-only AUC > 0.5 means higher FI values are "
        "associated with DOWN within that OOS tail."
    )
    print(
        "  All thresholds are learned from training data."
    )
    print()
    print(
        f"Results saved to: {output_path}"
    )
    print()
    print("Done.")
if __name__ == "__main__":
    main()
