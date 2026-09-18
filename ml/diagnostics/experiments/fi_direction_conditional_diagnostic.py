"""
Conditional FI direction diagnostic.
Question:
    Within an already identified high-risk event population,
    does short_interest_pct distinguish DOWN events from UP events?
The diagnostic deliberately separates two questions:
1. Event risk:
       Which observations are likely to experience a large
       absolute 5-day move?
2. Event direction:
       Among those high-risk observations, is short_interest_pct
       associated with DOWN versus UP?
The event-risk population is defined by an event-score tail.
The event-score threshold is calculated ONLY from the training
period and then applied unchanged to the OOS test period.
Within each training tail:
    LOW FI  = short_interest_pct <= training median
    HIGH FI = short_interest_pct > training median
The same FI threshold is then applied to the OOS tail.
For each OOS tail we report:
    - LOW/HIGH FI observation counts
    - DOWN/UP counts
    - DOWN rate in LOW FI
    - DOWN rate in HIGH FI
    - observed difference:
          HIGH_FI_DOWN_RATE - LOW_FI_DOWN_RATE
    - paired-by-group bootstrap confidence interval
    - probability that the difference is > 0
    - continuous FI-only AUC inside the same OOS tail
This is a conditional/regime diagnostic, not a causal test.
No repository files are modified by this script.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
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
FI_COLUMN = "short_interest_pct"
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
    frames = []
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
            lambda values: values.rolling(
                60,
                min_periods=20,
            ).std()
            * np.sqrt(252)
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
# Feature preparation
# ---------------------------------------------------------------------------
def prepare_data() -> pd.DataFrame:
    """
    Load features and attach 60-day volatility.
    Required columns:
        yahoo_symbol
        price_date / snapshot_date
        forward_return_5d
        short_interest_pct
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
        FI_COLUMN,
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
    data[FI_COLUMN] = numeric_series(
        data,
        FI_COLUMN,
    )
    data = data.dropna(
        subset=[
            "yahoo_symbol",
            "price_date",
            "forward_return_5d",
        ],
    )
    print(
        f"{FI_COLUMN} usable values: "
        f"{data[FI_COLUMN].notna().sum():,}"
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
    #
    #   1 = DOWN
    #   0 = UP
    #
    # Non-events remain NaN.
    #
    # Do NOT remove these rows here because the event model
    # requires both event and non-event observations.
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
    """
    Select the event model using validation AUC.
    """
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
        if y_true.nunique() < 2:
            continue
        auc = roc_auc_score(
            y_true,
            probabilities.loc[valid],
        )
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
def build_event_scores(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
    str,
    list[str],
]:
    """
    Select event model on validation data and refit on
    train + validation before producing test scores.
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
        "Selected event feature set: "
        f"{name}"
    )
    print(
        f"Validation AUC: "
        f"{validation_auc:.6f}"
    )
    combined = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )
    model = fit_model(
        combined,
        columns,
        "event",
    )
    train_scores = predict_probability(
        model,
        train,
        columns,
    )
    validation_scores = predict_probability(
        model,
        validation,
        columns,
    )
    test_scores = predict_probability(
        model,
        test,
        columns,
    )
    return (
        train_scores,
        validation_scores,
        test_scores,
        name,
        columns,
    )
# ---------------------------------------------------------------------------
# Event-risk tails
# ---------------------------------------------------------------------------
def tail_threshold(
    data: pd.DataFrame,
    score_column: str,
    fraction: float,
) -> float | None:
    """
    Calculate the threshold corresponding to the top fraction.
    This threshold is always calculated on training data.
    """
    values = numeric_series(
        data,
        score_column,
    ).dropna()
    if values.empty:
        return None
    count = max(
        1,
        int(
            np.ceil(
                len(values) * fraction
            )
        ),
    )
    count = min(
        count,
        len(values),
    )
    sorted_values = values.sort_values(
        ascending=False,
    )
    return float(
        sorted_values.iloc[count - 1]
    )
def apply_tail_threshold(
    data: pd.DataFrame,
    score_column: str,
    threshold: float,
) -> pd.DataFrame:
    """Apply a training-derived event-score threshold."""
    frame = data.dropna(
        subset=[
            score_column,
        ],
    ).copy()
    return frame.loc[
        frame[score_column] >= threshold
    ].copy()
# ---------------------------------------------------------------------------
# Conditional FI split
# ---------------------------------------------------------------------------
def fi_training_median(
    train_tail: pd.DataFrame,
) -> float | None:
    """
    Calculate the FI split exclusively from training-tail data.
    """
    values = numeric_series(
        train_tail,
        FI_COLUMN,
    ).dropna()
    if values.empty:
        return None
    return float(
        values.median()
    )
def apply_fi_split(
    data: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into LOW and HIGH FI groups.
    LOW:
        short_interest_pct <= threshold
    HIGH:
        short_interest_pct > threshold
    """
    frame = data.copy()
    frame[FI_COLUMN] = numeric_series(
        frame,
        FI_COLUMN,
    )
    frame = frame.dropna(
        subset=[
            FI_COLUMN,
            "direction",
        ],
    )
    low = frame.loc[
        frame[FI_COLUMN] <= threshold
    ].copy()
    high = frame.loc[
        frame[FI_COLUMN] > threshold
    ].copy()
    return (
        low,
        high,
    )
# ---------------------------------------------------------------------------
# Bootstrap for DOWN-rate difference
# ---------------------------------------------------------------------------
def bootstrap_down_rate_difference(
    low_direction: np.ndarray,
    high_direction: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """
    Bootstrap:
        HIGH_FI_DOWN_RATE - LOW_FI_DOWN_RATE
    LOW and HIGH observations are resampled independently,
    because they are two separate OOS groups.
    Both groups are resampled with replacement while preserving
    their respective group sizes.
    """
    low_direction = np.asarray(
        low_direction,
        dtype=float,
    )
    high_direction = np.asarray(
        high_direction,
        dtype=float,
    )
    low_direction = low_direction[
        np.isfinite(low_direction)
    ]
    high_direction = high_direction[
        np.isfinite(high_direction)
    ]
    if len(low_direction) < 10:
        return np.array([], dtype=float)
    if len(high_direction) < 10:
        return np.array([], dtype=float)
    rng = np.random.default_rng(
        random_state
    )
    deltas = np.empty(
        iterations,
        dtype=float,
    )
    for iteration in range(iterations):
        low_sample = rng.choice(
            low_direction,
            size=len(low_direction),
            replace=True,
        )
        high_sample = rng.choice(
            high_direction,
            size=len(high_direction),
            replace=True,
        )
        low_rate = float(
            np.mean(low_sample)
        )
        high_rate = float(
            np.mean(high_sample)
        )
        deltas[iteration] = (
            high_rate
            - low_rate
        )
    return deltas
def summarize_bootstrap(
    deltas: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
]:
    """Return mean, 95% CI and sign probabilities."""
    if deltas.size == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
    mean = float(
        np.mean(deltas)
    )
    low, high = np.percentile(
        deltas,
        [
            2.5,
            97.5,
        ],
    )
    probability_positive = float(
        np.mean(deltas > 0)
    )
    probability_non_positive = float(
        np.mean(deltas <= 0)
    )
    return (
        mean,
        float(low),
        float(high),
        probability_positive,
        probability_non_positive,
    )
# ---------------------------------------------------------------------------
# FI-only directional AUC
# ---------------------------------------------------------------------------
def calculate_fi_auc(
    data: pd.DataFrame,
) -> float:
    """
    Calculate continuous FI-only AUC within the OOS tail.
    Higher short interest is treated as higher probability of DOWN.
    """
    frame = data[
        [
            FI_COLUMN,
            "direction",
        ]
    ].copy()
    frame[FI_COLUMN] = numeric_series(
        frame,
        FI_COLUMN,
    )
    frame["direction"] = pd.to_numeric(
        frame["direction"],
        errors="coerce",
    )
    frame = frame.dropna()
    if len(frame) < 20:
        return np.nan
    if frame["direction"].nunique() < 2:
        return np.nan
    return float(
        roc_auc_score(
            frame["direction"],
            frame[FI_COLUMN],
        )
    )
# ---------------------------------------------------------------------------
# Single walk-forward window
# ---------------------------------------------------------------------------
def run_window(
    data: pd.DataFrame,
    window,
) -> list[DiagnosticResult]:
    """Run the conditional FI analysis for one walk-forward window."""
    train_end = pd.to_datetime(
        window.train_end
    )
    validation_end = pd.to_datetime(
        window.validation_end
    )
    test_end = pd.to_datetime(
        window.test_end
    )
    dates = pd.to_datetime(
        data["price_date"],
        errors="coerce",
    )
    train = data.loc[
        dates <= train_end
    ].copy()
    validation = data.loc[
        (dates > train_end)
        & (dates <= validation_end)
    ].copy()
    test = data.loc[
        (dates > validation_end)
        & (dates <= test_end)
    ].copy()
    print()
    print(
        "=" * 80
    )
    print(
        "WINDOW "
        f"{window.train_end} -> "
        f"{window.validation_end} -> "
        f"{window.test_end}"
    )
    print(
        "=" * 80
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
    if (
        train.empty
        or validation.empty
        or test.empty
    ):
        print(
            "Skipping window because one period is empty."
        )
        return []
    (
        train_scores,
        validation_scores,
        test_scores,
        event_model_name,
        event_model_columns,
    ) = build_event_scores(
        train,
        validation,
        test,
    )
    train["event_score"] = train_scores
    validation["event_score"] = validation_scores
    test["event_score"] = test_scores
    test_event = test.loc[
        test["event"] == 1
    ].copy()
    test_event = test_event.dropna(
        subset=[
            "event_score",
            "direction",
            FI_COLUMN,
        ],
    )
    if test_event.empty:
        print(
            "No usable OOS events in test period."
        )
        return []
    train_event = train.loc[
        train["event"] == 1
    ].copy()
    train_event = train_event.dropna(
        subset=[
            "event_score",
            "direction",
            FI_COLUMN,
        ],
    )
    if train_event.empty:
        print(
            "No usable training events."
        )
        return []
    print()
    print(
        "FI column: "
        f"{FI_COLUMN}"
    )
    results: list[DiagnosticResult] = []
    window_name = (
        f"{window.train_end}"
        f" -> "
        f"{window.validation_end}"
        f" -> "
        f"{window.test_end}"
    )
    for fraction in TOP_FRACTIONS:
        print()
        print(
            "-" * 80
        )
        print(
            f"TAIL {fraction:.0%}"
        )
        print(
            "-" * 80
        )
        # ---------------------------------------------------------------
        # Event-risk threshold
        # ---------------------------------------------------------------
        threshold = tail_threshold(
            train,
            "event_score",
            fraction,
        )
        if threshold is None:
            print(
                "No training event-score threshold."
            )
            continue
        train_tail = apply_tail_threshold(
            train_event,
            "event_score",
            threshold,
        )
        test_tail = apply_tail_threshold(
            test_event,
            "event_score",
            threshold,
        )
        if train_tail.empty:
            print(
                "No training observations in tail."
            )
            continue
        if test_tail.empty:
            print(
                "No OOS observations in tail."
            )
            continue
        print(
            f"Training tail events: "
            f"{len(train_tail):,}"
        )
        print(
            f"Test tail events:     "
            f"{len(test_tail):,}"
        )
        # ---------------------------------------------------------------
        # FI split learned from training only
        # ---------------------------------------------------------------
        fi_threshold = fi_training_median(
            train_tail
        )
        if fi_threshold is None:
            print(
                "No training FI median available."
            )
            continue
        print(
            f"Training {FI_COLUMN} median: "
            f"{fi_threshold:.6f}"
        )
        low_fi_train, high_fi_train = apply_fi_split(
            train_tail,
            fi_threshold,
        )
        low_fi_test, high_fi_test = apply_fi_split(
            test_tail,
            fi_threshold,
        )
        # ---------------------------------------------------------------
        # Counts
        # ---------------------------------------------------------------
        low_down = int(
            (low_fi_test["direction"] == 1).sum()
        )
        low_up = int(
            (low_fi_test["direction"] == 0).sum()
        )
        high_down = int(
            (high_fi_test["direction"] == 1).sum()
        )
        high_up = int(
            (high_fi_test["direction"] == 0).sum()
        )
        low_count = (
            low_down
            + low_up
        )
        high_count = (
            high_down
            + high_up
        )
        print()
        print(
            "LOW FI:"
        )
        print(
            f"  events: {low_count:,}"
        )
        print(
            f"  DOWN:   {low_down:,}"
        )
        print(
            f"  UP:     {low_up:,}"
        )
        print(
            "HIGH FI:"
        )
        print(
            f"  events: {high_count:,}"
        )
        print(
            f"  DOWN:   {high_down:,}"
        )
        print(
            f"  UP:     {high_up:,}"
        )
        if low_count == 0 or high_count == 0:
            print(
                "Skipping because one FI group is empty."
            )
            continue
        low_down_rate = (
            low_down
            / low_count
        )
        high_down_rate = (
            high_down
            / high_count
        )
        observed_delta = (
            high_down_rate
            - low_down_rate
        )
        # ---------------------------------------------------------------
        # Continuous FI AUC
        # ---------------------------------------------------------------
        auc_fi = calculate_fi_auc(
            test_tail
        )
        # ---------------------------------------------------------------
        # Bootstrap
        # ---------------------------------------------------------------
        deltas = bootstrap_down_rate_difference(
            low_direction=low_fi_test[
                "direction"
            ].to_numpy(),
            high_direction=high_fi_test[
                "direction"
            ].to_numpy(),
            iterations=BOOTSTRAP_ITERATIONS,
            random_state=(
                RANDOM_STATE
                + int(
                    round(
                        fraction * 10_000
                    )
                )
                + pd.Timestamp(
                    validation_end
                ).year
            ),
        )
        (
            bootstrap_mean,
            ci_low,
            ci_high,
            probability_positive,
            probability_non_positive,
        ) = summarize_bootstrap(
            deltas
        )
        print()
        print(
            f"LOW FI DOWN rate:  "
            f"{low_down_rate:.4f}"
        )
        print(
            f"HIGH FI DOWN rate: "
            f"{high_down_rate:.4f}"
        )
        print(
            f"Observed delta:     "
            f"{observed_delta:+.4f}"
        )
        print()
        print(
            f"FI-only AUC:        "
            f"{auc_fi:.6f}"
        )
        print(
            f"Bootstrap mean:     "
            f"{bootstrap_mean:+.4f}"
        )
        print(
            "Bootstrap 95% CI:   "
            f"[{ci_low:+.4f}, {ci_high:+.4f}]"
        )
        print(
            f"P(delta > 0):       "
            f"{probability_positive:.4f}"
        )
        print(
            f"P(delta <= 0):      "
            f"{probability_non_positive:.4f}"
        )
        print(
            f"Bootstrap n:         "
            f"{len(deltas):,}"
        )
        results.append(
            DiagnosticResult(
                window=window_name,
                tail=fraction,
                training_tail_events=len(
                    train_tail
                ),
                test_tail_events=len(
                    test_tail
                ),
                fi_threshold=fi_threshold,
                low_fi_events=low_count,
                high_fi_events=high_count,
                low_fi_down_events=low_down,
                low_fi_up_events=low_up,
                high_fi_down_events=high_down,
                high_fi_up_events=high_up,
                low_fi_down_rate=low_down_rate,
                high_fi_down_rate=high_down_rate,
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
                auc_fi=auc_fi,
            )
        )
    return results
# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def results_to_frame(
    results: list[DiagnosticResult],
) -> pd.DataFrame:
    """Convert results to a DataFrame."""
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "window": result.window,
                "tail": result.tail,
                "training_tail_events": (
                    result.training_tail_events
                ),
                "test_tail_events": (
                    result.test_tail_events
                ),
                "fi_threshold": (
                    result.fi_threshold
                ),
                "low_fi_events": (
                    result.low_fi_events
                ),
                "high_fi_events": (
                    result.high_fi_events
                ),
                "low_fi_down_events": (
                    result.low_fi_down_events
                ),
                "low_fi_up_events": (
                    result.low_fi_up_events
                ),
                "high_fi_down_events": (
                    result.high_fi_down_events
                ),
                "high_fi_up_events": (
                    result.high_fi_up_events
                ),
                "low_fi_down_rate": (
                    result.low_fi_down_rate
                ),
                "high_fi_down_rate": (
                    result.high_fi_down_rate
                ),
                "observed_delta_down_rate": (
                    result.observed_delta_down_rate
                ),
                "bootstrap_mean_delta_down_rate": (
                    result.bootstrap_mean_delta_down_rate
                ),
                "bootstrap_ci_low": (
                    result.bootstrap_ci_low
                ),
                "bootstrap_ci_high": (
                    result.bootstrap_ci_high
                ),
                "bootstrap_probability_positive": (
                    result.bootstrap_probability_positive
                ),
                "bootstrap_probability_non_positive": (
                    result.bootstrap_probability_non_positive
                ),
                "auc_fi": result.auc_fi,
            }
            for result in results
        ]
    )
def print_summary(
    results: list[DiagnosticResult],
) -> None:
    """Print the final summary."""
    frame = results_to_frame(
        results
    )
    print()
    print()
    print(
        "=" * 110
    )
    print(
        "CONDITIONAL FI DIRECTION SUMMARY"
    )
    print(
        "=" * 110
    )
    if frame.empty:
        print(
            "No results."
        )
        return
    display = frame.copy()
    display["tail"] = (
        display["tail"] * 100
    ).round(1)
    numeric_columns = [
        "fi_threshold",
        "low_fi_down_rate",
        "high_fi_down_rate",
        "observed_delta_down_rate",
        "bootstrap_mean_delta_down_rate",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "bootstrap_probability_positive",
        "bootstrap_probability_non_positive",
        "auc_fi",
    ]
    display[numeric_columns] = display[
        numeric_columns
    ].round(4)
    print(
        display.to_string(
            index=False
        )
    )
    print()
    print(
        "Interpretation:"
    )
    print(
        "  delta_down_rate > 0 means HIGH FI had "
        "a higher OOS DOWN rate than LOW FI."
    )
    print(
        "  delta_down_rate < 0 means HIGH FI had "
        "a lower OOS DOWN rate than LOW FI."
    )
    print(
        "  A CI entirely above zero would support "
        "a positive conditional FI association."
    )
    print(
        "  A CI entirely below zero would support "
        "a negative conditional FI association."
    )
    print(
        "  FI-only AUC > 0.5 means higher short interest "
        "was associated with DOWN within that OOS tail."
    )
    print(
        "  All thresholds are learned from training data."
    )
def save_results(
    results: list[DiagnosticResult],
) -> Path | None:
    """Save results beside this diagnostic."""
    frame = results_to_frame(
        results
    )
    if frame.empty:
        return None
    output_path = (
        Path(__file__).resolve().parent
        / "fi_direction_conditional_results.csv"
    )
    frame.to_csv(
        output_path,
        index=False,
    )
    return output_path
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run all configured walk-forward windows."""
    print()
    print(
        "=" * 80
    )
    print(
        "Blankdiss Conditional FI Direction Diagnostic"
    )
    print(
        "=" * 80
    )
    print()
    print(
        "Question:"
    )
    print(
        "Within an already identified event-risk tail, "
        "does short_interest_pct distinguish DOWN from UP?"
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
    print(
        "FI variable: "
        f"{FI_COLUMN}"
    )
    print(
        "Tails: "
        + ", ".join(
            f"{fraction:.0%}"
            for fraction in TOP_FRACTIONS
        )
    )
    data = prepare_data()
    print()
    print(
        f"Prepared {len(data):,} rows."
    )
    date_values = pd.to_datetime(
        data["price_date"],
        errors="coerce",
    )
    print(
        "Date range: "
        f"{date_values.min().date()} -> "
        f"{date_values.max().date()}"
    )
    event_count = int(
        data["event"].sum()
    )
    print(
        f"10% absolute-return events: "
        f"{event_count:,}"
    )
    all_results: list[DiagnosticResult] = []
    for window in WALK_FORWARD_WINDOWS:
        results = run_window(
            data,
            window,
        )
        all_results.extend(
            results
        )
    print_summary(
        all_results
    )
    output_path = save_results(
        all_results
    )
    if output_path is not None:
        print()
        print(
            "Results saved to:"
        )
        print(
            f"  {output_path}"
        )
    print()
    print(
        "Done."
    )
if __name__ == "__main__":
    main()
