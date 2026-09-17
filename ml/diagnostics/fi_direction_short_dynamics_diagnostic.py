"""
Conditional FI short-interest continuous-change diagnostic.
Question:
    Within an already identified 5% event-risk tail, does the
    continuous change in short interest distinguish DOWN from UP?
The diagnostic focuses on the strongest result from the previous
conditional analysis: the 5% event-risk tail.
For each walk-forward window:
1. Select the event-risk model using validation AUC.
2. Define the 5% event-score tail using TRAINING data only.
3. Restrict OOS observations to that tail.
4. Evaluate short-interest change as a continuous variable:
       - absolute change
       - relative change
5. Test:
       - continuous FI-only AUC
       - Spearman rank correlation with DOWN
       - logistic regression using continuous change
       - training-defined quantile bins
       - increase/decrease comparison
       - positive-change magnitude
The quantile boundaries are learned from TRAINING data only.
The purpose is to determine whether the previous result was:
    A. a genuinely monotonic relationship,
    B. mainly an increase-vs-decrease effect,
    C. driven by a few extreme changes,
    D. or just unstable small-sample noise.
No repository files are modified by this script.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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
# Main hypothesis from the previous diagnostic.
EVENT_TAIL = 0.05
BASE_FI_COLUMN = "short_interest_pct"
CHANGE_COLUMNS = (
    "short_interest_pct_change",
    "short_interest_pct_change_pct",
)
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
# Helpers
# ---------------------------------------------------------------------------
def first_existing_column(
    data: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    for column in candidates:
        if column in data.columns:
            return column
    return None
def normalize_date_column(
    data: pd.DataFrame,
    column: str,
) -> pd.Series:
    return pd.to_datetime(
        data[column],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
def deduplicate_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    duplicated = data.columns.duplicated(
        keep="first"
    )
    if duplicated.any():
        names = data.columns[duplicated].tolist()
        print(
            f"Duplicate columns removed: {names}"
        )
        data = data.loc[
            :,
            ~duplicated,
        ].copy()
    return data
def numeric_series(
    data: pd.DataFrame,
    column: str,
) -> pd.Series:
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
def safe_spearman(
    x: pd.Series,
    y: pd.Series,
) -> tuple[float, float]:
    frame = pd.DataFrame(
        {
            "x": x,
            "y": y,
        }
    ).dropna()
    if len(frame) < 3:
        return float("nan"), float("nan")
    if frame["x"].nunique() < 2:
        return float("nan"), float("nan")
    if frame["y"].nunique() < 2:
        return float("nan"), float("nan")
    correlation, p_value = spearmanr(
        frame["x"],
        frame["y"],
    )
    return float(correlation), float(p_value)
def bootstrap_mean_difference(
    first: pd.Series,
    second: pd.Series,
    iterations: int = BOOTSTRAP_ITERATIONS,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float, float, float]:
    """
    Bootstrap mean(first) - mean(second).
    Returns:
        mean bootstrap delta,
        CI low,
        CI high,
        P(delta > 0)
    """
    first = pd.to_numeric(
        first,
        errors="coerce",
    ).dropna().to_numpy()
    second = pd.to_numeric(
        second,
        errors="coerce",
    ).dropna().to_numpy()
    if len(first) == 0 or len(second) == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )
    rng = np.random.default_rng(
        random_state
    )
    deltas = np.empty(
        iterations,
        dtype=float,
    )
    for index in range(iterations):
        first_sample = rng.choice(
            first,
            size=len(first),
            replace=True,
        )
        second_sample = rng.choice(
            second,
            size=len(second),
            replace=True,
        )
        deltas[index] = (
            first_sample.mean()
            - second_sample.mean()
        )
    return (
        float(deltas.mean()),
        float(np.quantile(deltas, 0.025)),
        float(np.quantile(deltas, 0.975)),
        float((deltas > 0).mean()),
    )
# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
def load_price_data() -> pd.DataFrame:
    price_files = find_price_files(
        PRICE_DIR
    )
    if not price_files:
        raise RuntimeError(
            "No price files found."
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
            "No usable price data."
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
        ]
    )
    return prices
def find_close_column(
    prices: pd.DataFrame,
) -> str:
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
            "Could not find close-price column."
        )
    return column
def build_volatility_60d(
    prices: pd.DataFrame,
) -> pd.DataFrame:
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
        ]
    )
    prices = prices.sort_values(
        [
            "yahoo_symbol",
            "price_date",
        ]
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
# Short-interest dynamics
# ---------------------------------------------------------------------------
def add_short_interest_dynamics(
    data: pd.DataFrame,
) -> pd.DataFrame:
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
    data[
        "short_interest_pct_change"
    ] = base - previous
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
# Preparation
# ---------------------------------------------------------------------------
def prepare_data() -> pd.DataFrame:
    data = load_features()
    if data is None or data.empty:
        raise RuntimeError(
            "load_features() returned no data."
        )
    data = deduplicate_columns(
        data.copy()
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
            f"Missing required columns: {missing}"
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
            "No date column found."
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
        ]
    )
    data = add_short_interest_dynamics(
        data
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
# Event model
# ---------------------------------------------------------------------------
def build_event_model() -> Pipeline:
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
def fit_event_model(
    train: pd.DataFrame,
    features: list[str],
) -> Pipeline | None:
    frame = train[
        features + ["event"]
    ].copy()
    for column in features:
        frame[column] = numeric_series(
            frame,
            column,
        )
    frame["event"] = pd.to_numeric(
        frame["event"],
        errors="coerce",
    )
    frame = frame.dropna()
    if len(frame) < 100:
        return None
    if frame["event"].nunique() < 2:
        return None
    model = build_event_model()
    model.fit(
        frame[features],
        frame["event"],
    )
    return model
def predict_event_score(
    model: Pipeline | None,
    data: pd.DataFrame,
    features: list[str],
) -> pd.Series:
    result = pd.Series(
        np.nan,
        index=data.index,
        dtype=float,
    )
    if model is None:
        return result
    frame = data[
        features
    ].copy()
    for column in features:
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
                features,
            ]
        )[:, 1]
    )
    return result
def select_event_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[str, list[str], float]:
    best_name = ""
    best_features: list[str] = []
    best_auc = -np.inf
    for name, candidates in (
        EVENT_FEATURE_SETS.items()
    ):
        features = [
            column
            for column in candidates
            if column in train.columns
            and column in validation.columns
        ]
        if not features:
            continue
        model = fit_event_model(
            train,
            features,
        )
        if model is None:
            continue
        validation_score = predict_event_score(
            model,
            validation,
            features,
        )
        auc = safe_auc(
            validation["event"],
            validation_score,
        )
        if (
            not np.isnan(auc)
            and auc > best_auc
        ):
            best_name = name
            best_features = features
            best_auc = auc
    if not best_features:
        raise RuntimeError(
            "Could not select event model."
        )
    return (
        best_name,
        best_features,
        float(best_auc),
    )
# ---------------------------------------------------------------------------
# Tail
# ---------------------------------------------------------------------------
def training_tail_threshold(
    train_scores: pd.Series,
    fraction: float,
) -> float:
    scores = pd.to_numeric(
        train_scores,
        errors="coerce",
    ).dropna()
    if len(scores) == 0:
        return float("nan")
    return float(
        scores.quantile(
            1.0 - fraction
        )
    )
# ---------------------------------------------------------------------------
# Continuous analysis
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContinuousResult:
    window: str
    variable: str
    training_tail_events: int
    test_tail_events: int
    usable_events: int
    positive_events: int
    negative_events: int
    zero_events: int
    auc: float
    spearman_rho: float
    spearman_p: float
    logistic_auc: float
def continuous_analysis(
    test_tail: pd.DataFrame,
    variable: str,
    window: str,
) -> ContinuousResult:
    frame = test_tail[
        [
            variable,
            "direction",
        ]
    ].copy()
    frame[variable] = numeric_series(
        frame,
        variable,
    )
    frame["direction"] = pd.to_numeric(
        frame["direction"],
        errors="coerce",
    )
    frame = frame.dropna()
    values = frame[variable]
    positive = values > 0
    negative = values < 0
    zero = values == 0
    auc = safe_auc(
        frame["direction"],
        values,
    )
    rho, p_value = safe_spearman(
        values,
        frame["direction"],
    )
    logistic_auc = float("nan")
    if (
        len(frame) >= 10
        and frame["direction"].nunique() >= 2
        and values.nunique() >= 2
    ):
        model = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        x = values.to_numpy().reshape(
            -1,
            1,
        )
        y = frame[
            "direction"
        ].to_numpy()
        model.fit(
            x,
            y,
        )
        scores = model.predict_proba(
            x
        )[:, 1]
        logistic_auc = safe_auc(
            frame["direction"],
            pd.Series(
                scores,
                index=frame.index,
            ),
        )
    return ContinuousResult(
        window=window,
        variable=variable,
        training_tail_events=0,
        test_tail_events=len(test_tail),
        usable_events=len(frame),
        positive_events=int(
            positive.sum()
        ),
        negative_events=int(
            negative.sum()
        ),
        zero_events=int(
            zero.sum()
        ),
        auc=auc,
        spearman_rho=rho,
        spearman_p=p_value,
        logistic_auc=logistic_auc,
    )
# ---------------------------------------------------------------------------
# Quantile analysis
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuantileResult:
    window: str
    variable: str
    bin_name: str
    test_events: int
    down_events: int
    up_events: int
    down_rate: float
QUANTILES = (
    0.00,
    0.20,
    0.40,
    0.60,
    0.80,
    1.00,
)
def training_quantile_edges(
    train_tail: pd.DataFrame,
    variable: str,
) -> np.ndarray:
    values = numeric_series(
        train_tail,
        variable,
    ).dropna()
    if len(values) < 10:
        return np.array([])
    edges = np.quantile(
        values,
        QUANTILES,
    )
    edges = np.unique(
        edges
    )
    if len(edges) < 2:
        return np.array([])
    return edges
def assign_quantile_bins(
    values: pd.Series,
    edges: np.ndarray,
) -> pd.Series:
    if len(edges) < 2:
        return pd.Series(
            pd.NA,
            index=values.index,
            dtype="string",
        )
    result = pd.cut(
        values,
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    )
    return result.astype("string")
def quantile_analysis(
    train_tail: pd.DataFrame,
    test_tail: pd.DataFrame,
    variable: str,
    window: str,
) -> tuple[
    list[QuantileResult],
    np.ndarray,
]:
    edges = training_quantile_edges(
        train_tail,
        variable,
    )
    if len(edges) < 2:
        return [], edges
    frame = test_tail[
        [
            variable,
            "direction",
        ]
    ].copy()
    frame[variable] = numeric_series(
        frame,
        variable,
    )
    frame["direction"] = pd.to_numeric(
        frame["direction"],
        errors="coerce",
    )
    frame["bin"] = assign_quantile_bins(
        frame[variable],
        edges,
    )
    frame = frame.dropna(
        subset=[
            variable,
            "direction",
            "bin",
        ]
    )
    results: list[QuantileResult] = []
    for bin_name, group in frame.groupby(
        "bin",
        observed=True,
    ):
        down = int(
            (group["direction"] == 1)
            .sum()
        )
        up = int(
            (group["direction"] == 0)
            .sum()
        )
        total = down + up
        results.append(
            QuantileResult(
                window=window,
                variable=variable,
                bin_name=str(bin_name),
                test_events=total,
                down_events=down,
                up_events=up,
                down_rate=(
                    down / total
                    if total
                    else float("nan")
                ),
            )
        )
    return results, edges
# ---------------------------------------------------------------------------
# Positive-change analysis
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PositiveChangeResult:
    window: str
    variable: str
    threshold: float
    low_positive_events: int
    low_positive_down: int
    low_positive_down_rate: float
    high_positive_events: int
    high_positive_down: int
    high_positive_down_rate: float
    delta_high_minus_low: float
    bootstrap_mean: float
    ci_low: float
    ci_high: float
    probability_positive: float
def positive_change_analysis(
    train_tail: pd.DataFrame,
    test_tail: pd.DataFrame,
    variable: str,
    window: str,
) -> PositiveChangeResult:
    train_values = numeric_series(
        train_tail,
        variable,
    )
    positive_train = train_values[
        train_values > 0
    ].dropna()
    if len(positive_train) == 0:
        threshold = float("nan")
    else:
        threshold = float(
            positive_train.quantile(
                0.50
            )
        )
    frame = test_tail[
        [
            variable,
            "direction",
        ]
    ].copy()
    frame[variable] = numeric_series(
        frame,
        variable,
    )
    frame["direction"] = pd.to_numeric(
        frame["direction"],
        errors="coerce",
    )
    frame = frame.dropna()
    positive = frame[
        frame[variable] > 0
    ].copy()
    if positive.empty or np.isnan(
        threshold
    ):
        return PositiveChangeResult(
            window=window,
            variable=variable,
            threshold=threshold,
            low_positive_events=0,
            low_positive_down=0,
            low_positive_down_rate=float("nan"),
            high_positive_events=0,
            high_positive_down=0,
            high_positive_down_rate=float("nan"),
            delta_high_minus_low=float("nan"),
            bootstrap_mean=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            probability_positive=float("nan"),
        )
    low = positive[
        positive[variable] <= threshold
    ]
    high = positive[
        positive[variable] > threshold
    ]
    low_down = int(
        (low["direction"] == 1).sum()
    )
    high_down = int(
        (high["direction"] == 1).sum()
    )
    low_rate = (
        low_down / len(low)
        if len(low)
        else float("nan")
    )
    high_rate = (
        high_down / len(high)
        if len(high)
        else float("nan")
    )
    if (
        len(low) > 0
        and len(high) > 0
    ):
        (
            bootstrap_mean,
            ci_low,
            ci_high,
            probability_positive,
        ) = bootstrap_mean_difference(
            high["direction"],
            low["direction"],
        )
        delta = (
            high_rate - low_rate
        )
    else:
        bootstrap_mean = float("nan")
        ci_low = float("nan")
        ci_high = float("nan")
        probability_positive = float("nan")
        delta = float("nan")
    return PositiveChangeResult(
        window=window,
        variable=variable,
        threshold=threshold,
        low_positive_events=len(low),
        low_positive_down=low_down,
        low_positive_down_rate=low_rate,
        high_positive_events=len(high),
        high_positive_down=high_down,
        high_positive_down_rate=high_rate,
        delta_high_minus_low=delta,
        bootstrap_mean=bootstrap_mean,
        ci_low=ci_low,
        ci_high=ci_high,
        probability_positive=probability_positive,
    )
# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_continuous_result(
    result: ContinuousResult,
) -> None:
    print(
        f"  {result.variable}: "
        f"n={result.usable_events}, "
        f"+={result.positive_events}, "
        f"-={result.negative_events}, "
        f"0={result.zero_events}, "
        f"AUC={result.auc:.4f}, "
        f"Spearman={result.spearman_rho:+.4f} "
        f"(p={result.spearman_p:.4f}), "
        f"logistic AUC={result.logistic_auc:.4f}"
    )
def print_quantile_results(
    results: list[QuantileResult],
) -> None:
    for result in results:
        print(
            f"    {result.bin_name}: "
            f"n={result.test_events}, "
            f"DOWN={result.down_events}, "
            f"UP={result.up_events}, "
            f"DOWN rate={result.down_rate:.4f}"
        )
def print_positive_result(
    result: PositiveChangeResult,
) -> None:
    print(
        f"  positive changes: "
        f"threshold={result.threshold:.6f}, "
        f"LOW n={result.low_positive_events} "
        f"DOWN={result.low_positive_down} "
        f"rate={result.low_positive_down_rate:.4f}, "
        f"HIGH n={result.high_positive_events} "
        f"DOWN={result.high_positive_down} "
        f"rate={result.high_positive_down_rate:.4f}, "
        f"delta={result.delta_high_minus_low:+.4f}, "
        f"CI=[{result.ci_low:+.4f}, "
        f"{result.ci_high:+.4f}]"
    )
# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def run_window(
    data: pd.DataFrame,
    window,
) -> tuple[
    list[ContinuousResult],
    list[QuantileResult],
    list[PositiveChangeResult],
]:
    train = data[
        data["price_date"]
        <= window.train_end
    ].copy()
    validation = data[
        (
            data["price_date"]
            > window.train_end
        )
        & (
            data["price_date"]
            <= window.validation_end
        )
    ].copy()
    test = data[
        (
            data["price_date"]
            > window.validation_end
        )
        & (
            data["price_date"]
            <= window.test_end
        )
    ].copy()
    (
        feature_name,
        event_features,
        validation_auc,
    ) = select_event_model(
        train,
        validation,
    )
    event_model = fit_event_model(
        train,
        event_features,
    )
    train["event_score"] = predict_event_score(
        event_model,
        train,
        event_features,
    )
    validation["event_score"] = predict_event_score(
        event_model,
        validation,
        event_features,
    )
    test["event_score"] = predict_event_score(
        event_model,
        test,
        event_features,
    )
    threshold = training_tail_threshold(
        train["event_score"],
        EVENT_TAIL,
    )
    train_tail = train[
        train["event_score"]
        >= threshold
    ].copy()
    test_tail = test[
        test["event_score"]
        >= threshold
    ].copy()
    window_name = (
        f"{window.train_end} -> "
        f"{window.validation_end} -> "
        f"{window.test_end}"
    )
    print()
    print(
        f"WINDOW {window_name}"
    )
    print(
        f"event={feature_name}, "
        f"validation AUC={validation_auc:.4f}, "
        f"tail threshold={threshold:.6f}, "
        f"train tail={len(train_tail)}, "
        f"test tail={len(test_tail)}"
    )
    continuous_results: list[
        ContinuousResult
    ] = []
    quantile_results: list[
        QuantileResult
    ] = []
    positive_results: list[
        PositiveChangeResult
    ] = []
    for variable in CHANGE_COLUMNS:
        result = continuous_analysis(
            test_tail,
            variable,
            window_name,
        )
        result = ContinuousResult(
            window=result.window,
            variable=result.variable,
            training_tail_events=len(
                train_tail
            ),
            test_tail_events=result.test_tail_events,
            usable_events=result.usable_events,
            positive_events=result.positive_events,
            negative_events=result.negative_events,
            zero_events=result.zero_events,
            auc=result.auc,
            spearman_rho=result.spearman_rho,
            spearman_p=result.spearman_p,
            logistic_auc=result.logistic_auc,
        )
        continuous_results.append(
            result
        )
        print_continuous_result(
            result
        )
        bins, _ = quantile_analysis(
            train_tail,
            test_tail,
            variable,
            window_name,
        )
        quantile_results.extend(
            bins
        )
        print("  training-defined quintiles:")
        print_quantile_results(
            bins
        )
        positive_result = (
            positive_change_analysis(
                train_tail,
                test_tail,
                variable,
                window_name,
            )
        )
        positive_results.append(
            positive_result
        )
        print_positive_result(
            positive_result
        )
    return (
        continuous_results,
        quantile_results,
        positive_results,
    )
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(
        "Blankdiss Short-Interest Continuous "
        "Change Diagnostic"
    )
    print(
        f"Event tail: {EVENT_TAIL:.0%} | "
        f"Event threshold: {EVENT_THRESHOLD:.0%} | "
        f"Bootstrap: {BOOTSTRAP_ITERATIONS:,}"
    )
    data = prepare_data()
    print(
        f"Prepared {len(data):,} rows | "
        f"events={int(data['event'].sum()):,}"
    )
    all_continuous: list[
        ContinuousResult
    ] = []
    all_quantiles: list[
        QuantileResult
    ] = []
    all_positive: list[
        PositiveChangeResult
    ] = []
    for window in WALK_FORWARD_WINDOWS:
        (
            continuous,
            quantiles,
            positive,
        ) = run_window(
            data,
            window,
        )
        all_continuous.extend(
            continuous
        )
        all_quantiles.extend(
            quantiles
        )
        all_positive.extend(
            positive
        )
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for variable in CHANGE_COLUMNS:
        print()
        print(variable)
        matching = [
            result
            for result in all_continuous
            if result.variable == variable
        ]
        for result in matching:
            print(
                f"  {result.window}: "
                f"AUC={result.auc:.4f}, "
                f"Spearman={result.spearman_rho:+.4f}, "
                f"p={result.spearman_p:.4f}, "
                f"n={result.usable_events}"
            )
        positive_matching = [
            result
            for result in all_positive
            if result.variable == variable
        ]
        for result in positive_matching:
            print(
                f"  {result.window}: "
                f"positive HIGH-LOW="
                f"{result.delta_high_minus_low:+.4f}, "
                f"CI=["
                f"{result.ci_low:+.4f}, "
                f"{result.ci_high:+.4f}"
                f"]"
            )
    # ------------------------------------------------------------------
    # Pooled 5% tail analysis.
    #
    # Important:
    # The pooled result is descriptive only. Thresholds remain
    # window-specific and are learned from each training period.
    # ------------------------------------------------------------------
    print()
    print("POOLED WALK-FORWARD RESULT")
    print(
        "The pooled result combines OOS observations from both "
        "walk-forward windows after each window's own training-only "
        "tail threshold was applied."
    )
    pooled_frames: list[pd.DataFrame] = []
    for window in WALK_FORWARD_WINDOWS:
        train = data[
            data["price_date"]
            <= window.train_end
        ].copy()
        validation = data[
            (
                data["price_date"]
                > window.train_end
            )
            & (
                data["price_date"]
                <= window.validation_end
            )
        ].copy()
        test = data[
            (
                data["price_date"]
                > window.validation_end
            )
            & (
                data["price_date"]
                <= window.test_end
            )
        ].copy()
        (
            _,
            features,
            _,
        ) = select_event_model(
            train,
            validation,
        )
        model = fit_event_model(
            train,
            features,
        )
        train_score = predict_event_score(
            model,
            train,
            features,
        )
        test_score = predict_event_score(
            model,
            test,
            features,
        )
        threshold = training_tail_threshold(
            train_score,
            EVENT_TAIL,
        )
        tail = test[
            test_score >= threshold
        ].copy()
        tail["window"] = (
            f"{window.train_end} -> "
            f"{window.validation_end} -> "
            f"{window.test_end}"
        )
        pooled_frames.append(
            tail
        )
    pooled = pd.concat(
        pooled_frames,
        ignore_index=True,
    )
    for variable in CHANGE_COLUMNS:
        frame = pooled[
            [
                variable,
                "direction",
            ]
        ].copy()
        frame[variable] = numeric_series(
            frame,
            variable,
        )
        frame["direction"] = pd.to_numeric(
            frame["direction"],
            errors="coerce",
        )
        frame = frame.dropna()
        print()
        print(
            f"{variable}: "
            f"pooled n={len(frame)}"
        )
        continuous_auc = safe_auc(
            frame["direction"],
            frame[variable],
        )
        rho, p_value = safe_spearman(
            frame[variable],
            frame["direction"],
        )
        print(
            f"  continuous AUC={continuous_auc:.4f}, "
            f"Spearman={rho:+.4f}, "
            f"p={p_value:.4f}"
        )
        positive = frame[
            frame[variable] > 0
        ]
        negative = frame[
            frame[variable] < 0
        ]
        zero = frame[
            frame[variable] == 0
        ]
        print(
            f"  direction: "
            f"DECREASE n={len(negative)} "
            f"DOWN={int((negative['direction'] == 1).sum())}, "
            f"NO_CHANGE n={len(zero)}, "
            f"DOWN={int((zero['direction'] == 1).sum())}, "
            f"INCREASE n={len(positive)} "
            f"DOWN={int((positive['direction'] == 1).sum())}"
        )
if __name__ == "__main__":
    main()
