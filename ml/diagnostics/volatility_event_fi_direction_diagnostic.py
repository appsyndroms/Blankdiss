"""
Paired bootstrap diagnostic for incremental FI information.
Question:
    Does FI add statistically credible directional information
    on top of an already identified volatility/event-risk score?
The diagnostic compares:
    baseline:
        direction model using event_score only
    treatment:
        direction model using event_score + FI
The comparison is made on the exact same OOS observations:
    delta_auc = auc(event_score + FI) - auc(event_score)
A paired, stratified bootstrap is then used to estimate:
    - mean delta AUC
    - bootstrap 95% confidence interval
    - probability that delta AUC <= 0
    - probability that delta AUC > 0
The diagnostic is deliberately separate from the existing
volatility_event_fi_direction_diagnostic.py.
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
FI_COLUMNS = (
    "short_interest_pct",
    "short_interest",
    "short_interest_change",
    "short_interest_delta",
    "short_interest_rank",
    "short_interest_zscore",
    "short_interest_acceleration",
    "short_interest_days",
    "short_interest_ratio",
    "short_interest_change_5d",
    "short_interest_change_20d",
    "short_interest_change_60d",
    "short_interest_trend",
    "short_interest_volatility",
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
@dataclass(frozen=True)
class DiagnosticResult:
    window: str
    tail: float
    test_events: int
    down_events: int
    up_events: int
    auc_event_score: float
    auc_event_fi: float
    delta_auc: float
    bootstrap_mean_delta_auc: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    bootstrap_probability_positive: float
    bootstrap_probability_non_positive: float
# ---------------------------------------------------------------------------
# Generic helpers
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
        duplicate_names = (
            data.columns[duplicated]
            .tolist()
        )
        print()
        print(
            "Duplicate feature columns detected; "
            "keeping first occurrence:"
        )
        print(
            f"  {duplicate_names}"
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
    if isinstance(
        values,
        pd.DataFrame,
    ):
        values = values.iloc[:, 0]
    is_boolean = (
        pd.api.types.is_bool_dtype(
            values
        )
    )
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
    """
    Load all raw price files from the project's configured price directory.
    """
    price_files = find_price_files(
        PRICE_DIR
    )
    if not price_files:
        raise RuntimeError(
            "No price files found by "
            "analysis.feature_prices.find_price_files()."
        )
    frames = []
    for path in price_files:
        prices = load_prices(
            path
        )
        if prices is None or prices.empty:
            continue
        prices = prices.copy()
        if "yahoo_symbol" not in prices.columns:
            continue
        if "date" not in prices.columns:
            continue
        if "close" not in prices.columns:
            continue
        frames.append(
            prices
        )
    if not frames:
        raise RuntimeError(
            "Price files were found, but no usable price data "
            "could be loaded."
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
        ],
    )
    prices = prices.loc[
        prices["close"] > 0
    ].copy()
    prices = (
        prices.sort_values(
            [
                "yahoo_symbol",
                "date",
            ],
            kind="mergesort",
        )
        .drop_duplicates(
            subset=[
                "yahoo_symbol",
                "date",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )
    return prices
def build_volatility_60d(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate 60-day realized volatility per Yahoo symbol.
    """
    prices = prices.copy()
    prices["close"] = pd.to_numeric(
        prices["close"],
        errors="coerce",
    )
    prices = prices.dropna(
        subset=[
            "yahoo_symbol",
            "date",
            "close",
        ],
    )
    prices = prices.sort_values(
        [
            "yahoo_symbol",
            "date",
        ],
        kind="mergesort",
    )
    prices["daily_return"] = (
        prices
        .groupby(
            "yahoo_symbol"
        )["close"]
        .pct_change()
    )
    prices["volatility_60d"] = (
        prices
        .groupby(
            "yahoo_symbol"
        )["daily_return"]
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
            "date",
            "volatility_60d",
        ]
    ].copy()
# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def prepare_data() -> pd.DataFrame:
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
        "security_key",
        "forward_return_5d",
        "snapshot_date",
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
    data["snapshot_date"] = pd.to_datetime(
        data["snapshot_date"],
        errors="coerce",
    )
    data = data.dropna(
        subset=[
            "snapshot_date",
            "forward_return_5d",
        ],
    ).copy()
    if "yahoo_symbol" not in data.columns:
        data["yahoo_symbol"] = (
            data["security_key"]
            .astype(str)
        )
    data["forward_return_5d"] = pd.to_numeric(
        data["forward_return_5d"],
        errors="coerce",
    )
    print(
        "Loading prices for volatility_60d..."
    )
    prices = load_price_data()
    volatility = build_volatility_60d(
        prices
    )
    volatility["date"] = pd.to_datetime(
        volatility["date"],
        errors="coerce",
    )
    data["price_date"] = data[
        "snapshot_date"
    ].dt.strftime(
        "%Y-%m-%d"
    )
    volatility["price_date"] = (
        volatility["date"]
        .dt.strftime("%Y-%m-%d")
    )
    volatility = (
        volatility.drop_duplicates(
            subset=[
                "yahoo_symbol",
                "price_date",
            ],
            keep="last",
        )
        .loc[
            :,
            [
                "yahoo_symbol",
                "price_date",
                "volatility_60d",
            ],
        ]
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
    data = data.dropna(
        subset=[
            "direction",
        ],
    ).copy()
    data["direction"] = (
        data["direction"]
        .astype(int)
    )
    return data
# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_logistic_model() -> Pipeline:
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
    if not feature_columns:
        return None
    frame = data[
        feature_columns
        + [target_column]
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
    if (
        frame[target_column]
        .nunique()
        < 2
    ):
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
    result = pd.Series(
        np.nan,
        index=data.index,
        dtype=float,
    )
    if (
        model is None
        or not feature_columns
    ):
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
        result.append(
            column
        )
    return result
def select_event_feature_set(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[
    str,
    list[str],
    float,
]:
    best_name = None
    best_columns: list[str] = []
    best_auc = -np.inf
    for (
        name,
        candidates,
    ) in EVENT_FEATURE_SETS.items():
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
        probabilities = (
            predict_probability(
                model,
                validation,
                columns,
            )
        )
        valid = (
            probabilities.notna()
        )
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
            probabilities.loc[
                valid
            ],
        )
        print(
            f"    event feature set "
            f"{name}: validation AUC="
            f"{auc:.6f}"
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
    train_scores = (
        predict_probability(
            model,
            train,
            columns,
        )
    )
    validation_scores = (
        predict_probability(
            model,
            validation,
            columns,
        )
    )
    test_scores = (
        predict_probability(
            model,
            test,
            columns,
        )
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
                len(values)
                * fraction
            )
        ),
    )
    count = min(
        count,
        len(values),
    )
    sorted_values = (
        values.sort_values(
            ascending=False
        )
    )
    return float(
        sorted_values.iloc[
            count - 1
        ]
    )
def apply_tail_threshold(
    data: pd.DataFrame,
    score_column: str,
    threshold: float,
) -> pd.DataFrame:
    frame = data.dropna(
        subset=[
            score_column,
        ]
    ).copy()
    return frame.loc[
        frame[score_column]
        >= threshold
    ].copy()
def prepare_direction_training_data(
    train: pd.DataFrame,
    fraction: float,
) -> tuple[
    pd.DataFrame,
    float | None,
]:
    threshold = tail_threshold(
        train,
        "event_score",
        fraction,
    )
    if threshold is None:
        return (
            train.iloc[0:0].copy(),
            None,
        )
    frame = apply_tail_threshold(
        train,
        "event_score",
        threshold,
    )
    frame = frame.loc[
        frame["event"] == 1
    ].copy()
    return (
        frame,
        threshold,
    )
def choose_fi_columns(
    data: pd.DataFrame,
) -> list[str]:
    columns = []
    for column in FI_COLUMNS:
        if column not in data.columns:
            continue
        values = numeric_series(
            data,
            column,
        )
        if values.notna().sum() == 0:
            continue
        columns.append(
            column
        )
    return columns
def prepare_direction_frame(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    frame = data.copy()
    for column in feature_columns:
        frame[column] = numeric_series(
            frame,
            column,
        )
    frame["direction"] = (
        pd.to_numeric(
            frame["direction"],
            errors="coerce",
        )
    )
    frame = frame.dropna(
        subset=(
            feature_columns
            + ["direction"]
        )
    )
    return frame
# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------
def paired_stratified_bootstrap_delta_auc(
    y_true: np.ndarray,
    baseline_score: np.ndarray,
    treatment_score: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """
    Calculate paired bootstrap samples of:
        AUC(treatment) - AUC(baseline)
    DOWN and UP are sampled separately.
    Each bootstrap draw uses exactly the same sampled observations
    for both models, preserving the pairing.
    """
    y_true = np.asarray(
        y_true,
        dtype=int,
    )
    baseline_score = np.asarray(
        baseline_score,
        dtype=float,
    )
    treatment_score = np.asarray(
        treatment_score,
        dtype=float,
    )
    valid = (
        np.isfinite(
            baseline_score
        )
        & np.isfinite(
            treatment_score
        )
    )
    y_true = y_true[valid]
    baseline_score = (
        baseline_score[valid]
    )
    treatment_score = (
        treatment_score[valid]
    )
    if len(y_true) < 20:
        return np.array(
            [],
            dtype=float,
        )
    down_indices = np.flatnonzero(
        y_true == 1
    )
    up_indices = np.flatnonzero(
        y_true == 0
    )
    if (
        len(down_indices) == 0
        or len(up_indices) == 0
    ):
        return np.array(
            [],
            dtype=float,
        )
    rng = np.random.default_rng(
        random_state
    )
    deltas = np.empty(
        iterations,
        dtype=float,
    )
    for iteration in range(
        iterations
    ):
        sampled_down = rng.choice(
            down_indices,
            size=len(down_indices),
            replace=True,
        )
        sampled_up = rng.choice(
            up_indices,
            size=len(up_indices),
            replace=True,
        )
        sampled = np.concatenate(
            [
                sampled_down,
                sampled_up,
            ]
        )
        y_sample = y_true[
            sampled
        ]
        baseline_sample = (
            baseline_score[
                sampled
            ]
        )
        treatment_sample = (
            treatment_score[
                sampled
            ]
        )
        try:
            baseline_auc = (
                roc_auc_score(
                    y_sample,
                    baseline_sample,
                )
            )
            treatment_auc = (
                roc_auc_score(
                    y_sample,
                    treatment_sample,
                )
            )
            deltas[iteration] = (
                treatment_auc
                - baseline_auc
            )
        except ValueError:
            deltas[iteration] = np.nan
    return deltas[
        np.isfinite(deltas)
    ]
def summarize_bootstrap(
    deltas: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
]:
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
        np.mean(
            deltas > 0
        )
    )
    probability_non_positive = float(
        np.mean(
            deltas <= 0
        )
    )
    return (
        mean,
        float(low),
        float(high),
        probability_positive,
        probability_non_positive,
    )
# ---------------------------------------------------------------------------
# Walk-forward window
# ---------------------------------------------------------------------------
def run_window(
    data: pd.DataFrame,
    window,
) -> list[DiagnosticResult]:
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
        data["snapshot_date"],
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
            "Skipping window because "
            "one period is empty."
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
    train["event_score"] = (
        train_scores
    )
    validation["event_score"] = (
        validation_scores
    )
    test["event_score"] = (
        test_scores
    )
    test_event = test.loc[
        test["event"] == 1
    ].copy()
    test_event = test_event.dropna(
        subset=[
            "event_score",
            "direction",
        ]
    )
    if test_event.empty:
        print(
            "No OOS events in test period."
        )
        return []
    fi_columns = choose_fi_columns(
        train
    )
    print()
    print(
        "FI columns available: "
        f"{len(fi_columns)}"
    )
    if fi_columns:
        print(
            "  "
            + ", ".join(
                fi_columns
            )
        )
    else:
        print(
            "  No FI columns found."
        )
    results: list[
        DiagnosticResult
    ] = []
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
        (
            train_direction,
            threshold,
        ) = prepare_direction_training_data(
            train,
            fraction,
        )
        if threshold is None:
            print(
                "No training threshold available."
            )
            continue
        test_tail = apply_tail_threshold(
            test_event,
            "event_score",
            threshold,
        )
        if test_tail.empty:
            print(
                "No OOS observations in tail."
            )
            continue
        down_count = int(
            (
                test_tail["direction"]
                == 1
            ).sum()
        )
        up_count = int(
            (
                test_tail["direction"]
                == 0
            ).sum()
        )
        print(
            f"Training tail events: "
            f"{len(train_direction):,}"
        )
        print(
            f"Test tail events:     "
            f"{len(test_tail):,}"
        )
        print(
            f"DOWN / UP: "
            f"{down_count:,} / "
            f"{up_count:,}"
        )
        if (
            down_count == 0
            or up_count == 0
        ):
            print(
                "Skipping because OOS tail "
                "has only one direction."
            )
            continue
        # ---------------------------------------------------------------
        # Baseline: event score only
        # ---------------------------------------------------------------
        baseline_features = [
            "event_score",
        ]
        baseline_train = (
            prepare_direction_frame(
                train_direction,
                baseline_features,
            )
        )
        baseline_model = fit_model(
            baseline_train,
            baseline_features,
            "direction",
        )
        baseline_predictions = (
            predict_probability(
                baseline_model,
                test_tail,
                baseline_features,
            )
        )
        # ---------------------------------------------------------------
        # Treatment: event score + FI
        # ---------------------------------------------------------------
        treatment_features = [
            "event_score",
            *fi_columns,
        ]
        treatment_train = (
            prepare_direction_frame(
                train_direction,
                treatment_features,
            )
        )
        treatment_model = fit_model(
            treatment_train,
            treatment_features,
            "direction",
        )
        treatment_predictions = (
            predict_probability(
                treatment_model,
                test_tail,
                treatment_features,
            )
        )
        frame = test_tail.copy()
        frame[
            "baseline_prediction"
        ] = baseline_predictions
        frame[
            "treatment_prediction"
        ] = treatment_predictions
        frame = frame.dropna(
            subset=[
                "direction",
                "baseline_prediction",
                "treatment_prediction",
            ]
        )
        if (
            frame["direction"]
            .nunique()
            < 2
        ):
            print(
                "Skipping because valid "
                "predictions contain only "
                "one direction."
            )
            continue
        auc_baseline = (
            roc_auc_score(
                frame["direction"],
                frame[
                    "baseline_prediction"
                ],
            )
        )
        auc_treatment = (
            roc_auc_score(
                frame["direction"],
                frame[
                    "treatment_prediction"
                ],
            )
        )
        delta_auc = (
            auc_treatment
            - auc_baseline
        )
        bootstrap_seed = (
            RANDOM_STATE
            + int(
                round(
                    fraction
                    * 10_000
                )
            )
            + pd.Timestamp(
                validation_end
            ).year
        )
        deltas = (
            paired_stratified_bootstrap_delta_auc(
                y_true=frame[
                    "direction"
                ].to_numpy(),
                baseline_score=frame[
                    "baseline_prediction"
                ].to_numpy(),
                treatment_score=frame[
                    "treatment_prediction"
                ].to_numpy(),
                iterations=(
                    BOOTSTRAP_ITERATIONS
                ),
                random_state=(
                    bootstrap_seed
                ),
            )
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
            f"Event-score AUC: "
            f"{auc_baseline:.6f}"
        )
        print(
            f"Event + FI AUC:  "
            f"{auc_treatment:.6f}"
        )
        print(
            f"Observed delta:   "
            f"{delta_auc:+.6f}"
        )
        print()
        print(
            f"Bootstrap mean:   "
            f"{bootstrap_mean:+.6f}"
        )
        print(
            "Bootstrap 95% CI: "
            f"[{ci_low:+.6f}, "
            f"{ci_high:+.6f}]"
        )
        print(
            f"P(delta > 0):     "
            f"{probability_positive:.4f}"
        )
        print(
            f"P(delta <= 0):    "
            f"{probability_non_positive:.4f}"
        )
        print(
            f"Bootstrap n:      "
            f"{len(deltas):,}"
        )
        results.append(
            DiagnosticResult(
                window=window_name,
                tail=fraction,
                test_events=len(frame),
                down_events=down_count,
                up_events=up_count,
                auc_event_score=auc_baseline,
                auc_event_fi=auc_treatment,
                delta_auc=delta_auc,
                bootstrap_mean_delta_auc=(
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
            )
        )
    return results
# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def results_to_frame(
    results: list[DiagnosticResult],
) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "window": result.window,
                "tail": result.tail,
                "test_events": result.test_events,
                "down_events": result.down_events,
                "up_events": result.up_events,
                "auc_event_score": (
                    result.auc_event_score
                ),
                "auc_event_fi": (
                    result.auc_event_fi
                ),
                "delta_auc": (
                    result.delta_auc
                ),
                "bootstrap_mean_delta_auc": (
                    result.bootstrap_mean_delta_auc
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
            }
            for result in results
        ]
    )
def print_summary(
    results: list[DiagnosticResult],
) -> None:
    frame = results_to_frame(
        results
    )
    print()
    print()
    print(
        "=" * 100
    )
    print(
        "PAIRED BOOTSTRAP SUMMARY"
    )
    print(
        "=" * 100
    )
    if frame.empty:
        print(
            "No results."
        )
        return
    display = frame.copy()
    display["tail"] = (
        display["tail"]
        * 100
    ).round(1)
    numeric_columns = [
        "auc_event_score",
        "auc_event_fi",
        "delta_auc",
        "bootstrap_mean_delta_auc",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "bootstrap_probability_positive",
        "bootstrap_probability_non_positive",
    ]
    display[
        numeric_columns
    ] = display[
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
        "  delta_auc > 0 means Event + FI "
        "performed better on the same OOS observations."
    )
    print(
        "  A 95% CI entirely above 0 means "
        "the central bootstrap interval does not "
        "contain zero."
    )
    print(
        "  This is evidence for incremental "
        "predictive information, not proof of "
        "a causal FI effect."
    )
def save_results(
    results: list[DiagnosticResult],
) -> Path | None:
    frame = results_to_frame(
        results
    )
    if frame.empty:
        return None
    output_path = (
        Path(__file__).resolve().parent
        / "fi_direction_bootstrap_results.csv"
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
    print()
    print(
        "=" * 80
    )
    print(
        "Blankdiss FI Direction Paired Bootstrap Diagnostic"
    )
    print(
        "=" * 80
    )
    print()
    print(
        "Question:"
    )
    print(
        "Does FI add statistically credible directional "
        "information above event-score?"
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
        data["snapshot_date"],
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
    all_results: list[
        DiagnosticResult
    ] = []
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
