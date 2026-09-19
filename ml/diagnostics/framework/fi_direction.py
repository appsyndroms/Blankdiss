from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import ExperimentResult


BOOTSTRAP_ITERATIONS = 2_000
RANDOM_STATE = 42
EVENT_THRESHOLD = 0.10

TOP_FRACTIONS = (
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
)

FI_COLUMNS = (
    "short_interest",
    "short_interest_change",
    "short_interest_pct",
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


def _numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    values = frame[column]

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


def _existing_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> list[str]:
    result: list[str] = []

    for column in columns:
        if column not in frame.columns:
            continue

        values = _numeric(
            frame,
            column,
        )

        if values.notna().sum() == 0:
            continue

        result.append(column)

    return result


def _model() -> Pipeline:
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


def _fit(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
) -> Pipeline | None:
    if not features:
        return None

    data = frame[
        features + [target]
    ].copy()

    for column in features:
        data[column] = _numeric(
            data,
            column,
        )

    data[target] = _numeric(
        data,
        target,
    )

    data = data.dropna()

    if len(data) < 30:
        return None

    if data[target].nunique() < 2:
        return None

    model = _model()

    model.fit(
        data[features],
        data[target],
    )

    return model


def _predict(
    model: Pipeline | None,
    frame: pd.DataFrame,
    features: list[str],
) -> pd.Series:
    result = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    if model is None or not features:
        return result

    data = frame[
        features
    ].copy()

    for column in features:
        data[column] = _numeric(
            data,
            column,
        )

    valid = data.notna().all(axis=1)

    if not valid.any():
        return result

    result.loc[valid] = (
        model.predict_proba(
            data.loc[
                valid,
                features,
            ]
        )[:, 1]
    )

    return result


def _auc(
    y_true: pd.Series,
    score: pd.Series,
) -> float:
    valid = (
        y_true.notna()
        & score.notna()
    )

    if valid.sum() < 30:
        return np.nan

    y = y_true.loc[valid]

    if y.nunique() < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            score.loc[valid],
        )
    )


def _tail_threshold(
    frame: pd.DataFrame,
    score_column: str,
    fraction: float,
) -> float | None:
    values = _numeric(
        frame,
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

    values = values.sort_values(
        ascending=False
    )

    return float(
        values.iloc[count - 1]
    )


def _apply_tail_threshold(
    frame: pd.DataFrame,
    score_column: str,
    threshold: float,
) -> pd.DataFrame:
    result = frame.dropna(
        subset=[
            score_column,
        ]
    ).copy()

    return result.loc[
        result[score_column] >= threshold
    ].copy()


def _paired_stratified_bootstrap_delta_auc(
    y_true: np.ndarray,
    baseline_score: np.ndarray,
    treatment_score: np.ndarray,
    iterations: int,
    random_state: int,
) -> np.ndarray:
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
        np.isfinite(baseline_score)
        & np.isfinite(treatment_score)
    )

    y_true = y_true[valid]
    baseline_score = baseline_score[valid]
    treatment_score = treatment_score[valid]

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

    for iteration in range(iterations):
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
            baseline_score[sampled]
        )

        treatment_sample = (
            treatment_score[sampled]
        )

        try:
            baseline_auc = roc_auc_score(
                y_sample,
                baseline_sample,
            )

            treatment_auc = roc_auc_score(
                y_sample,
                treatment_sample,
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


def _summarize_bootstrap(
    deltas: np.ndarray,
) -> dict[str, float]:
    if deltas.size == 0:
        return {
            "bootstrap_mean_delta_auc": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
            "bootstrap_probability_positive": np.nan,
            "bootstrap_probability_non_positive": np.nan,
        }

    low, high = np.percentile(
        deltas,
        [
            2.5,
            97.5,
        ],
    )

    return {
        "bootstrap_mean_delta_auc": float(
            np.mean(deltas)
        ),
        "bootstrap_ci_low": float(
            low
        ),
        "bootstrap_ci_high": float(
            high
        ),
        "bootstrap_probability_positive": float(
            np.mean(deltas > 0)
        ),
        "bootstrap_probability_non_positive": float(
            np.mean(deltas <= 0)
        ),
    }


def _select_event_feature_set(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[
    str,
    list[str],
    float,
]:
    best_name: str | None = None
    best_columns: list[str] = []
    best_auc = -np.inf

    for name, candidates in (
        EVENT_FEATURE_SETS.items()
    ):
        columns = _existing_columns(
            train,
            candidates,
        )

        if not columns:
            continue

        model = _fit(
            train,
            columns,
            "event",
        )

        if model is None:
            continue

        probabilities = _predict(
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

        auc = float(
            roc_auc_score(
                y_true,
                probabilities.loc[valid],
            )
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


def _build_event_scores(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
    str,
    list[str],
    float,
]:
    (
        name,
        columns,
        validation_auc,
    ) = _select_event_feature_set(
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

    model = _fit(
        combined,
        columns,
        "event",
    )

    if model is None:
        raise RuntimeError(
            "Could not fit selected event model."
        )

    train_scores = _predict(
        model,
        train,
        columns,
    )

    validation_scores = _predict(
        model,
        validation,
        columns,
    )

    test_scores = _predict(
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
        validation_auc,
    )


def _prepare_direction_frame(
    frame: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    result = frame.copy()

    for column in features:
        result[column] = _numeric(
            result,
            column,
        )

    result["direction"] = pd.to_numeric(
        result["direction"],
        errors="coerce",
    )

    result = result.dropna(
        subset=features + ["direction"]
    )

    return result


def run_incremental_fi_bootstrap(
    context,
    *,
    fi_columns: tuple[str, ...],
    tail_fractions: tuple[float, ...],
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> ExperimentResult:
    train = context.train.copy()
    validation = context.validation.copy()
    test = context.test.copy()

    required = [
        "forward_return_5d",
    ]

    missing = [
        column
        for column in required
        if column not in context.data.columns
    ]

    if missing:
        raise KeyError(
            "Missing required columns: "
            f"{missing}"
        )

    for frame in (
        train,
        validation,
        test,
    ):
        frame["forward_return_5d"] = (
            _numeric(
                frame,
                "forward_return_5d",
            )
        )

        frame["event"] = (
            frame["forward_return_5d"].abs()
            >= EVENT_THRESHOLD
        ).astype(int)

        frame["direction"] = np.where(
            frame["forward_return_5d"]
            <= -EVENT_THRESHOLD,
            1,
            np.where(
                frame["forward_return_5d"]
                >= EVENT_THRESHOLD,
                0,
                np.nan,
            ),
        )

    (
        train_event_score,
        validation_event_score,
        test_event_score,
        event_model_name,
        event_model_features,
        event_model_validation_auc,
    ) = _build_event_scores(
        train,
        validation,
        test,
    )

    train["event_score"] = (
        train_event_score
    )

    validation["event_score"] = (
        validation_event_score
    )

    test["event_score"] = (
        test_event_score
    )

    available_fi_columns = _existing_columns(
        train,
        fi_columns,
    )

    print()
    print(
        "FI columns available: "
        f"{len(available_fi_columns)}"
    )

    if available_fi_columns:
        print(
            "  "
            + ", ".join(
                available_fi_columns
            )
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

    rows = []

    for fraction in tail_fractions:
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

        threshold = _tail_threshold(
            train,
            "event_score",
            fraction,
        )

        if threshold is None:
            print(
                "No training threshold available."
            )
            continue

        train_direction = _apply_tail_threshold(
            train,
            "event_score",
            threshold,
        )

        train_direction = train_direction.loc[
            train_direction["event"] == 1
        ].copy()

        test_tail = _apply_tail_threshold(
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
                test_tail["direction"] == 1
            ).sum()
        )

        up_count = int(
            (
                test_tail["direction"] == 0
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
            f"{down_count:,} / {up_count:,}"
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

        baseline_features = [
            "event_score",
        ]

        baseline_train = (
            _prepare_direction_frame(
                train_direction,
                baseline_features,
            )
        )

        baseline_model = _fit(
            baseline_train,
            baseline_features,
            "direction",
        )

        baseline_predictions = _predict(
            baseline_model,
            test_tail,
            baseline_features,
        )

        treatment_features = [
            "event_score",
            *available_fi_columns,
        ]

        treatment_train = (
            _prepare_direction_frame(
                train_direction,
                treatment_features,
            )
        )

        treatment_model = _fit(
            treatment_train,
            treatment_features,
            "direction",
        )

        treatment_predictions = _predict(
            treatment_model,
            test_tail,
            treatment_features,
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
            len(frame) < 20
            or frame["direction"].nunique() < 2
        ):
            print(
                "Skipping because valid predictions "
                "do not contain both directions."
            )
            continue

        auc_baseline = float(
            roc_auc_score(
                frame["direction"],
                frame[
                    "baseline_prediction"
                ],
            )
        )

        auc_treatment = float(
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

        validation_year = (
            pd.Timestamp(
                context.validation_end
            ).year
        )

        bootstrap_seed = (
            RANDOM_STATE
            + int(
                round(
                    fraction * 10_000
                )
            )
            + validation_year
        )

        deltas = (
            _paired_stratified_bootstrap_delta_auc(
                y_true=frame[
                    "direction"
                ].to_numpy(),
                baseline_score=frame[
                    "baseline_prediction"
                ].to_numpy(),
                treatment_score=frame[
                    "treatment_prediction"
                ].to_numpy(),
                iterations=bootstrap_iterations,
                random_state=bootstrap_seed,
            )
        )

        bootstrap = _summarize_bootstrap(
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

        print(
            f"Bootstrap mean:   "
            f"{bootstrap['bootstrap_mean_delta_auc']:+.6f}"
        )

        print(
            "Bootstrap 95% CI: "
            f"["
            f"{bootstrap['bootstrap_ci_low']:+.6f}, "
            f"{bootstrap['bootstrap_ci_high']:+.6f}"
            f"]"
        )

        print(
            f"P(delta > 0):     "
            f"{bootstrap['bootstrap_probability_positive']:.4f}"
        )

        print(
            f"P(delta <= 0):    "
            f"{bootstrap['bootstrap_probability_non_positive']:.4f}"
        )

        print(
            f"Bootstrap n:      "
            f"{len(deltas):,}"
        )

        rows.append(
            {
                "tail": fraction,
                "training_tail_events": int(
                    len(train_direction)
                ),
                "test_events": int(
                    len(frame)
                ),
                "down_events": down_count,
                "up_events": up_count,
                "event_score_threshold": (
                    threshold
                ),
                "auc_event_score": auc_baseline,
                "auc_event_fi": auc_treatment,
                "delta_auc": delta_auc,
                **bootstrap,
            }
        )

    result = ExperimentResult(
        name="fi_direction_bootstrap",
        description=(
            "Paired bootstrap av inkrementell "
            "FI-information ovanpå event-risk."
        ),
    )

    result.add_table(
        "paired_bootstrap",
        pd.DataFrame(rows),
    )

    result.add_metadata(
        "event_model",
        event_model_name,
    )

    result.add_metadata(
        "event_model_features",
        event_model_features,
    )

    result.add_metric(
        "event_model_validation_auc",
        event_model_validation_auc,
    )

    result.add_metadata(
        "fi_columns",
        available_fi_columns,
    )

    result.add_metadata(
        "bootstrap_iterations",
        bootstrap_iterations,
    )

    result.add_metadata(
        "event_threshold",
        EVENT_THRESHOLD,
    )

    return result
