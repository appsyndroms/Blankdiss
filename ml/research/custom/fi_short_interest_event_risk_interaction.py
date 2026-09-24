from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
BOOTSTRAP_ITERATIONS = 2_000

EVENT_THRESHOLD = 0.10

RISK_CUTOFFS = (
    0.20,
    0.10,
    0.05,
    0.025,
    0.01,
)

POSITIVE_CHANGE_CUTOFF = 0.20

EVENT_FEATURE_SETS: dict[str, tuple[str, ...]] = {
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
class EventModelSelection:
    name: str
    features: tuple[str, ...]
    validation_auc: float


def _numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def _add_short_interest_dynamics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preserve the original diagnostic semantics.

    Short-interest change is calculated independently within the
    supplied frame. This is important because the old diagnostic
    prepares train, validation, pretest and test separately.
    """

    result = frame.copy()

    if "short_interest_pct" not in result.columns:
        raise KeyError(
            "Short-interest dynamics requires "
            "'short_interest_pct'."
        )

    if "snapshot_date" not in result.columns:
        raise KeyError(
            "Short-interest dynamics requires "
            "'snapshot_date'."
        )

    if "yahoo_symbol" in result.columns:
        group_column = "yahoo_symbol"
    elif "security_key" in result.columns:
        group_column = "security_key"
    else:
        raise KeyError(
            "Short-interest dynamics requires either "
            "'yahoo_symbol' or 'security_key'."
        )

    result["short_interest_pct"] = _numeric(
        result,
        "short_interest_pct",
    )

    result["snapshot_date"] = pd.to_datetime(
        result["snapshot_date"],
        errors="coerce",
    )

    result = result.sort_values(
        [
            group_column,
            "snapshot_date",
        ]
    ).copy()

    previous = (
        result
        .groupby(
            group_column,
            sort=False,
        )["short_interest_pct"]
        .shift(1)
    )

    current = result[
        "short_interest_pct"
    ]

    result["short_interest_pct_change"] = (
        current - previous
    )

    denominator = previous.abs()

    result["short_interest_pct_change_pct"] = (
        (current - previous)
        .div(
            denominator.where(
                denominator > 0
            )
        )
    )

    return result


def _prepare_event_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = _add_short_interest_dynamics(
        frame
    )

    if "forward_return_5d" not in result.columns:
        raise KeyError(
            "Event-risk diagnostic requires "
            "'forward_return_5d'."
        )

    result["forward_return_5d"] = _numeric(
        result,
        "forward_return_5d",
    )

    result["event"] = (
        result["forward_return_5d"].abs()
        >= EVENT_THRESHOLD
    ).astype(float)

    result["direction"] = np.where(
        result["forward_return_5d"]
        <= -EVENT_THRESHOLD,
        1.0,
        np.where(
            result["forward_return_5d"]
            >= EVENT_THRESHOLD,
            0.0,
            np.nan,
        ),
    )

    return result


def _build_event_model() -> Pipeline:
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


def _fit_event_model(
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> Pipeline | None:
    missing = [
        column
        for column in features
        if column not in frame.columns
    ]

    if missing:
        return None

    model_frame = frame[
        [
            *features,
            "event",
        ]
    ].copy()

    for column in features:
        model_frame[column] = _numeric(
            model_frame,
            column,
        )

    model_frame["event"] = _numeric(
        model_frame,
        "event",
    )

    model_frame = model_frame.dropna()

    if len(model_frame) < 100:
        return None

    if model_frame["event"].nunique() < 2:
        return None

    model = _build_event_model()

    model.fit(
        model_frame[list(features)],
        model_frame["event"],
    )

    return model


def _predict_event_score(
    model: Pipeline,
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> pd.Series:
    scores = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    missing = [
        column
        for column in features
        if column not in frame.columns
    ]

    if missing:
        return scores

    values = frame[
        list(features)
    ].copy()

    for column in features:
        values[column] = _numeric(
            values,
            column,
        )

    valid = values.notna().all(
        axis=1
    )

    if not valid.any():
        return scores

    scores.loc[valid] = (
        model.predict_proba(
            values.loc[
                valid,
                list(features),
            ]
        )[:, 1]
    )

    return scores


def _safe_auc(
    y_true: pd.Series,
    scores: pd.Series,
) -> float:
    frame = pd.DataFrame(
        {
            "y": pd.to_numeric(
                y_true,
                errors="coerce",
            ),
            "score": pd.to_numeric(
                scores,
                errors="coerce",
            ),
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


def _select_event_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> EventModelSelection:
    best_name = ""
    best_features: tuple[str, ...] = ()
    best_auc = -np.inf

    for name, features in EVENT_FEATURE_SETS.items():
        model = _fit_event_model(
            train,
            features,
        )

        if model is None:
            continue

        validation_scores = (
            _predict_event_score(
                model,
                validation,
                features,
            )
        )

        auc = _safe_auc(
            validation["event"],
            validation_scores,
        )

        if np.isnan(auc):
            continue

        if auc > best_auc:
            best_name = name
            best_features = features
            best_auc = auc

    if not best_features:
        raise RuntimeError(
            "Could not select an event-risk model."
        )

    return EventModelSelection(
        name=best_name,
        features=best_features,
        validation_auc=float(best_auc),
    )


def _fit_selected_model_on_pretest(
    pretest: pd.DataFrame,
    features: tuple[str, ...],
) -> Pipeline:
    model = _fit_event_model(
        pretest,
        features,
    )

    if model is None:
        raise RuntimeError(
            "Could not refit selected event-risk model "
            "on pre-test data."
        )

    return model


def _prepare_event_risk_data(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    pretest: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    EventModelSelection,
]:
    train = _prepare_event_frame(
        train
    )

    validation = _prepare_event_frame(
        validation
    )

    pretest = _prepare_event_frame(
        pretest
    )

    test = _prepare_event_frame(
        test
    )

    selection = _select_event_model(
        train,
        validation,
    )

    model = _fit_selected_model_on_pretest(
        pretest,
        selection.features,
    )

    pretest["event_score"] = (
        _predict_event_score(
            model,
            pretest,
            selection.features,
        )
    )

    test["event_score"] = (
        _predict_event_score(
            model,
            test,
            selection.features,
        )
    )

    return (
        pretest,
        test,
        selection,
    )


def _tail_threshold(
    scores: pd.Series,
    fraction: float,
) -> float:
    values = pd.to_numeric(
        scores,
        errors="coerce",
    ).dropna()

    if values.empty:
        return float("nan")

    return float(
        values.quantile(
            1.0 - fraction
        )
    )


def _risk_label(
    fraction: float,
) -> str:
    return f"top_{fraction:g}"


def _bootstrap_interaction(
    risk_low: pd.Series,
    risk_high: pd.Series,
    outside_low: pd.Series,
    outside_high: pd.Series,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    groups = [
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .dropna()
        .to_numpy()
        for values in (
            risk_low,
            risk_high,
            outside_low,
            outside_high,
        )
    ]

    if any(
        len(values) == 0
        for values in groups
    ):
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    (
        risk_low_values,
        risk_high_values,
        outside_low_values,
        outside_high_values,
    ) = groups

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    deltas = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=float,
    )

    for index in range(
        BOOTSTRAP_ITERATIONS
    ):
        rl = rng.choice(
            risk_low_values,
            size=len(risk_low_values),
            replace=True,
        )

        rh = rng.choice(
            risk_high_values,
            size=len(risk_high_values),
            replace=True,
        )

        ol = rng.choice(
            outside_low_values,
            size=len(outside_low_values),
            replace=True,
        )

        oh = rng.choice(
            outside_high_values,
            size=len(outside_high_values),
            replace=True,
        )

        deltas[index] = (
            rh.mean()
            - rl.mean()
            - oh.mean()
            + ol.mean()
        )

    observed = (
        risk_high_values.mean()
        - risk_low_values.mean()
        - outside_high_values.mean()
        + outside_low_values.mean()
    )

    return (
        float(observed),
        float(
            np.quantile(
                deltas,
                0.025,
            )
        ),
        float(
            np.quantile(
                deltas,
                0.975,
            )
        ),
        float(
            (deltas > 0).mean()
        ),
    )


def _interaction_groups(
    frame: pd.DataFrame,
    *,
    risk_fraction: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    risk_label = _risk_label(
        risk_fraction
    )

    risk_high = frame[
        (frame["event_risk_group"] == risk_label)
        & frame["high_change"]
    ]

    risk_low = frame[
        (frame["event_risk_group"] == risk_label)
        & (~frame["high_change"])
    ]

    outside_high = frame[
        (frame["event_risk_group"] != risk_label)
        & frame["high_change"]
    ]

    outside_low = frame[
        (frame["event_risk_group"] != risk_label)
        & (~frame["high_change"])
    ]

    return (
        risk_low,
        risk_high,
        outside_low,
        outside_high,
    )


def run(
    frame: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    test_end: str | pd.Timestamp,
    risk_cutoffs: tuple[float, ...] = RISK_CUTOFFS,
    positive_change_cutoff: float = (
        POSITIVE_CHANGE_CUTOFF
    ),
) -> tuple[
    pd.DataFrame,
    dict[str, object],
]:
    """
    Run the FI short-interest/event-risk interaction analysis.

    The implementation deliberately preserves the old diagnostic's
    split-local short-interest dynamics:

        train
        validation
        pretest
        test

    each receives its own short-interest diff calculation.
    """

    data = frame.copy()

    data["snapshot_date"] = pd.to_datetime(
        data["snapshot_date"],
        errors="coerce",
    )

    train_end = pd.Timestamp(
        train_end
    )

    validation_end = pd.Timestamp(
        validation_end
    )

    test_end = pd.Timestamp(
        test_end
    )

    train = data[
        data["snapshot_date"]
        <= train_end
    ].copy()

    validation = data[
        (
            data["snapshot_date"]
            > train_end
        )
        & (
            data["snapshot_date"]
            <= validation_end
        )
    ].copy()

    pretest = data[
        data["snapshot_date"]
        <= validation_end
    ].copy()

    test = data[
        (
            data["snapshot_date"]
            > validation_end
        )
        & (
            data["snapshot_date"]
            <= test_end
        )
    ].copy()

    (
        pretest,
        test,
        selection,
    ) = _prepare_event_risk_data(
        train,
        validation,
        pretest,
        test,
    )

    if (
        "short_interest_pct_change"
        not in pretest.columns
    ):
        raise KeyError(
            "Event-risk diagnostic requires "
            "'short_interest_pct_change'."
        )

    if (
        "short_interest_pct_change"
        not in test.columns
    ):
        raise KeyError(
            "Event-risk diagnostic requires "
            "'short_interest_pct_change'."
        )

    change = pd.to_numeric(
        pretest[
            "short_interest_pct_change"
        ],
        errors="coerce",
    )

    positive_change = change[
        change > 0
    ].dropna()

    if positive_change.empty:
        raise RuntimeError(
            "Could not calculate positive "
            "short-interest change threshold."
        )

    change_threshold = float(
        positive_change.quantile(
            1.0 - positive_change_cutoff
        )
    )

    test = test.copy()

    test_change = pd.to_numeric(
        test[
            "short_interest_pct_change"
        ],
        errors="coerce",
    )

    test["positive_change"] = (
        test_change > 0
    )

    test["high_change"] = (
        test_change >= change_threshold
    )

    test["down"] = (
        test["direction"] == 1.0
    )

    rows: list[dict] = []

    for risk_fraction in risk_cutoffs:
        risk_threshold = _tail_threshold(
            pretest["event_score"],
            risk_fraction,
        )

        if not np.isfinite(
            risk_threshold
        ):
            continue

        test["event_risk_group"] = np.where(
            test["event_score"]
            >= risk_threshold,
            _risk_label(
                risk_fraction
            ),
            "outside",
        )

        local = test[
            test["positive_change"]
        ].copy()

        (
            risk_low,
            risk_high,
            outside_low,
            outside_high,
        ) = _interaction_groups(
            local,
            risk_fraction=risk_fraction,
        )

        if any(
            len(group) == 0
            for group in (
                risk_low,
                risk_high,
                outside_low,
                outside_high,
            )
        ):
            continue

        risk_low_rate = (
            risk_low["down"].mean()
        )

        risk_high_rate = (
            risk_high["down"].mean()
        )

        outside_low_rate = (
            outside_low["down"].mean()
        )

        outside_high_rate = (
            outside_high["down"].mean()
        )

        (
            interaction,
            ci_low,
            ci_high,
            p_positive,
        ) = _bootstrap_interaction(
            risk_low["down"].astype(float),
            risk_high["down"].astype(float),
            outside_low["down"].astype(float),
            outside_high["down"].astype(float),
        )

        rows.append(
            {
                "risk_cutoff": risk_fraction,
                "risk_threshold": risk_threshold,
                "positive_change_cutoff": (
                    positive_change_cutoff
                ),
                "positive_change_threshold": (
                    change_threshold
                ),
                "risk_low_n": len(risk_low),
                "risk_high_n": len(risk_high),
                "outside_low_n": len(outside_low),
                "outside_high_n": len(outside_high),
                "risk_low_down_rate": (
                    risk_low_rate
                ),
                "risk_high_down_rate": (
                    risk_high_rate
                ),
                "outside_low_down_rate": (
                    outside_low_rate
                ),
                "outside_high_down_rate": (
                    outside_high_rate
                ),
                "risk_change_effect": (
                    risk_high_rate
                    - risk_low_rate
                ),
                "outside_change_effect": (
                    outside_high_rate
                    - outside_low_rate
                ),
                "change_event_interaction": (
                    interaction
                ),
                "interaction_ci_low": ci_low,
                "interaction_ci_high": ci_high,
                "interaction_p_positive": (
                    p_positive
                ),
            }
        )

    metrics = {
        "event_model": selection.name,
        "event_features": list(
            selection.features
        ),
        "validation_auc": (
            selection.validation_auc
        ),
        "event_threshold": EVENT_THRESHOLD,
        "positive_change_cutoff": (
            positive_change_cutoff
        ),
    }

    return (
        pd.DataFrame(rows),
        metrics,
    )
