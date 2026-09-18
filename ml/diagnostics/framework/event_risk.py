from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import ExperimentResult


RANDOM_STATE = 42
BOOTSTRAP_ITERATIONS = 2_000

EVENT_THRESHOLD = 0.10


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


def _prepare_event_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare event/direction labels.

    event:
        absolute 5d move >= 10%.

    direction:
        1 = DOWN <= -10%
        0 = UP >= +10%
        NaN = neither.
    """

    frame = frame.copy()

    if "forward_return_5d" not in frame.columns:
        raise KeyError(
            "Event-risk diagnostic requires "
            "'forward_return_5d'."
        )

    frame["forward_return_5d"] = _numeric(
        frame,
        "forward_return_5d",
    )

    frame["event"] = (
        frame["forward_return_5d"].abs()
        >= EVENT_THRESHOLD
    ).astype(float)

    frame["direction"] = np.where(
        frame["forward_return_5d"] <= -EVENT_THRESHOLD,
        1.0,
        np.where(
            frame["forward_return_5d"] >= EVENT_THRESHOLD,
            0.0,
            np.nan,
        ),
    )

    return frame


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
        [*features, "event"]
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

    valid = values.notna().all(axis=1)

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
) -> tuple[str, tuple[str, ...], float]:
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

        validation_scores = _predict_event_score(
            model,
            validation,
            features,
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

    return (
        best_name,
        best_features,
        float(best_auc),
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
    context,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    str,
    tuple[str, ...],
    float,
]:
    train = _prepare_event_frame(
        context.train
    )

    validation = _prepare_event_frame(
        context.validation
    )

    pretest = _prepare_event_frame(
        context.pretest
    )

    test = _prepare_event_frame(
        context.test
    )

    (
        model_name,
        features,
        validation_auc,
    ) = _select_event_model(
        train,
        validation,
    )

    model = _fit_selected_model_on_pretest(
        pretest,
        features,
    )

    pretest["event_score"] = _predict_event_score(
        model,
        pretest,
        features,
    )

    test["event_score"] = _predict_event_score(
        model,
        test,
        features,
    )

    return (
        pretest,
        test,
        model_name,
        features,
        validation_auc,
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


def _bootstrap_difference(
    first: pd.Series,
    second: pd.Series,
) -> tuple[float, float, float, float]:
    first_values = pd.to_numeric(
        first,
        errors="coerce",
    ).dropna().to_numpy()

    second_values = pd.to_numeric(
        second,
        errors="coerce",
    ).dropna().to_numpy()

    if (
        len(first_values) == 0
        or len(second_values) == 0
    ):
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

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
        first_sample = rng.choice(
            first_values,
            size=len(first_values),
            replace=True,
        )

        second_sample = rng.choice(
            second_values,
            size=len(second_values),
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


def _bootstrap_interaction(
    risk_low: pd.Series,
    risk_high: pd.Series,
    outside_low: pd.Series,
    outside_high: pd.Series,
) -> tuple[float, float, float, float]:
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
        float(np.quantile(deltas, 0.025)),
        float(np.quantile(deltas, 0.975)),
        float((deltas > 0).mean()),
    )


def _safe_spearman(
    x: pd.Series,
    y: pd.Series,
) -> tuple[float, float]:
    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(
                x,
                errors="coerce",
            ),
            "y": pd.to_numeric(
                y,
                errors="coerce",
            ),
        }
    ).dropna()

    if len(frame) < 3:
        return float("nan"), float("nan")

    if frame["x"].nunique() < 2:
        return float("nan"), float("nan")

    if frame["y"].nunique() < 2:
        return float("nan"), float("nan")

    rho, p_value = spearmanr(
        frame["x"],
        frame["y"],
    )

    return (
        float(rho),
        float(p_value),
    )


def _require_column(
    frame: pd.DataFrame,
    column: str,
) -> None:
    if column not in frame.columns:
        raise KeyError(
            f"Event-risk diagnostic requires "
            f"'{column}'."
        )


def _interaction_groups(
    frame: pd.DataFrame,
    *,
    risk_fraction: float,
    high_column: str,
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
        & frame[high_column]
    ]

    risk_low = frame[
        (frame["event_risk_group"] == risk_label)
        & (~frame[high_column])
    ]

    outside_high = frame[
        (frame["event_risk_group"] != risk_label)
        & frame[high_column]
    ]

    outside_low = frame[
        (frame["event_risk_group"] != risk_label)
        & (~frame[high_column])
    ]

    return (
        risk_low,
        risk_high,
        outside_low,
        outside_high,
    )


def run_si_level_confirmation(
    context,
    *,
    risk_cutoffs: tuple[float, ...],
    short_interest_cutoff: float,
) -> ExperimentResult:
    (
        pretest,
        test,
        model_name,
        features,
        validation_auc,
    ) = _prepare_event_risk_data(
        context
    )

    _require_column(
        pretest,
        "short_interest_pct",
    )

    _require_column(
        test,
        "short_interest_pct",
    )

    si_threshold = _tail_threshold(
        pretest["short_interest_pct"],
        short_interest_cutoff,
    )

    if not np.isfinite(si_threshold):
        raise RuntimeError(
            "Could not calculate short-interest "
            "threshold from pre-test data."
        )

    test = test.copy()

    test["down"] = (
        test["direction"] == 1.0
    )

    test["high_si"] = (
        pd.to_numeric(
            test["short_interest_pct"],
            errors="coerce",
        )
        >= si_threshold
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
            test["event_score"] >= risk_threshold,
            _risk_label(risk_fraction),
            "outside",
        )

        (
            risk_low,
            risk_high,
            outside_low,
            outside_high,
        ) = _interaction_groups(
            test,
            risk_fraction=risk_fraction,
            high_column="high_si",
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

        risk_down_low = (
            risk_low["down"].mean()
        )

        risk_down_high = (
            risk_high["down"].mean()
        )

        outside_down_low = (
            outside_low["down"].mean()
        )

        outside_down_high = (
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

        tail = test[
            test["event_risk_group"]
            == _risk_label(risk_fraction)
        ].copy()

        rho_down, p_down = _safe_spearman(
            tail["short_interest_pct"],
            tail["down"].astype(float),
        )

        rho_return, p_return = _safe_spearman(
            tail["short_interest_pct"],
            tail["forward_return_5d"],
        )

        rows.append(
            {
                "risk_cutoff": risk_fraction,
                "risk_threshold": risk_threshold,
                "short_interest_cutoff": (
                    short_interest_cutoff
                ),
                "short_interest_threshold": (
                    si_threshold
                ),
                "risk_low_n": len(risk_low),
                "risk_high_n": len(risk_high),
                "outside_low_n": len(outside_low),
                "outside_high_n": len(outside_high),
                "risk_low_down_rate": (
                    risk_down_low
                ),
                "risk_high_down_rate": (
                    risk_down_high
                ),
                "outside_low_down_rate": (
                    outside_down_low
                ),
                "outside_high_down_rate": (
                    outside_down_high
                ),
                "risk_si_effect": (
                    risk_down_high
                    - risk_down_low
                ),
                "outside_si_effect": (
                    outside_down_high
                    - outside_down_low
                ),
                "si_event_interaction": (
                    interaction
                ),
                "interaction_ci_low": ci_low,
                "interaction_ci_high": ci_high,
                "interaction_p_positive": (
                    p_positive
                ),
                "continuous_si_rho_down": (
                    rho_down
                ),
                "continuous_si_p_down": (
                    p_down
                ),
                "continuous_si_rho_return": (
                    rho_return
                ),
                "continuous_si_p_return": (
                    p_return
                ),
            }
        )

    result = ExperimentResult(
        name="si_level_event_risk_confirmation",
        description=(
            "Bekräftar om short-interest-nivå "
            "tillför information inom extrem "
            "event-risk."
        ),
    )

    result.add_metric(
        "event_model",
        model_name,
    )

    result.add_metric(
        "event_features",
        list(features),
    )

    result.add_metric(
        "validation_auc",
        validation_auc,
    )

    result.add_metric(
        "event_threshold",
        EVENT_THRESHOLD,
    )

    result.add_metric(
        "short_interest_cutoff",
        short_interest_cutoff,
    )

    result.add_table(
        "risk_tail_results",
        pd.DataFrame(rows),
    )

    return result


def run_event_risk_interaction(
    context,
    *,
    risk_cutoffs: tuple[float, ...],
    positive_change_cutoff: float,
) -> ExperimentResult:
    (
        pretest,
        test,
        model_name,
        features,
        validation_auc,
    ) = _prepare_event_risk_data(
        context
    )

    _require_column(
        pretest,
        "short_interest_pct_change",
    )

    _require_column(
        test,
        "short_interest_pct_change",
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
            test["event_score"] >= risk_threshold,
            _risk_label(risk_fraction),
            "outside",
        )

        # The original diagnostic asks specifically whether
        # the magnitude of positive short-interest changes
        # behaves differently inside the event-risk tail.
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
            high_column="high_change",
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

    result = ExperimentResult(
        name="fi_short_interest_event_risk_interaction",
        description=(
            "Testar om effekten av "
            "short-interest-förändring är starkare "
            "vid extrem event-risk."
        ),
    )

    result.add_metric(
        "event_model",
        model_name,
    )

    result.add_metric(
        "event_features",
        list(features),
    )

    result.add_metric(
        "validation_auc",
        validation_auc,
    )

    result.add_metric(
        "event_threshold",
        EVENT_THRESHOLD,
    )

    result.add_metric(
        "positive_change_cutoff",
        positive_change_cutoff,
    )

    result.add_table(
        "risk_tail_results",
        pd.DataFrame(rows),
    )

    return result
