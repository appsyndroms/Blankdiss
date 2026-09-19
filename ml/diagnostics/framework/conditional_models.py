from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import ExperimentResult
from .event_risk import (
    _fit_event_model,
    _fit_selected_model_on_pretest,
    _predict_event_score,
    _prepare_event_frame,
    _select_event_model,
)


RANDOM_STATE = 42

TARGET_COLUMN = "forward_return_5d"
TARGET_THRESHOLD = -0.05

SI_COLUMN = "short_interest_pct_change"
PRIOR_RETURN_COLUMN = "price_return_5d"


MODEL_FEATURES = {
    "si_only": (
        SI_COLUMN,
    ),
    "event_risk_only": (
        "event_score",
    ),
    "si_plus_event_risk": (
        SI_COLUMN,
        "event_score",
    ),
    "si_plus_event_risk_plus_prior_return": (
        SI_COLUMN,
        "event_score",
        PRIOR_RETURN_COLUMN,
    ),
}


def _numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def _build_model() -> Pipeline:
    return Pipeline(
        [
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
                    max_iter=2_000,
                    C=1.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _fit_model(
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> Pipeline:
    missing = [
        column
        for column in features
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            "Conditional model saknar kolumner: "
            + ", ".join(missing)
        )

    values = frame[
        list(features)
    ].copy()

    for column in features:
        values[column] = _numeric(
            values,
            column,
        )

    target = (
        _numeric(
            frame,
            TARGET_COLUMN,
        )
        <= TARGET_THRESHOLD
    ).astype(float)

    valid = (
        values.notna().any(axis=1)
        & target.notna()
    )

    values = values.loc[valid]
    target = target.loc[valid]

    if len(values) < 100:
        raise RuntimeError(
            "För få rader för conditional model: "
            f"{len(values)}"
        )

    if target.nunique() < 2:
        raise RuntimeError(
            "Conditional target innehåller "
            "inte båda klasserna."
        )

    model = _build_model()

    model.fit(
        values[list(features)],
        target,
    )

    return model


def _predict(
    model: Pipeline,
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> pd.Series:
    scores = pd.Series(
        np.nan,
        index=frame.index,
        dtype=float,
    )

    values = frame[
        list(features)
    ].copy()

    for column in features:
        values[column] = _numeric(
            values,
            column,
        )

    valid = values.notna().any(axis=1)

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


def _classification_metrics(
    frame: pd.DataFrame,
    scores: pd.Series,
) -> dict[str, float | int]:
    target = (
        _numeric(
            frame,
            TARGET_COLUMN,
        )
        <= TARGET_THRESHOLD
    ).astype(float)

    working = pd.DataFrame(
        {
            "target": target,
            "score": scores,
            "return": _numeric(
                frame,
                TARGET_COLUMN,
            ),
        }
    ).dropna()

    if len(working) < 2:
        return {
            "n": 0,
            "down_rate": float("nan"),
            "auc": float("nan"),
            "brier": float("nan"),
            "log_loss": float("nan"),
        }

    if working["target"].nunique() < 2:
        auc = float("nan")
    else:
        auc = float(
            roc_auc_score(
                working["target"],
                working["score"],
            )
        )

    clipped_scores = working["score"].clip(
        1e-6,
        1.0 - 1e-6,
    )

    return {
        "n": int(len(working)),
        "down_rate": float(
            working["target"].mean()
        ),
        "auc": auc,
        "brier": float(
            brier_score_loss(
                working["target"],
                working["score"],
            )
        ),
        "log_loss": float(
            log_loss(
                working["target"],
                clipped_scores,
                labels=[0.0, 1.0],
            )
        ),
    }


def _tail_analysis(
    frame: pd.DataFrame,
    scores: pd.Series,
) -> dict[str, float | int]:
    working = pd.DataFrame(
        {
            "score": scores,
            "return": _numeric(
                frame,
                TARGET_COLUMN,
            ),
        }
    ).dropna()

    if len(working) < 10:
        return {
            "high_n": 0,
            "low_n": 0,
            "high_down_rate": float("nan"),
            "low_down_rate": float("nan"),
            "down_rate_spread": float("nan"),
            "high_mean_return": float("nan"),
            "low_mean_return": float("nan"),
            "mean_return_spread": float("nan"),
        }

    high_threshold = working[
        "score"
    ].quantile(0.80)

    low_threshold = working[
        "score"
    ].quantile(0.20)

    high = working[
        working["score"] >= high_threshold
    ]

    low = working[
        working["score"] <= low_threshold
    ]

    high_down_rate = float(
        (
            high["return"]
            <= TARGET_THRESHOLD
        ).mean()
    )

    low_down_rate = float(
        (
            low["return"]
            <= TARGET_THRESHOLD
        ).mean()
    )

    high_mean_return = float(
        high["return"].mean()
    )

    low_mean_return = float(
        low["return"].mean()
    )

    return {
        "high_n": int(len(high)),
        "low_n": int(len(low)),
        "high_down_rate": high_down_rate,
        "low_down_rate": low_down_rate,
        "down_rate_spread": (
            high_down_rate
            - low_down_rate
        ),
        "high_mean_return": high_mean_return,
        "low_mean_return": low_mean_return,
        "mean_return_spread": (
            high_mean_return
            - low_mean_return
        ),
    }


def _prepare_event_scores(
    context,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
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
        event_features,
        validation_auc,
    ) = _select_event_model(
        train,
        validation,
    )

    train_event_model = _fit_event_model(
        train,
        event_features,
    )

    if train_event_model is None:
        raise RuntimeError(
            "Could not fit event-risk model "
            "on train data."
        )

    train["event_score"] = (
        _predict_event_score(
            train_event_model,
            train,
            event_features,
        )
    )

    validation["event_score"] = (
        _predict_event_score(
            train_event_model,
            validation,
            event_features,
        )
    )

    pretest_event_model = (
        _fit_selected_model_on_pretest(
            pretest,
            event_features,
        )
    )

    pretest["event_score"] = (
        _predict_event_score(
            pretest_event_model,
            pretest,
            event_features,
        )
    )

    test["event_score"] = (
        _predict_event_score(
            pretest_event_model,
            test,
            event_features,
        )
    )

    return (
        train,
        validation,
        pretest,
        test,
        model_name,
        event_features,
        validation_auc,
    )


def run_conditional_model_comparison(
    context,
) -> ExperimentResult:
    (
        train,
        validation,
        pretest,
        test,
        event_model_name,
        event_features,
        event_validation_auc,
    ) = _prepare_event_scores(
        context
    )

    required = {
        SI_COLUMN,
        PRIOR_RETURN_COLUMN,
        TARGET_COLUMN,
    }

    missing = [
        column
        for column in required
        if column not in train.columns
        or column not in validation.columns
        or column not in pretest.columns
        or column not in test.columns
    ]

    if missing:
        raise KeyError(
            "Conditional model comparison saknar "
            "kolumner: "
            + ", ".join(missing)
        )

    rows: list[dict] = []

    for model_name, features in MODEL_FEATURES.items():
        train_model = _fit_model(
            train,
            features,
        )

        validation_scores = _predict(
            train_model,
            validation,
            features,
        )

        validation_metrics = (
            _classification_metrics(
                validation,
                validation_scores,
            )
        )

        pretest_model = _fit_model(
            pretest,
            features,
        )

        test_scores = _predict(
            pretest_model,
            test,
            features,
        )

        test_metrics = _classification_metrics(
            test,
            test_scores,
        )

        tail_metrics = _tail_analysis(
            test,
            test_scores,
        )

        rows.append(
            {
                "model": model_name,
                "features": ",".join(
                    features
                ),
                "validation_n": (
                    validation_metrics["n"]
                ),
                "validation_auc": (
                    validation_metrics["auc"]
                ),
                "validation_brier": (
                    validation_metrics["brier"]
                ),
                "test_n": (
                    test_metrics["n"]
                ),
                "test_down_rate": (
                    test_metrics["down_rate"]
                ),
                "test_auc": (
                    test_metrics["auc"]
                ),
                "test_brier": (
                    test_metrics["brier"]
                ),
                "test_log_loss": (
                    test_metrics["log_loss"]
                ),
                "test_high_risk_n": (
                    tail_metrics["high_n"]
                ),
                "test_low_risk_n": (
                    tail_metrics["low_n"]
                ),
                "test_high_risk_down_rate": (
                    tail_metrics[
                        "high_down_rate"
                    ]
                ),
                "test_low_risk_down_rate": (
                    tail_metrics[
                        "low_down_rate"
                    ]
                ),
                "test_down_rate_spread": (
                    tail_metrics[
                        "down_rate_spread"
                    ]
                ),
                "test_high_risk_mean_return": (
                    tail_metrics[
                        "high_mean_return"
                    ]
                ),
                "test_low_risk_mean_return": (
                    tail_metrics[
                        "low_mean_return"
                    ]
                ),
                "test_mean_return_spread": (
                    tail_metrics[
                        "mean_return_spread"
                    ]
                ),
            }
        )

    result = ExperimentResult(
        name="si_conditional_model_comparison",
        description=(
            "Jämför fyra walk-forward-modeller "
            "för 5d-nedgång: SI-förändring, "
            "event-risk, kombinationen och "
            "kombinationen med tidigare avkastning."
        ),
    )

    result.add_metric(
        "target",
        TARGET_COLUMN,
    )

    result.add_metric(
        "target_threshold",
        TARGET_THRESHOLD,
    )

    result.add_metric(
        "event_model",
        event_model_name,
    )

    result.add_metric(
        "event_features",
        list(event_features),
    )

    result.add_metric(
        "event_validation_auc",
        event_validation_auc,
    )

    result.add_table(
        "model_comparison",
        pd.DataFrame(rows),
    )

    return result
