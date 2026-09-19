from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from .base import ExperimentResult


RANDOM_STATE = 42
BOOTSTRAP_ITERATIONS = 2_000

EVENT_THRESHOLD = 0.10
EVENT_TAIL = 0.05

VOLATILITY_TAIL = 0.20
SHORT_INTEREST_TAIL = 0.20

TARGETS = (
    "down_10pct_5d",
    "down_7pct_5d",
    "down_5pct_5d",
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
class InteractionResult:
    window: str
    target: str

    event_risk_tail_n: int

    low_vol_low_si_n: int
    low_vol_high_si_n: int
    high_vol_low_si_n: int
    high_vol_high_si_n: int

    low_vol_low_si_rate: float
    low_vol_high_si_rate: float
    high_vol_low_si_rate: float
    high_vol_high_si_rate: float

    low_vol_si_effect: float
    high_vol_si_effect: float

    interaction: float
    ci_low: float
    ci_high: float
    p_positive: float


def _numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    values = frame[column]

    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]

    return pd.to_numeric(
        values,
        errors="coerce",
    )


def _safe_auc(
    y_true,
    scores,
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
        return np.nan

    if frame["y"].nunique() < 2:
        return np.nan

    if frame["score"].nunique() < 2:
        return 0.5

    return float(
        roc_auc_score(
            frame["y"],
            frame["score"],
        )
    )


def _safe_rate(
    values: pd.Series,
) -> float:
    if len(values) == 0:
        return np.nan

    return float(
        values.mean()
    )


def _bootstrap_interaction(
    low_vol_low_si: pd.Series,
    low_vol_high_si: pd.Series,
    high_vol_low_si: pd.Series,
    high_vol_high_si: pd.Series,
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
        ).dropna().to_numpy()
        for values in (
            low_vol_low_si,
            low_vol_high_si,
            high_vol_low_si,
            high_vol_high_si,
        )
    ]

    if any(
        len(values) == 0
        for values in groups
    ):
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    (
        low_low,
        low_high,
        high_low,
        high_high,
    ) = groups

    observed = (
        high_high.mean()
        - high_low.mean()
        - low_high.mean()
        + low_low.mean()
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    interactions = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=float,
    )

    for index in range(
        BOOTSTRAP_ITERATIONS
    ):
        low_low_sample = rng.choice(
            low_low,
            size=len(low_low),
            replace=True,
        )

        low_high_sample = rng.choice(
            low_high,
            size=len(low_high),
            replace=True,
        )

        high_low_sample = rng.choice(
            high_low,
            size=len(high_low),
            replace=True,
        )

        high_high_sample = rng.choice(
            high_high,
            size=len(high_high),
            replace=True,
        )

        interactions[index] = (
            high_high_sample.mean()
            - high_low_sample.mean()
            - low_high_sample.mean()
            + low_low_sample.mean()
        )

    return (
        float(observed),
        float(
            np.quantile(
                interactions,
                0.025,
            )
        ),
        float(
            np.quantile(
                interactions,
                0.975,
            )
        ),
        float(
            (interactions > 0).mean()
        ),
    )


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
    train: pd.DataFrame,
    features: list[str],
) -> Pipeline | None:
    frame = train[
        features + ["event"]
    ].copy()

    for column in features:
        frame[column] = _numeric(
            frame,
            column,
        )

    frame["event"] = _numeric(
        frame,
        "event",
    )

    frame = frame.dropna()

    if len(frame) < 100:
        return None

    if frame["event"].nunique() < 2:
        return None

    model = _build_event_model()

    model.fit(
        frame[features],
        frame["event"],
    )

    return model


def _predict_event_score(
    model: Pipeline,
    data: pd.DataFrame,
    features: list[str],
) -> pd.Series:
    result = pd.Series(
        np.nan,
        index=data.index,
        dtype=float,
    )

    frame = data[
        features
    ].copy()

    for column in features:
        frame[column] = _numeric(
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


def _select_event_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[
    str,
    list[str],
    float,
]:
    best_name = ""
    best_features: list[str] = []
    best_auc = -np.inf

    for (
        name,
        candidates,
    ) in EVENT_FEATURE_SETS.items():
        features = [
            column
            for column in candidates
            if (
                column in train.columns
                and column in validation.columns
            )
        ]

        if not features:
            continue

        model = _fit_event_model(
            train,
            features,
        )

        if model is None:
            continue

        scores = _predict_event_score(
            model,
            validation,
            features,
        )

        auc = _safe_auc(
            validation["event"],
            scores,
        )

        print(
            f"    {name}: "
            f"validation AUC="
            f"{auc:.6f}"
        )

        if (
            np.isfinite(auc)
            and auc > best_auc
        ):
            best_name = name
            best_features = features
            best_auc = auc

    if not best_features:
        raise RuntimeError(
            "Could not select event-risk model."
        )

    return (
        best_name,
        best_features,
        float(best_auc),
    )


def _percentile_threshold(
    values: pd.Series,
    upper_tail: float,
) -> float:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(
        values.quantile(
            1.0 - upper_tail
        )
    )


def _interaction_analysis(
    frame: pd.DataFrame,
    target: str,
    window_name: str,
) -> InteractionResult | None:
    frame = frame.dropna(
        subset=[
            "volatility_group",
            "short_interest_group",
            target,
        ]
    ).copy()

    if frame.empty:
        return None

    cells = {
        "low_low": frame.loc[
            (
                frame["volatility_group"]
                == "LOW"
            )
            & (
                frame["short_interest_group"]
                == "LOW"
            ),
            target,
        ],
        "low_high": frame.loc[
            (
                frame["volatility_group"]
                == "LOW"
            )
            & (
                frame["short_interest_group"]
                == "HIGH"
            ),
            target,
        ],
        "high_low": frame.loc[
            (
                frame["volatility_group"]
                == "HIGH"
            )
            & (
                frame["short_interest_group"]
                == "LOW"
            ),
            target,
        ],
        "high_high": frame.loc[
            (
                frame["volatility_group"]
                == "HIGH"
            )
            & (
                frame["short_interest_group"]
                == "HIGH"
            ),
            target,
        ],
    }

    if any(
        len(values) == 0
        for values in cells.values()
    ):
        print(
            f"  {target}: "
            "insufficient cells for interaction."
        )
        return None

    low_low_rate = _safe_rate(
        cells["low_low"]
    )
    low_high_rate = _safe_rate(
        cells["low_high"]
    )
    high_low_rate = _safe_rate(
        cells["high_low"]
    )
    high_high_rate = _safe_rate(
        cells["high_high"]
    )

    low_vol_si_effect = (
        low_high_rate
        - low_low_rate
    )

    high_vol_si_effect = (
        high_high_rate
        - high_low_rate
    )

    (
        interaction,
        ci_low,
        ci_high,
        p_positive,
    ) = _bootstrap_interaction(
        cells["low_low"],
        cells["low_high"],
        cells["high_low"],
        cells["high_high"],
    )

    return InteractionResult(
        window=window_name,
        target=target,
        event_risk_tail_n=len(frame),
        low_vol_low_si_n=len(
            cells["low_low"]
        ),
        low_vol_high_si_n=len(
            cells["low_high"]
        ),
        high_vol_low_si_n=len(
            cells["high_low"]
        ),
        high_vol_high_si_n=len(
            cells["high_high"]
        ),
        low_vol_low_si_rate=low_low_rate,
        low_vol_high_si_rate=low_high_rate,
        high_vol_low_si_rate=high_low_rate,
        high_vol_high_si_rate=high_high_rate,
        low_vol_si_effect=low_vol_si_effect,
        high_vol_si_effect=high_vol_si_effect,
        interaction=interaction,
        ci_low=ci_low,
        ci_high=ci_high,
        p_positive=p_positive,
    )


def run_volatility_si_level_interaction(
    context,
) -> ExperimentResult:
    data = context.data.copy()

    required = [
        "snapshot_date",
        "security_key",
        "forward_return_5d",
        "short_interest_pct",
        "price_volatility_20d",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    data["snapshot_date"] = pd.to_datetime(
        data["snapshot_date"],
        errors="coerce",
    )

    for column in (
        "short_interest_pct",
        "price_volatility_20d",
        "forward_return_5d",
    ):
        data[column] = _numeric(
            data,
            column,
        )

    data = data.dropna(
        subset=[
            "snapshot_date",
            "security_key",
            "forward_return_5d",
        ]
    ).copy()

    data["event"] = (
        data["forward_return_5d"]
        .abs()
        >= EVENT_THRESHOLD
    ).astype(int)

    data["down_10pct_5d"] = (
        data["forward_return_5d"]
        <= -0.10
    ).astype(int)

    data["down_7pct_5d"] = (
        data["forward_return_5d"]
        <= -0.07
    ).astype(int)

    data["down_5pct_5d"] = (
        data["forward_return_5d"]
        <= -0.05
    ).astype(int)

    train = data.loc[
        data["snapshot_date"]
        <= pd.Timestamp(
            context.train_end
        )
    ].copy()

    validation = data.loc[
        (
            data["snapshot_date"]
            > pd.Timestamp(
                context.train_end
            )
        )
        & (
            data["snapshot_date"]
            <= pd.Timestamp(
                context.validation_end
            )
        )
    ].copy()

    test = data.loc[
        data["snapshot_date"]
        > pd.Timestamp(
            context.validation_end
        )
    ].copy()

    (
        event_model_name,
        event_features,
        validation_auc,
    ) = _select_event_model(
        train,
        validation,
    )

    print()
    print(
        f"Event-risk model: "
        f"{event_model_name}"
    )

    print(
        f"Validation AUC: "
        f"{validation_auc:.6f}"
    )

    pretest = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    final_model = _fit_event_model(
        pretest,
        event_features,
    )

    if final_model is None:
        raise RuntimeError(
            "Could not fit final event model."
        )

    pretest_scores = _predict_event_score(
        final_model,
        pretest,
        event_features,
    )

    event_threshold = (
        _percentile_threshold(
            pretest_scores,
            EVENT_TAIL,
        )
    )

    volatility_threshold = (
        _percentile_threshold(
            pretest[
                "price_volatility_20d"
            ],
            VOLATILITY_TAIL,
        )
    )

    short_interest_threshold = (
        _percentile_threshold(
            pretest[
                "short_interest_pct"
            ],
            SHORT_INTEREST_TAIL,
        )
    )

    print()
    print(
        "PRE-TEST THRESHOLDS"
    )
    print(
        f"  Event-risk top "
        f"{EVENT_TAIL:.0%}: "
        f"{event_threshold:.6f}"
    )
    print(
        f"  Volatility top "
        f"{VOLATILITY_TAIL:.0%}: "
        f"{volatility_threshold:.6f}"
    )
    print(
        f"  Short interest top "
        f"{SHORT_INTEREST_TAIL:.0%}: "
        f"{short_interest_threshold:.6f}"
    )

    test["event_score"] = (
        _predict_event_score(
            final_model,
            test,
            event_features,
        )
    )

    test = test.dropna(
        subset=[
            "event_score",
            "price_volatility_20d",
            "short_interest_pct",
        ]
    ).copy()

    event_tail = test.loc[
        test["event_score"]
        >= event_threshold
    ].copy()

    event_tail["volatility_group"] = np.where(
        event_tail[
            "price_volatility_20d"
        ]
        >= volatility_threshold,
        "HIGH",
        "LOW",
    )

    event_tail["short_interest_group"] = np.where(
        event_tail[
            "short_interest_pct"
        ]
        >= short_interest_threshold,
        "HIGH",
        "LOW",
    )

    rows = []

    for target in TARGETS:
        interaction = _interaction_analysis(
            event_tail,
            target,
            (
                f"{context.train_end}_"
                f"{context.validation_end}_"
                f"{context.test_end}"
            ),
        )

        if interaction is None:
            continue

        rows.append(
            {
                "window": interaction.window,
                "target": interaction.target,
                "event_risk_tail_n": (
                    interaction.event_risk_tail_n
                ),
                "low_vol_low_si_n": (
                    interaction.low_vol_low_si_n
                ),
                "low_vol_high_si_n": (
                    interaction.low_vol_high_si_n
                ),
                "high_vol_low_si_n": (
                    interaction.high_vol_low_si_n
                ),
                "high_vol_high_si_n": (
                    interaction.high_vol_high_si_n
                ),
                "low_vol_low_si_rate": (
                    interaction.low_vol_low_si_rate
                ),
                "low_vol_high_si_rate": (
                    interaction.low_vol_high_si_rate
                ),
                "high_vol_low_si_rate": (
                    interaction.high_vol_low_si_rate
                ),
                "high_vol_high_si_rate": (
                    interaction.high_vol_high_si_rate
                ),
                "low_vol_si_effect": (
                    interaction.low_vol_si_effect
                ),
                "high_vol_si_effect": (
                    interaction.high_vol_si_effect
                ),
                "interaction": (
                    interaction.interaction
                ),
                "ci_low": (
                    interaction.ci_low
                ),
                "ci_high": (
                    interaction.ci_high
                ),
                "p_positive": (
                    interaction.p_positive
                ),
            }
        )

    result = ExperimentResult(
        name="volatility_si_level_interaction",
        description=(
            "Testar om hög short interest "
            "tillför information inom "
            "extrem event-risk när volatiliteten "
            "redan är hög."
        ),
    )

    result.add_table(
        "interaction",
        pd.DataFrame(rows),
    )

    result.add_metadata(
        "event_model",
        event_model_name,
    )

    result.add_metadata(
        "event_features",
        event_features,
    )

    result.add_metadata(
        "validation_auc",
        validation_auc,
    )

    result.add_metadata(
        "event_tail",
        EVENT_TAIL,
    )

    result.add_metadata(
        "volatility_tail",
        VOLATILITY_TAIL,
    )

    result.add_metadata(
        "short_interest_tail",
        SHORT_INTEREST_TAIL,
    )

    result.add_metadata(
        "bootstrap_iterations",
        BOOTSTRAP_ITERATIONS,
    )

    return result
