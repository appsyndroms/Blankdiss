from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .event_risk_grid import (
    LOCKED_DOWNSIDE_TARGET,
    LOCKED_RISK_CUTOFF,
    LOCKED_SI_CHANGE_CUTOFF,
    RANDOM_STATE,
    _fit_event_model,
    _numeric,
    _predict_event_score,
    _select_event_model,
    _tail_threshold,
)
BOOTSTRAP_ITERATIONS = 2_000
LOCKED_HORIZON = 1
TIMING_LAGS = (0, 1, 2)
SECTOR_COLUMN_CANDIDATES = (
    "sector",
    "sector_name",
    "gics_sector",
    "industry_sector",
    "sector_name_en",
)
@dataclass(frozen=True)
class MechanismEventModel:
    name: str
    features: tuple[str, ...]
    validation_auc: float
    model: Pipeline
def _return_column(horizon: int) -> str:
    return f"forward_return_{horizon}d"
def _prepare_frame(
    frame: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    frame = frame.copy()
    if "snapshot_date" not in frame.columns:
        raise KeyError(
            "Mechanism diagnostic requires 'snapshot_date'."
        )
    if "short_interest_pct" not in frame.columns:
        raise KeyError(
            "Mechanism diagnostic requires "
            "'short_interest_pct'."
        )
    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )
    frame["short_interest_pct"] = _numeric(
        frame,
        "short_interest_pct",
    )
    if "yahoo_symbol" in frame.columns:
        group_column = "yahoo_symbol"
    elif "security_key" in frame.columns:
        group_column = "security_key"
    else:
        raise KeyError(
            "Mechanism diagnostic requires either "
            "'yahoo_symbol' or 'security_key'."
        )
    frame = frame.sort_values(
        [
            group_column,
            "snapshot_date",
        ]
    ).copy()
    previous = (
        frame
        .groupby(
            group_column,
            sort=False,
        )["short_interest_pct"]
        .shift(1)
    )
    frame["short_interest_pct_change"] = (
        frame["short_interest_pct"] - previous
    )
    for lag in TIMING_LAGS:
        if lag == 0:
            frame["si_change_lag_0"] = (
                frame["short_interest_pct_change"]
            )
        else:
            frame[f"si_change_lag_{lag}"] = (
                frame
                .groupby(
                    group_column,
                    sort=False,
                )["short_interest_pct_change"]
                .shift(lag)
            )
    return_column = _return_column(horizon)
    if return_column not in frame.columns:
        raise KeyError(
            f"Missing required return column: "
            f"{return_column}"
        )
    frame[return_column] = _numeric(
        frame,
        return_column,
    )
    return frame
def _bootstrap_interaction(
    risk_low: pd.Series,
    risk_high: pd.Series,
    outside_low: pd.Series,
    outside_high: pd.Series,
) -> tuple[float, float, float]:
    groups = [
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
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
        )
    (
        risk_low_values,
        risk_high_values,
        outside_low_values,
        outside_high_values,
    ) = groups
    observed = (
        risk_high_values.mean()
        - risk_low_values.mean()
        - outside_high_values.mean()
        + outside_low_values.mean()
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
        risk_low_sample = rng.choice(
            risk_low_values,
            size=len(risk_low_values),
            replace=True,
        )
        risk_high_sample = rng.choice(
            risk_high_values,
            size=len(risk_high_values),
            replace=True,
        )
        outside_low_sample = rng.choice(
            outside_low_values,
            size=len(outside_low_values),
            replace=True,
        )
        outside_high_sample = rng.choice(
            outside_high_values,
            size=len(outside_high_values),
            replace=True,
        )
        deltas[index] = (
            risk_high_sample.mean()
            - risk_low_sample.mean()
            - outside_high_sample.mean()
            + outside_low_sample.mean()
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
    )
def _find_sector_column(
    frame: pd.DataFrame,
) -> str | None:
    for column in SECTOR_COLUMN_CANDIDATES:
        if column in frame.columns:
            return column
    return None
def _build_control_model() -> Pipeline:
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
def _safe_auc(
    y_true: pd.Series,
    scores: pd.Series,
) -> float:
    values = pd.DataFrame(
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
    if len(values) < 2:
        return float("nan")
    if values["y"].nunique() < 2:
        return float("nan")
    if values["score"].nunique() < 2:
        return 0.5
    return float(
        roc_auc_score(
            values["y"],
            values["score"],
        )
    )
def _fit_control_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    horizon: int,
    event_threshold: float,
) -> tuple[MechanismEventModel, Pipeline] | None:
    event_model = _select_event_model(
        train,
        validation,
        horizon,
        event_threshold,
    )
    features = [
        "event_score",
        "short_interest_pct_change",
        "price_volatility_20d",
    ]
    train_values = train.copy()
    validation_values = validation.copy()
    base_event_model = _fit_event_model(
        train_values,
        event_model.features,
        horizon,
        event_threshold,
    )
    if base_event_model is None:
        return None
    train_values["event_score"] = (
        _predict_event_score(
            base_event_model,
            train_values,
            event_model.features,
        )
    )
    validation_values["event_score"] = (
        _predict_event_score(
            base_event_model,
            validation_values,
            event_model.features,
        )
    )
    return_column = _return_column(horizon)
    train_values["target"] = (
        _numeric(
            train_values,
            return_column,
        )
        <= LOCKED_DOWNSIDE_TARGET
    ).astype(float)
    validation_values["target"] = (
        _numeric(
            validation_values,
            return_column,
        )
        <= LOCKED_DOWNSIDE_TARGET
    ).astype(float)
    train_values = train_values[
        features + ["target"]
    ].dropna()
    validation_values = validation_values[
        features + ["target"]
    ].dropna()
    if len(train_values) < 100:
        return None
    if len(validation_values) < 100:
        return None
    if train_values["target"].nunique() < 2:
        return None
    if validation_values["target"].nunique() < 2:
        return None
    model = _build_control_model()
    model.fit(
        train_values[features],
        train_values["target"],
    )
    validation_scores = model.predict_proba(
        validation_values[features]
    )[:, 1]
    auc = _safe_auc(
        validation_values["target"],
        pd.Series(
            validation_scores,
            index=validation_values.index,
        ),
    )
    if not np.isfinite(auc):
        return None
    return (
        MechanismEventModel(
            name=event_model.name,
            features=event_model.features,
            validation_auc=event_model.validation_auc,
            model=model,
        ),
        base_event_model,
    )
def _calculate_locked_groups(
    frame: pd.DataFrame,
    risk_threshold: float,
    si_column: str,
) -> dict[str, pd.Series]:
    frame = frame.copy()
    frame["high_event_risk"] = (
        frame["event_score"]
        >= risk_threshold
    )
    positive_si = _numeric(
        frame,
        si_column,
    )
    frame["high_si"] = (
        positive_si
        >= frame.attrs["si_threshold"]
    )
    return {
        "risk_high_si": frame[
            frame["high_event_risk"]
            & frame["high_si"]
        ]["down"],
        "risk_low_si": frame[
            frame["high_event_risk"]
            & ~frame["high_si"]
        ]["down"],
        "outside_high_si": frame[
            ~frame["high_event_risk"]
            & frame["high_si"]
        ]["down"],
        "outside_low_si": frame[
            ~frame["high_event_risk"]
            & ~frame["high_si"]
        ]["down"],
    }
def _timing_analysis(
    test: pd.DataFrame,
    risk_threshold: float,
    si_threshold: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for lag in TIMING_LAGS:
        column = f"si_change_lag_{lag}"
        if column not in test.columns:
            continue
        local = test.copy()
        local["high_event_risk"] = (
            local["event_score"]
            >= risk_threshold
        )
        local["high_si"] = (
            _numeric(
                local,
                column,
            )
            >= si_threshold
        )
        target = _numeric(
            local,
            _return_column(
                LOCKED_HORIZON
            ),
        ) <= LOCKED_DOWNSIDE_TARGET
        local["down"] = (
            target.astype(float)
        )
        local = local[
            target.notna()
        ].copy()
        groups = {
            "risk_high_si": local[
                local["high_event_risk"]
                & local["high_si"]
            ]["down"],
            "risk_low_si": local[
                local["high_event_risk"]
                & ~local["high_si"]
            ]["down"],
            "outside_high_si": local[
                ~local["high_event_risk"]
                & local["high_si"]
            ]["down"],
            "outside_low_si": local[
                ~local["high_event_risk"]
                & ~local["high_si"]
            ]["down"],
        }
        if any(
            len(group) == 0
            for group in groups.values()
        ):
            continue
        interaction, ci_low, ci_high = (
            _bootstrap_interaction(
                groups["risk_low_si"],
                groups["risk_high_si"],
                groups["outside_low_si"],
                groups["outside_high_si"],
            )
        )
        rows.append(
            {
                "analysis": "si_timing",
                "lag": lag,
                "lag_description": (
                    "current"
                    if lag == 0
                    else f"{lag}_prior_si_observation"
                ),
                "risk_high_si_n": len(
                    groups["risk_high_si"]
                ),
                "risk_low_si_n": len(
                    groups["risk_low_si"]
                ),
                "outside_high_si_n": len(
                    groups["outside_high_si"]
                ),
                "outside_low_si_n": len(
                    groups["outside_low_si"]
                ),
                "risk_si_effect": float(
                    groups["risk_high_si"].mean()
                    - groups["risk_low_si"].mean()
                ),
                "outside_si_effect": float(
                    groups["outside_high_si"].mean()
                    - groups["outside_low_si"].mean()
                ),
                "interaction": interaction,
                "interaction_ci_low": ci_low,
                "interaction_ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)
def _controlled_model_analysis(
    context,
    horizon: int,
    event_threshold: float,
) -> pd.DataFrame:
    train = context.train.copy()
    validation = context.validation.copy()
    pretest = context.pretest.copy()
    test = context.test.copy()
    for frame in (
        train,
        validation,
        pretest,
        test,
    ):
        frame["snapshot_date"] = pd.to_datetime(
            frame["snapshot_date"],
            errors="coerce",
        )
    train = _prepare_frame(
        train,
        horizon,
    )
    validation = _prepare_frame(
        validation,
        horizon,
    )
    pretest = _prepare_frame(
        pretest,
        horizon,
    )
    test = _prepare_frame(
        test,
        horizon,
    )
    event_model = _select_event_model(
        train,
        validation,
        horizon,
        event_threshold,
    )
    pretest_event_model = _fit_event_model(
        pretest,
        event_model.features,
        horizon,
        event_threshold,
    )
    if pretest_event_model is None:
        return pd.DataFrame()
    pretest["event_score"] = (
        _predict_event_score(
            pretest_event_model,
            pretest,
            event_model.features,
        )
    )
    test["event_score"] = (
        _predict_event_score(
            pretest_event_model,
            test,
            event_model.features,
        )
    )
    control_features = [
        "event_score",
        "short_interest_pct_change",
        "price_volatility_20d",
    ]
    return_column = _return_column(
        horizon
    )
    train["event_score"] = (
        _predict_event_score(
            _fit_event_model(
                train,
                event_model.features,
                horizon,
                event_threshold,
            ),
            train,
            event_model.features,
        )
    )
    train["target"] = (
        _numeric(
            train,
            return_column,
        )
        <= LOCKED_DOWNSIDE_TARGET
    ).astype(float)
    pretest["target"] = (
        _numeric(
            pretest,
            return_column,
        )
        <= LOCKED_DOWNSIDE_TARGET
    ).astype(float)
    test["target"] = (
        _numeric(
            test,
            return_column,
        )
        <= LOCKED_DOWNSIDE_TARGET
    ).astype(float)
    train_values = train[
        control_features + ["target"]
    ].dropna()
    pretest_values = pretest[
        control_features + ["target"]
    ].dropna()
    test_values = test[
        control_features + ["target"]
    ].dropna()
    if (
        len(train_values) < 100
        or len(pretest_values) < 100
        or len(test_values) < 100
    ):
        return pd.DataFrame()
    model = _build_control_model()
    model.fit(
        train_values[control_features],
        train_values["target"],
    )
    pretest_auc = _safe_auc(
        pretest_values["target"],
        pd.Series(
            model.predict_proba(
                pretest_values[control_features]
            )[:, 1],
            index=pretest_values.index,
        ),
    )
    model.fit(
        pretest_values[control_features],
        pretest_values["target"],
    )
    test_scores = model.predict_proba(
        test_values[control_features]
    )[:, 1]
    test_auc = _safe_auc(
        test_values["target"],
        pd.Series(
            test_scores,
            index=test_values.index,
        ),
    )
    coefficients = dict(
        zip(
            control_features,
            model.named_steps[
                "model"
            ].coef_[0],
        )
    )
    return pd.DataFrame(
        [
            {
                "analysis": "volatility_control",
                "horizon_days": horizon,
                "event_threshold": event_threshold,
                "event_model": event_model.name,
                "event_validation_auc": (
                    event_model.validation_auc
                ),
                "pretest_auc": pretest_auc,
                "test_auc": test_auc,
                "coefficient_event_score": (
                    coefficients[
                        "event_score"
                    ]
                ),
                "coefficient_si_change": (
                    coefficients[
                        "short_interest_pct_change"
                    ]
                ),
                "coefficient_volatility_20d": (
                    coefficients[
                        "price_volatility_20d"
                    ]
                ),
                "n_train": len(
                    train_values
                ),
                "n_pretest": len(
                    pretest_values
                ),
                "n_test": len(
                    test_values
                ),
            }
        ]
    )
def _sector_relative_analysis(
    test: pd.DataFrame,
    sector_column: str | None,
) -> pd.DataFrame:
    if sector_column is None:
        return pd.DataFrame(
            [
                {
                    "analysis": "sector_relative",
                    "status": "unavailable",
                    "reason": (
                        "No known sector column "
                        "was present in the feature dataset."
                    ),
                }
            ]
        )
    local = test.copy()
    local[sector_column] = (
        local[sector_column]
        .astype("string")
        .str.strip()
    )
    local["forward_return"] = _numeric(
        local,
        _return_column(
            LOCKED_HORIZON
        ),
    )
    local = local.dropna(
        subset=[
            sector_column,
            "snapshot_date",
            "forward_return",
        ]
    )
    if local.empty:
        return pd.DataFrame(
            [
                {
                    "analysis": "sector_relative",
                    "status": "unavailable",
                    "reason": (
                        "Sector column existed but "
                        "contained no usable observations."
                    ),
                }
            ]
        )
    sector_median = (
        local
        .groupby(
            [
                "snapshot_date",
                sector_column,
            ]
        )["forward_return"]
        .transform("median")
    )
    market_median = (
        local
        .groupby(
            "snapshot_date"
        )["forward_return"]
        .transform("median")
    )
    local[
        "sector_relative_return"
    ] = (
        local["forward_return"]
        - sector_median
    )
    local[
        "market_relative_return"
    ] = (
        local["forward_return"]
        - market_median
    )
    local[
        "sector_relative_down"
    ] = (
        local["sector_relative_return"]
        <= LOCKED_DOWNSIDE_TARGET
    )
    local[
        "market_relative_down"
    ] = (
        local["market_relative_return"]
        <= LOCKED_DOWNSIDE_TARGET
    )
    rows = []
    for target_column in (
        "sector_relative_down",
        "market_relative_down",
    ):
        rows.append(
            {
                "analysis": "sector_relative",
                "status": "completed",
                "target": target_column,
                "sector_column": sector_column,
                "n": len(local),
                "down_rate": float(
                    local[target_column].mean()
                ),
                "median_relative_return": float(
                    local[
                        (
                            "sector_relative_return"
                            if target_column
                            == "sector_relative_down"
                            else "market_relative_return"
                        )
                    ].median()
                ),
            }
        )
    return pd.DataFrame(rows)
def run_si_event_risk_mechanism(
    context,
) -> object:
    from .base import ExperimentResult
    horizon = LOCKED_HORIZON
    event_threshold = 0.07
    train = _prepare_frame(
        context.train,
        horizon,
    )
    validation = _prepare_frame(
        context.validation,
        horizon,
    )
    pretest = _prepare_frame(
        context.pretest,
        horizon,
    )
    test = _prepare_frame(
        context.test,
        horizon,
    )
    event_model = _select_event_model(
        train,
        validation,
        horizon,
        event_threshold,
    )
    pretest_model = _fit_event_model(
        pretest,
        event_model.features,
        horizon,
        event_threshold,
    )
    if pretest_model is None:
        raise RuntimeError(
            "Could not fit pretest event-risk model."
        )
    pretest["event_score"] = (
        _predict_event_score(
            pretest_model,
            pretest,
            event_model.features,
        )
    )
    test["event_score"] = (
        _predict_event_score(
            pretest_model,
            test,
            event_model.features,
        )
    )
    positive_si = _numeric(
        pretest,
        "short_interest_pct_change",
    )
    positive_si = positive_si[
        positive_si > 0
    ].dropna()
    si_threshold = float(
        positive_si.quantile(
            1.0 - LOCKED_SI_CHANGE_CUTOFF
        )
    )
    risk_threshold = _tail_threshold(
        pretest["event_score"],
        LOCKED_RISK_CUTOFF,
    )
    timing = _timing_analysis(
        test=test,
        risk_threshold=risk_threshold,
        si_threshold=si_threshold,
    )
    control = _controlled_model_analysis(
        context=context,
        horizon=horizon,
        event_threshold=event_threshold,
    )
    sector_column = _find_sector_column(
        test
    )
    sector = _sector_relative_analysis(
        test=test,
        sector_column=sector_column,
    )
    result = ExperimentResult(
        name="si_event_risk_mechanism",
        description=(
            "Låst mekanismtest av SI/event-risk-"
            "hypotesen med timing, kontinuerlig "
            "volatilitetskontroll och sektor-/"
            "marknadsrelativt utfall."
        ),
    )
    result.add_metric(
        "analysis_type",
        "locked_mechanism_follow_up",
    )
    result.add_metric(
        "horizon_days",
        horizon,
    )
    result.add_metric(
        "event_threshold",
        event_threshold,
    )
    result.add_metric(
        "risk_cutoff",
        LOCKED_RISK_CUTOFF,
    )
    result.add_metric(
        "si_change_cutoff",
        LOCKED_SI_CHANGE_CUTOFF,
    )
    result.add_metric(
        "downside_target",
        LOCKED_DOWNSIDE_TARGET,
    )
    result.add_metric(
        "event_model",
        event_model.name,
    )
    result.add_metric(
        "event_features",
        list(event_model.features),
    )
    result.add_metric(
        "event_validation_auc",
        event_model.validation_auc,
    )
    result.add_metric(
        "risk_threshold",
        risk_threshold,
    )
    result.add_metric(
        "si_change_threshold",
        si_threshold,
    )
    result.add_metric(
        "selection_warning",
        (
            "Den låsta hypotesen valdes efter "
            "tidigare discovery-resultat. "
            "Detta är därför fortfarande inte "
            "en oberoende ny OOS-bekräftelse."
        ),
    )
    result.add_metric(
        "sector_column",
        sector_column,
    )
    result.add_table(
        "timing",
        timing,
    )
    result.add_table(
        "volatility_control",
        control,
    )
    result.add_table(
        "sector_relative",
        sector,
    )
    return result
