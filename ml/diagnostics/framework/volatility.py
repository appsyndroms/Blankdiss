from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.config import RANDOM_STATE
from ml.dataset import get_feature_columns
from ml.models import build_models

from .base import ExperimentResult


BENCHMARK_TREES = 100

ECONOMIC_TARGET = "down_5pct_5d"

VOLATILITY_FEATURE = (
    "price_volatility_20d"
)

ECONOMIC_FRACTIONS = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
)

REGIMES = (
    "LOW",
    "MID",
    "HIGH",
)

STRATEGIES = (
    "fi_only",
    "fi_plus_volatility",
    "gate_validation",
    "gate_vol_mid_high",
)


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
    y = np.asarray(
        y_true,
        dtype=float,
    )

    score = np.asarray(
        scores,
        dtype=float,
    )

    valid = (
        np.isfinite(y)
        & np.isfinite(score)
    )

    y = y[valid]
    score = score[valid]

    if len(y) == 0:
        return np.nan

    if len(np.unique(y)) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            score,
        )
    )


def _make_target(
    frame: pd.DataFrame,
) -> pd.Series:
    return (
        _numeric(
            frame,
            "forward_return_5d",
        )
        <= -0.05
    ).astype(int)


def _build_models() -> dict[str, Any]:
    models = build_models(
        RANDOM_STATE,
        task="classification",
    )

    random_forest = models.get(
        "random_forest"
    )

    if random_forest is not None:
        random_forest.set_params(
            model__n_estimators=(
                BENCHMARK_TREES
            ),
            model__n_jobs=1,
        )

    return models


def _available_features(
    train: pd.DataFrame,
    requested: list[str],
) -> list[str]:
    result = []

    for column in requested:
        if column not in train.columns:
            continue

        if train[column].notna().any():
            result.append(column)

    return result


def _train_and_select(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    y_train: pd.Series,
    y_validation: pd.Series,
) -> dict[str, Any]:
    features = _available_features(
        train,
        feature_columns,
    )

    if not features:
        raise RuntimeError(
            "No requested features available "
            "in training data."
        )

    train_x = train[
        features
    ].copy()

    validation_x = validation[
        features
    ].copy()

    models = _build_models()

    trained = []

    for name, model in models.items():
        model.fit(
            train_x,
            y_train,
        )

        validation_prediction = (
            model.predict_proba(
                validation_x
            )[:, 1]
        )

        validation_auc = _safe_auc(
            y_validation,
            validation_prediction,
        )

        print(
            f"    {name}: "
            f"validation AUC="
            f"{validation_auc:.6f}"
        )

        trained.append(
            {
                "name": name,
                "model": model,
                "validation_auc": (
                    validation_auc
                ),
            }
        )

    if not trained:
        raise RuntimeError(
            "No models were trained."
        )

    trained.sort(
        key=lambda item: (
            item["validation_auc"]
            if np.isfinite(
                item["validation_auc"]
            )
            else -np.inf
        ),
        reverse=True,
    )

    selected = trained[0]

    return {
        "selected_model": selected[
            "name"
        ],
        "model": selected[
            "model"
        ],
        "validation_auc": selected[
            "validation_auc"
        ],
        "features": features,
        "all_models": trained,
    }


def _calculate_training_regime_boundaries(
    train: pd.DataFrame,
) -> tuple[float, float]:
    values = _numeric(
        train,
        VOLATILITY_FEATURE,
    ).dropna()

    if values.empty:
        raise RuntimeError(
            "No training volatility values available."
        )

    q1 = float(
        values.quantile(
            1.0 / 3.0
        )
    )

    q2 = float(
        values.quantile(
            2.0 / 3.0
        )
    )

    return q1, q2


def _assign_regime(
    volatility,
    q1: float,
    q2: float,
):
    if pd.isna(volatility):
        return None

    if volatility <= q1:
        return "LOW"

    if volatility <= q2:
        return "MID"

    return "HIGH"


def _calculate_top_metrics(
    frame: pd.DataFrame,
    score_column: str,
    fraction: float,
) -> dict[str, Any]:
    valid = frame.loc[
        frame[score_column].notna()
        & frame["target"].notna()
        & frame["target_return"].notna()
    ].copy()

    if valid.empty:
        return {
            "n": 0,
            "event_rate": np.nan,
            "lift": np.nan,
            "mean_return": np.nan,
            "median_return": np.nan,
        }

    n = max(
        1,
        int(
            np.ceil(
                len(valid)
                * fraction
            )
        ),
    )

    ranked = (
        valid
        .sort_values(
            score_column,
            ascending=False,
        )
        .head(n)
    )

    event_rate = float(
        ranked["target"].mean()
    )

    baseline = float(
        valid["target"].mean()
    )

    lift = (
        event_rate / baseline
        if baseline > 0
        else np.nan
    )

    return {
        "n": int(len(ranked)),
        "event_rate": event_rate,
        "lift": lift,
        "mean_return": float(
            ranked["target_return"].mean()
        ),
        "median_return": float(
            ranked["target_return"].median()
        ),
    }


def _build_output_frame(
    source: pd.DataFrame,
    prediction: np.ndarray,
) -> pd.DataFrame:
    result = source[
        [
            "snapshot_date",
            "security_key",
            "forward_return_5d",
        ]
    ].copy()

    result = result.rename(
        columns={
            "forward_return_5d":
                "target_return",
        }
    )

    result["target"] = _make_target(
        source
    )

    result["prediction"] = prediction

    return result


def _select_regime_gate(
    fi_validation: pd.DataFrame,
    vol_validation: pd.DataFrame,
    q1: float,
    q2: float,
) -> dict[str, str]:
    fi = fi_validation[
        [
            "snapshot_date",
            "security_key",
            "target_return",
            "target",
            VOLATILITY_FEATURE,
            "prediction",
        ]
    ].rename(
        columns={
            "prediction":
                "fi_prediction",
        }
    )

    vol = vol_validation[
        [
            "snapshot_date",
            "security_key",
            "target_return",
            "target",
            "prediction",
        ]
    ].rename(
        columns={
            "prediction":
                "vol_prediction",
        }
    )

    merged = fi.merge(
        vol,
        on=[
            "snapshot_date",
            "security_key",
        ],
        how="inner",
        suffixes=(
            "_fi",
            "_vol",
        ),
        validate="one_to_one",
    )

    merged["regime"] = (
        merged[
            VOLATILITY_FEATURE
        ]
        .apply(
            lambda value:
            _assign_regime(
                value,
                q1,
                q2,
            )
        )
    )

    gate: dict[str, str] = {}

    print()
    print(
        "Validation regime selection:"
    )

    for regime in REGIMES:
        subset = merged.loc[
            merged["regime"] == regime
        ]

        fi_auc = _safe_auc(
            subset["target_fi"],
            subset["fi_prediction"],
        )

        vol_auc = _safe_auc(
            subset["target_vol"],
            subset["vol_prediction"],
        )

        if (
            np.isfinite(fi_auc)
            and np.isfinite(vol_auc)
        ):
            selected = (
                "fi"
                if fi_auc >= vol_auc
                else "vol"
            )
        elif np.isfinite(fi_auc):
            selected = "fi"
        elif np.isfinite(vol_auc):
            selected = "vol"
        else:
            selected = "fi"

        gate[regime] = selected

        print(
            f"  {regime}: "
            f"rows={len(subset):,} "
            f"FI AUC={fi_auc:.6f} | "
            f"FI+VOL AUC={vol_auc:.6f} | "
            f"selected={selected}"
        )

    return gate


def _apply_strategy(
    fi_test: pd.DataFrame,
    vol_test: pd.DataFrame,
    strategy: str,
    gate: dict[str, str],
    q1: float,
    q2: float,
) -> pd.DataFrame:
    fi = fi_test[
        [
            "snapshot_date",
            "security_key",
            "target_return",
            "target",
            VOLATILITY_FEATURE,
            "prediction",
        ]
    ].rename(
        columns={
            "prediction":
                "fi_prediction",
        }
    )

    vol = vol_test[
        [
            "snapshot_date",
            "security_key",
            "target_return",
            "target",
            "prediction",
        ]
    ].rename(
        columns={
            "prediction":
                "vol_prediction",
        }
    )

    merged = fi.merge(
        vol,
        on=[
            "snapshot_date",
            "security_key",
        ],
        how="inner",
        suffixes=(
            "_fi",
            "_vol",
        ),
        validate="one_to_one",
    )

    target_difference = (
        merged["target_return_fi"]
        - merged["target_return_vol"]
    ).abs()

    if (
        target_difference
        > 1e-10
    ).any():
        raise ValueError(
            "Target return mismatch between "
            "FI and FI+VOL test rows."
        )

    merged["target"] = (
        merged["target_fi"]
    )

    merged["target_return"] = (
        merged["target_return_fi"]
    )

    merged["regime"] = (
        merged[
            VOLATILITY_FEATURE
        ]
        .apply(
            lambda value:
            _assign_regime(
                value,
                q1,
                q2,
            )
        )
    )

    if strategy == "fi_only":
        merged["score"] = (
            merged["fi_prediction"]
        )

        merged["chosen_model"] = "FI"

    elif strategy == "fi_plus_volatility":
        merged["score"] = (
            merged["vol_prediction"]
        )

        merged["chosen_model"] = (
            "FI+VOL"
        )

    elif strategy == "gate_validation":
        selected = (
            merged["regime"]
            .map(
                lambda regime:
                gate.get(
                    regime,
                    "fi",
                )
            )
        )

        merged["score"] = np.where(
            selected == "vol",
            merged["vol_prediction"],
            merged["fi_prediction"],
        )

        merged["chosen_model"] = np.where(
            selected == "vol",
            "FI+VOL",
            "FI",
        )

    elif strategy == "gate_vol_mid_high":
        selected = np.where(
            merged["regime"] == "LOW",
            "fi",
            "vol",
        )

        merged["score"] = np.where(
            selected == "vol",
            merged["vol_prediction"],
            merged["fi_prediction"],
        )

        merged["chosen_model"] = np.where(
            selected == "vol",
            "FI+VOL",
            "FI",
        )

    else:
        raise ValueError(
            f"Unknown strategy: "
            f"{strategy}"
        )

    return merged[
        [
            "snapshot_date",
            "security_key",
            "target_return",
            "target",
            VOLATILITY_FEATURE,
            "regime",
            "score",
            "chosen_model",
        ]
    ].copy()


def run_volatility_regime(
    context,
    *,
    volatility_column: str = (
        VOLATILITY_FEATURE
    ),
    strategies: tuple[str, ...] = (
        STRATEGIES
    ),
) -> ExperimentResult:
    if volatility_column != (
        VOLATILITY_FEATURE
    ):
        raise ValueError(
            "Historical volatility-regime "
            "diagnostic uses "
            f"'{VOLATILITY_FEATURE}'."
        )

    train = context.train.copy()
    validation = context.validation.copy()
    test = context.test.copy()

    required = [
        "snapshot_date",
        "security_key",
        "forward_return_5d",
        VOLATILITY_FEATURE,
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

    y_train = _make_target(
        train
    )

    y_validation = _make_target(
        validation
    )

    y_test = _make_target(
        test
    )

    # ------------------------------------------------------------------
    # Resolve the same feature families used by the historical
    # prepare_feature_set() calls.
    # ------------------------------------------------------------------

    fi_features = get_feature_columns(
        context.data,
        include_price_features=False,
        price_features=None,
    )

    vol_features = get_feature_columns(
        context.data,
        include_price_features=True,
        price_features={
            VOLATILITY_FEATURE,
        },
    )

    fi_features = _available_features(
        train,
        fi_features,
    )

    vol_features = _available_features(
        train,
        vol_features,
    )

    print()
    print(
        f"FI-only features: "
        f"{len(fi_features)}"
    )

    print(
        f"FI+volatility features: "
        f"{len(vol_features)}"
    )

    # ------------------------------------------------------------------
    # Regime boundaries are TRAIN-only.
    # ------------------------------------------------------------------

    q1, q2 = (
        _calculate_training_regime_boundaries(
            train
        )
    )

    print()
    print(
        "Training volatility boundaries:"
    )

    print(
        f"  LOW <= {q1:.6f}"
    )

    print(
        f"  MID <= {q2:.6f}"
    )

    print(
        f"  HIGH > {q2:.6f}"
    )

    # ------------------------------------------------------------------
    # FI model.
    # ------------------------------------------------------------------

    print()
    print(
        "Training FI-only models..."
    )

    fi_result = _train_and_select(
        train,
        validation,
        fi_features,
        y_train,
        y_validation,
    )

    fi_validation_prediction = (
        fi_result["model"]
        .predict_proba(
            validation[
                fi_result["features"]
            ]
        )[:, 1]
    )

    fi_test_prediction = (
        fi_result["model"]
        .predict_proba(
            test[
                fi_result["features"]
            ]
        )[:, 1]
    )

    fi_validation = (
        _build_output_frame(
            validation,
            fi_validation_prediction,
        )
    )

    fi_test = (
        _build_output_frame(
            test,
            fi_test_prediction,
        )
    )

    # Volatility is metadata for the gate.
    fi_validation[
        VOLATILITY_FEATURE
    ] = _numeric(
        validation,
        VOLATILITY_FEATURE,
    ).to_numpy()

    fi_test[
        VOLATILITY_FEATURE
    ] = _numeric(
        test,
        VOLATILITY_FEATURE,
    ).to_numpy()

    # ------------------------------------------------------------------
    # FI + volatility model.
    # ------------------------------------------------------------------

    print()
    print(
        "Training FI+volatility models..."
    )

    vol_result = _train_and_select(
        train,
        validation,
        vol_features,
        y_train,
        y_validation,
    )

    vol_validation_prediction = (
        vol_result["model"]
        .predict_proba(
            validation[
                vol_result["features"]
            ]
        )[:, 1]
    )

    vol_test_prediction = (
        vol_result["model"]
        .predict_proba(
            test[
                vol_result["features"]
            ]
        )[:, 1]
    )

    vol_validation = (
        _build_output_frame(
            validation,
            vol_validation_prediction,
        )
    )

    vol_test = (
        _build_output_frame(
            test,
            vol_test_prediction,
        )
    )

    # ------------------------------------------------------------------
    # Gate selection is VALIDATION-only.
    # ------------------------------------------------------------------

    gate = _select_regime_gate(
        fi_validation,
        vol_validation,
        q1,
        q2,
    )

    # ------------------------------------------------------------------
    # Apply frozen strategies to TEST.
    # ------------------------------------------------------------------

    strategy_rows = []
    strategy_tables: dict[
        str,
        pd.DataFrame,
    ] = {}

    for strategy in strategies:
        frame = _apply_strategy(
            fi_test,
            vol_test,
            strategy,
            gate,
            q1,
            q2,
        )

        strategy_tables[strategy] = (
            frame
        )

        auc = _safe_auc(
            frame["target"],
            frame["score"],
        )

        top_metrics = {}

        for fraction in (
            ECONOMIC_FRACTIONS
        ):
            metrics = (
                _calculate_top_metrics(
                    frame,
                    "score",
                    fraction,
                )
            )

            key = (
                f"top_"
                f"{fraction:g}"
            )

            top_metrics[
                key
            ] = metrics

        row = {
            "strategy": strategy,
            "test_rows": int(
                len(frame)
            ),
            "test_auc": auc,
        }

        for fraction in (
            ECONOMIC_FRACTIONS
        ):
            metrics = top_metrics[
                f"top_{fraction:g}"
            ]

            suffix = (
                str(fraction)
                .replace(
                    ".",
                    "_",
                )
            )

            row[
                f"top_{suffix}_n"
            ] = metrics["n"]

            row[
                f"top_{suffix}_event_rate"
            ] = metrics[
                "event_rate"
            ]

            row[
                f"top_{suffix}_lift"
            ] = metrics["lift"]

            row[
                f"top_{suffix}_mean_return"
            ] = metrics[
                "mean_return"
            ]

            row[
                f"top_{suffix}_median_return"
            ] = metrics[
                "median_return"
            ]

        strategy_rows.append(
            row
        )

    # ------------------------------------------------------------------
    # Regime-level gate information.
    # ------------------------------------------------------------------

    gate_rows = []

    for regime in REGIMES:
        subset = fi_validation.loc[
            fi_validation[
                VOLATILITY_FEATURE
            ].apply(
                lambda value:
                _assign_regime(
                    value,
                    q1,
                    q2,
                )
            )
            == regime
        ]

        fi_auc = _safe_auc(
            subset["target"],
            subset["prediction"],
        )

        vol_subset = vol_validation.loc[
            vol_validation[
                VOLATILITY_FEATURE
            ].apply(
                lambda value:
                _assign_regime(
                    value,
                    q1,
                    q2,
                )
            )
            == regime
        ]

        vol_auc = _safe_auc(
            vol_subset["target"],
            vol_subset["prediction"],
        )

        gate_rows.append(
            {
                "regime": regime,
                "rows": int(
                    len(subset)
                ),
                "fi_validation_auc": fi_auc,
                "fi_plus_vol_validation_auc": (
                    vol_auc
                ),
                "selected": gate.get(
                    regime,
                    "fi",
                ),
            }
        )

    # ------------------------------------------------------------------
    # Result.
    # ------------------------------------------------------------------

    result = ExperimentResult(
        name="volatility_regime",
        description=(
            "Testar om en "
            "volatility-regime-gate "
            "förbättrar OOS-rankning "
            "jämfört med fasta modeller."
        ),
    )

    result.add_table(
        "strategies",
        pd.DataFrame(
            strategy_rows
        ),
    )

    result.add_table(
        "validation_gate",
        pd.DataFrame(
            gate_rows
        ),
    )

    result.add_metadata(
        "selected_fi_model",
        fi_result[
            "selected_model"
        ],
    )

    result.add_metadata(
        "selected_fi_validation_auc",
        fi_result[
            "validation_auc"
        ],
    )

    result.add_metadata(
        "selected_volatility_model",
        vol_result[
            "selected_model"
        ],
    )

    result.add_metadata(
        "selected_volatility_validation_auc",
        vol_result[
            "validation_auc"
        ],
    )

    result.add_metadata(
        "fi_feature_count",
        len(fi_result[
            "features"
        ]),
    )

    result.add_metadata(
        "fi_plus_volatility_feature_count",
        len(vol_result[
            "features"
        ]),
    )

    result.add_metadata(
        "training_regime_q1",
        q1,
    )

    result.add_metadata(
        "training_regime_q2",
        q2,
    )

    result.add_metadata(
        "validation_gate",
        gate,
    )

    result.add_metadata(
        "strategies",
        list(strategies),
    )

    return result
