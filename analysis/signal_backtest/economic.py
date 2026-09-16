"""Ekonomiskt backtest av Blankdiss OOS-signaler."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from analysis.signal_backtest.config import (
    ECONOMIC_BACKTEST_FRACTIONS,
    ECONOMIC_MAX_POSITION_WEIGHT,
    ECONOMIC_REBALANCE_DAYS,
    ECONOMIC_REBALANCE_DAYS_SENSITIVITY,
    ECONOMIC_TRANSACTION_COST_BPS,
)
from analysis.signal_backtest.ml_adapter import (
    load_and_prepare_economic_predictions,
)
from analysis.signal_backtest.robustness import (
    build_random_security_sensitivity,
)
from analysis.signal_backtest.sensitivity import (
    build_concentration_sensitivity,
    build_cost_sensitivity,
    build_rebalance_sensitivity,
)
from analysis.signal_backtest.strategy import (
    StrategyCache,
    build_strategy,
    build_yearly_results,
)


def clean_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Validera och normalisera ekonomiska OOS-prediktioner."""

    required = {
        "snapshot_date",
        "security_key",
        "target_return",
        "score",
        "economic_direction",
    }

    missing = required - set(
        predictions.columns
    )

    if missing:
        raise ValueError(
            "Ekonomiskt backtest saknar "
            "kolumner: "
            + ", ".join(
                sorted(missing)
            )
        )

    clean = predictions.copy()

    clean["snapshot_date"] = pd.to_datetime(
        clean["snapshot_date"],
        errors="coerce",
    )

    clean["target_return"] = pd.to_numeric(
        clean["target_return"],
        errors="coerce",
    )

    clean["score"] = pd.to_numeric(
        clean["score"],
        errors="coerce",
    )

    clean = clean.dropna(
        subset=[
            "snapshot_date",
            "security_key",
            "target_return",
            "score",
        ]
    )

    clean["security_key"] = (
        clean["security_key"]
        .astype(str)
    )

    return clean.sort_values(
        [
            "snapshot_date",
            "score",
            "security_key",
        ],
        ascending=[
            True,
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )


def _target_horizon_days(
    target_name: str,
) -> int | None:
    """Försök härleda target-horisonten från targetnamnet."""

    matches = re.findall(
        r"_(\d+)d(?:$|_)",
        target_name,
    )

    if not matches:
        return None

    return int(
        matches[-1]
    )


def _selected_models(
    predictions: pd.DataFrame,
) -> list[str]:
    """Returnera alla modeller som faktiskt producerat OOS-rader."""

    if "model" not in predictions.columns:
        return []

    return (
        predictions["model"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )


def _window_group_key(
    window: Any,
) -> str:
    """Skapa en deterministisk gruppnyckel för ett walk-forward-fönster."""

    if isinstance(window, dict):
        return json.dumps(
            window,
            sort_keys=True,
            ensure_ascii=False,
        )

    return str(window)


def _window_output(
    window_key: str,
    original_window: Any,
) -> Any:
    """Återställ serialiserat window till ursprunglig struktur."""

    if isinstance(original_window, dict):
        return original_window

    try:
        return json.loads(
            window_key
        )
    except (
        TypeError,
        json.JSONDecodeError,
    ):
        return window_key


def _model_summary(
    predictions: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Sammanfatta vald modell per walk-forward-fönster."""

    if (
        "model" not in predictions.columns
        or "window" not in predictions.columns
    ):
        return []

    working = predictions.loc[
        predictions["model"].notna()
        & predictions["window"].notna()
    ].copy()

    if working.empty:
        return []

    working[
        "_window_group_key"
    ] = working["window"].map(
        _window_group_key
    )

    rows: list[
        dict[str, Any]
    ] = []

    grouped = working.groupby(
        "_window_group_key",
        sort=True,
    )

    for window_key, group in grouped:
        models = (
            group["model"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        original_window = group.iloc[0][
            "window"
        ]

        rows.append(
            {
                "window": _window_output(
                    str(window_key),
                    original_window,
                ),
                "models": models,
                "rows": int(
                    len(group)
                ),
            }
        )

    return rows


def _run_single_experiment(
    predictions: pd.DataFrame,
    target_name: str,
    direction: str,
    feature_set: str,
    task: str,
    model: str | None,
) -> dict[str, Any]:
    """Kör hela ekonomiska analysen för ett ML-experiment."""

    clean = clean_predictions(
        predictions
    )

    if clean.empty:
        raise ValueError(
            "Ekonomiskt backtest fick "
            "inga giltiga OOS-prediktioner."
        )

    primary_transaction_cost_bps = 10.0

    # EN cache för hela experimentet.
    # Alla strategier som använder samma predictions
    # återanvänder datumgrupper och ranking.
    strategy_cache = StrategyCache(
        clean
    )

    strategies: list[
        dict[str, Any]
    ] = []

    for fraction in (
        ECONOMIC_BACKTEST_FRACTIONS
    ):
        primary = build_strategy(
            clean,
            fraction,
            direction,
            primary_transaction_cost_bps,
            ECONOMIC_REBALANCE_DAYS,
            cache=strategy_cache,
        )

        yearly = build_yearly_results(
            clean,
            fraction,
            direction,
            primary_transaction_cost_bps,
            ECONOMIC_REBALANCE_DAYS,
        )

        cost_sensitivity = (
            build_cost_sensitivity(
                clean,
                fraction,
                direction,
                ECONOMIC_REBALANCE_DAYS,
                cache=strategy_cache,
            )
        )

        rebalance_sensitivity = (
            build_rebalance_sensitivity(
                clean,
                fraction,
                direction,
                primary_transaction_cost_bps,
                cache=strategy_cache,
            )
        )

        primary["yearly"] = yearly

        primary[
            "transaction_cost_sensitivity"
        ] = cost_sensitivity

        primary[
            "rebalance_sensitivity"
        ] = rebalance_sensitivity

        strategies.append(
            primary
        )

    concentration_sensitivity = (
        build_concentration_sensitivity(
            clean,
            fraction=0.01,
            direction=direction,
            transaction_cost_bps=(
                primary_transaction_cost_bps
            ),
            rebalance_days=(
                ECONOMIC_REBALANCE_DAYS
            ),
            top_n=10,
        )
    )

    random_security_sensitivity = (
        build_random_security_sensitivity(
            clean,
            strategy_builder=build_strategy,
            fraction=0.01,
            direction=direction,
            transaction_cost_bps=(
                primary_transaction_cost_bps
            ),
            rebalance_days=(
                ECONOMIC_REBALANCE_DAYS
            ),
        )
    )

    selected_models = _selected_models(
        clean
    )

    model_by_window = _model_summary(
        clean
    )

    horizon = _target_horizon_days(
        target_name
    )

    if model is None:
        if len(selected_models) == 1:
            model_label = (
                selected_models[0]
            )
        else:
            model_label = (
                "walk_forward"
            )
    else:
        model_label = model

    return {
        "feature_set": feature_set,
        "target": target_name,
        "task": task,
        "model": model_label,
        "selected_models": selected_models,
        "model_by_window": model_by_window,
        "direction": direction,
        "economic_direction": direction,
        "selection": (
            "highest economic score"
        ),
        "no_overlapping_periods": True,
        "portfolio_weighting": (
            "equal_weighted_with_max_position_cap"
        ),
        "max_position_weight": float(
            ECONOMIC_MAX_POSITION_WEIGHT
        ),
        "cash_allowed": True,
        "turnover_cost_model": (
            "actual_portfolio_turnover"
        ),
        "primary_transaction_cost_bps": (
            primary_transaction_cost_bps
        ),
        "transaction_cost_scenarios_bps": [
            float(value)
            for value in (
                ECONOMIC_TRANSACTION_COST_BPS
            )
        ],
        "primary_rebalance_days": int(
            ECONOMIC_REBALANCE_DAYS
        ),
        "rebalance_scenarios_days": [
            int(value)
            for value in (
                ECONOMIC_REBALANCE_DAYS_SENSITIVITY
            )
        ],
        "target_horizon_days": horizon,
        "oos_rows": int(
            len(clean)
        ),
        "oos_start": str(
            clean["snapshot_date"]
            .min()
            .date()
        ),
        "oos_end": str(
            clean["snapshot_date"]
            .max()
            .date()
        ),
        "strategies": strategies,
        "concentration_sensitivity": (
            concentration_sensitivity
        ),
        "random_security_sensitivity": (
            random_security_sensitivity
        ),
    }


def run_economic_backtest(
    predictions: pd.DataFrame,
    target_name: str,
    direction: str | None = None,
    feature_set: str | None = None,
    task: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Kör ekonomiskt backtest för ett filtrerat experiment."""

    if predictions.empty:
        raise ValueError(
            "Ekonomiskt backtest fick "
            "inga prediktioner."
        )

    if direction is None:
        direction = str(
            predictions.iloc[0][
                "economic_direction"
            ]
        )

    if feature_set is None:
        feature_set = str(
            predictions.iloc[0][
                "feature_set"
            ]
        )

    if task is None:
        task = str(
            predictions.iloc[0][
                "task"
            ]
        )

    if model is None:
        models = _selected_models(
            predictions
        )

        if len(models) == 1:
            model = models[0]

    return _run_single_experiment(
        predictions,
        target_name,
        direction,
        feature_set,
        task,
        model,
    )


def run_all_economic_backtests() -> list[
    dict[str, Any]
]:
    """Kör ekonomiskt backtest för alla experiment."""

    predictions = (
        load_and_prepare_economic_predictions()
    )

    if predictions.empty:
        raise ValueError(
            "OOS-prediktionerna innehåller "
            "inga giltiga ekonomiska signaler."
        )

    experiment_columns = [
        "feature_set",
        "target",
        "task",
        "economic_direction",
    ]

    results: list[
        dict[str, Any]
    ] = []

    grouped = predictions.groupby(
        experiment_columns,
        sort=True,
    )

    for (
        feature_set,
        target,
        task,
        economic_direction,
    ), group in grouped:
        selected_models = _selected_models(
            group
        )

        model = (
            selected_models[0]
            if len(selected_models) == 1
            else None
        )

        result = run_economic_backtest(
            group.copy(),
            target_name=str(target),
            direction=str(
                economic_direction
            ),
            feature_set=str(
                feature_set
            ),
            task=str(task),
            model=model,
        )

        results.append(
            result
        )

    return results
