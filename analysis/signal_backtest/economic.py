"""Ekonomiskt backtest av Blankdiss OOS-signaler."""
from __future__ import annotations

from typing import Any

import pandas as pd

from analysis.signal_backtest.config import (
    ECONOMIC_BACKTEST_FRACTIONS,
    ECONOMIC_MAX_POSITION_WEIGHT,
    ECONOMIC_REBALANCE_DAYS,
    ECONOMIC_REBALANCE_DAYS_SENSITIVITY,
    ECONOMIC_TRANSACTION_COST_BPS,
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
    build_strategy,
    build_yearly_results,
)


def clean_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Validera och normalisera OOS-prediktioner."""
    required = {
        "snapshot_date",
        "security_key",
        "target_return",
        "probability",
    }

    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(
            "Ekonomiskt backtest saknar kolumner: "
            + ", ".join(sorted(missing))
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

    clean["probability"] = pd.to_numeric(
        clean["probability"],
        errors="coerce",
    )

    clean = clean.dropna(
        subset=[
            "snapshot_date",
            "security_key",
            "target_return",
            "probability",
        ]
    )

    clean["security_key"] = (
        clean["security_key"]
        .astype(str)
    )

    return clean.sort_values(
        [
            "snapshot_date",
            "probability",
            "security_key",
        ],
        ascending=[
            True,
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def run_economic_backtest(
    predictions: pd.DataFrame,
    target_name: str,
) -> dict[str, Any]:
    """
    Kör ett OOS-ekonomiskt backtest.

    För short-strategier används -target_return.

    Borrow cost, locate constraints, borrow availability
    och faktisk short execution modelleras ännu inte.

    Rebalance-sensitiviteten ändrar inte targetens längd.
    Targeten är fortfarande 5 dagar.

    Koncentrationsanalysen körs på 1 %-urvalet och testar
    hur känsligt resultatet är för de mest frekvent valda
    värdepappren.

    Random robustness testar samma 1 %-strategi efter att
    10 %, 20 % eller 30 % av universums värdepapper slumpmässigt
    tagits bort.
    """
    clean = clean_predictions(
        predictions
    )

    if clean.empty:
        raise ValueError(
            "Ekonomiskt backtest fick "
            "inga giltiga OOS-prediktioner."
        )

    direction = (
        "short"
        if target_name.startswith(
            "down_"
        )
        else "long"
    )

    # 10 bps används som huvudscenario för jämförbarhet
    # med tidigare körningar.
    primary_transaction_cost_bps = 10.0

    strategies: list[
        dict[str, Any]
    ] = []

    for fraction in ECONOMIC_BACKTEST_FRACTIONS:
        primary = build_strategy(
            clean,
            fraction,
            direction,
            primary_transaction_cost_bps,
            ECONOMIC_REBALANCE_DAYS,
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
            )
        )

        rebalance_sensitivity = (
            build_rebalance_sensitivity(
                clean,
                fraction,
                direction,
                primary_transaction_cost_bps,
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

    return {
        "target": target_name,
        "direction": direction,
        "selection": (
            "highest predicted probability"
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
            for value
            in ECONOMIC_TRANSACTION_COST_BPS
        ],
        "primary_rebalance_days": int(
            ECONOMIC_REBALANCE_DAYS
        ),
        "rebalance_scenarios_days": [
            int(value)
            for value
            in ECONOMIC_REBALANCE_DAYS_SENSITIVITY
        ],
        "target_horizon_days": 5,
        "strategies": strategies,
        "concentration_sensitivity": (
            concentration_sensitivity
        ),
        "random_security_sensitivity": (
            random_security_sensitivity
        ),
    }
