"""Ekonomiskt backtest av Blankdiss OOS-signaler."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from analysis.signal_backtest.config import (
    ECONOMIC_BACKTEST_FRACTIONS,
    ECONOMIC_MAX_POSITION_WEIGHT,
    ECONOMIC_REBALANCE_DAYS,
    ECONOMIC_REBALANCE_DAYS_SENSITIVITY,
    ECONOMIC_TRANSACTION_COST_BPS,
)


def _clean_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
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


def _portfolio_weights(
    selected: pd.DataFrame,
    max_position_weight: float,
) -> dict[str, float]:
    """
    Skapa lika vikter med ett hårt maxviktstak.

    Kapital som inte kan investeras på grund av maxvikten
    ligger kvar som cash.

    Exempel med maxvikt 5 %:
        1 aktie  -> 5 %
        2 aktier -> 10 %
        10 aktier -> 50 %
        20 aktier -> 100 %

    Om antalet aktier är större än 20 viktas de lika så
    att hela portföljen används.
    """
    if selected.empty:
        return {}

    if not 0.0 < max_position_weight <= 1.0:
        raise ValueError(
            "max_position_weight måste vara > 0 och <= 1."
        )

    securities = (
        selected["security_key"]
        .astype(str)
        .tolist()
    )

    count = len(securities)

    equal_weight = 1.0 / count

    if equal_weight <= max_position_weight:
        return {
            security: equal_weight
            for security in securities
        }

    return {
        security: max_position_weight
        for security in securities
    }


def _portfolio_return(
    selected: pd.DataFrame,
    direction: str,
    max_position_weight: float,
) -> float:
    """
    Beräkna portföljens bruttoavkastning.

    Ej investerat kapital behandlas som cash med 0 % avkastning.
    """
    if selected.empty:
        raise ValueError(
            "Kan inte beräkna portföljavkastning utan innehav."
        )

    returns = selected[
        "target_return"
    ].to_numpy(
        dtype=float
    )

    if direction == "short":
        returns = -returns

    weights = _portfolio_weights(
        selected,
        max_position_weight,
    )

    weight_array = np.asarray(
        [
            weights[str(security)]
            for security in selected[
                "security_key"
            ]
        ],
        dtype=float,
    )

    return float(
        np.sum(
            weight_array
            * returns
        )
    )


def _invested_weight(
    weights: dict[str, float],
) -> float:
    """Total investerad vikt; resterande vikt är cash."""
    return float(
        sum(weights.values())
    )


def _turnover(
    previous_weights: dict[str, float],
    current_weights: dict[str, float],
) -> float:
    """
    Total portföljomsättning som halv-L1-avstånd.

    Cash ingår implicit som resterande vikt. Eftersom cashvikten
    kan förändras måste även den förändringen räknas med.

    Exempel:
        tidigare 100 % cash
        nu 5 % aktie + 95 % cash
        turnover = 5 %
    """
    securities = (
        set(previous_weights)
        | set(current_weights)
    )

    security_turnover = sum(
        abs(
            current_weights.get(
                security,
                0.0,
            )
            - previous_weights.get(
                security,
                0.0,
            )
        )
        for security in securities
    )

    previous_cash = 1.0 - _invested_weight(
        previous_weights
    )

    current_cash = 1.0 - _invested_weight(
        current_weights
    )

    cash_turnover = abs(
        current_cash
        - previous_cash
    )

    return float(
        0.5
        * (
            security_turnover
            + cash_turnover
        )
    )


def _compound(
    returns: list[float],
) -> float:
    if not returns:
        return 0.0

    returns_array = np.asarray(
        returns,
        dtype=float,
    )

    if np.any(
        1.0 + returns_array <= 0.0
    ):
        return -1.0

    return float(
        np.prod(
            1.0
            + returns_array
        )
        - 1.0
    )


def _max_drawdown(
    returns: list[float],
) -> float:
    if not returns:
        return 0.0

    returns_array = np.asarray(
        returns,
        dtype=float,
    )

    if np.any(
        1.0 + returns_array <= 0.0
    ):
        return -1.0

    equity = np.cumprod(
        1.0
        + returns_array
    )

    peaks = np.maximum.accumulate(
        equity
    )

    drawdowns = (
        equity / peaks
        - 1.0
    )

    return float(
        drawdowns.min()
    )


def _build_strategy(
    predictions: pd.DataFrame,
    fraction: float,
    direction: str,
    transaction_cost_bps: float,
    rebalance_days: int | None = None,
) -> dict[str, Any]:
    """
    Bygg en ekonomisk strategi.

    rebalance_days anger hur många observationsdagar som ska
    hoppas mellan nya portföljurval.

    Targeten är fortfarande den befintliga target_return,
    normalt en 5-dagars forward return.

    Därför ska rebalance_days inte tolkas som holding period.
    """
    if rebalance_days is None:
        rebalance_days = ECONOMIC_REBALANCE_DAYS

    if rebalance_days < 1:
        raise ValueError(
            "rebalance_days måste vara >= 1."
        )

    dates = sorted(
        predictions[
            "snapshot_date"
        ]
        .dt.normalize()
        .unique()
    )

    rebalance_dates = dates[
        ::rebalance_days
    ]

    transaction_cost_rate = (
        transaction_cost_bps
        / 10_000.0
    )

    period_returns: list[float] = []
    gross_returns: list[float] = []
    benchmark_returns: list[float] = []
    turnover_values: list[float] = []
    invested_weights: list[float] = []

    periods: list[
        dict[str, Any]
    ] = []

    previous_weights: dict[str, float] = {}

    for date in rebalance_dates:
        day = predictions.loc[
            predictions[
                "snapshot_date"
            ].dt.normalize()
            == date
        ].copy()

        if day.empty:
            continue

        day = day.sort_values(
            [
                "probability",
                "security_key",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )

        count = max(
            1,
            int(
                np.ceil(
                    len(day)
                    * fraction
                )
            ),
        )

        selected = day.iloc[
            :count
        ].copy()

        current_weights = _portfolio_weights(
            selected,
            ECONOMIC_MAX_POSITION_WEIGHT,
        )

        turnover = _turnover(
            previous_weights,
            current_weights,
        )

        gross = _portfolio_return(
            selected,
            direction,
            ECONOMIC_MAX_POSITION_WEIGHT,
        )

        transaction_cost = (
            turnover
            * transaction_cost_rate
        )

        net = (
            gross
            - transaction_cost
        )

        universe = day[
            "target_return"
        ].to_numpy(
            dtype=float
        )

        if direction == "short":
            universe = -universe

        benchmark = float(
            np.mean(
                universe
            )
        )

        invested_weight = _invested_weight(
            current_weights
        )

        period_returns.append(
            net
        )

        gross_returns.append(
            gross
        )

        benchmark_returns.append(
            benchmark
        )

        turnover_values.append(
            turnover
        )

        invested_weights.append(
            invested_weight
        )

        periods.append(
            {
                "date": str(
                    pd.Timestamp(
                        date
                    ).date()
                ),
                "rows": int(
                    len(selected)
                ),
                "invested_weight": (
                    invested_weight
                ),
                "cash_weight": (
                    1.0
                    - invested_weight
                ),
                "gross_return": gross,
                "transaction_cost": (
                    transaction_cost
                ),
                "turnover": turnover,
                "net_return": net,
                "benchmark_return": (
                    benchmark
                ),
            }
        )

        previous_weights = current_weights

    benchmark_compounded = _compound(
        benchmark_returns
    )

    gross_compounded = _compound(
        gross_returns
    )

    net_compounded = _compound(
        period_returns
    )

    return {
        "fraction": float(
            fraction
        ),
        "percentage": float(
            fraction * 100
        ),
        "direction": direction,
        "rebalance_days": int(
            rebalance_days
        ),
        "transaction_cost_bps": float(
            transaction_cost_bps
        ),
        "max_position_weight": float(
            ECONOMIC_MAX_POSITION_WEIGHT
        ),
        "periods": int(
            len(period_returns)
        ),
        "trades": int(
            sum(
                period["rows"]
                for period in periods
            )
        ),
        "mean_turnover": (
            float(
                np.mean(
                    turnover_values
                )
            )
            if turnover_values
            else 0.0
        ),
        "median_turnover": (
            float(
                np.median(
                    turnover_values
                )
            )
            if turnover_values
            else 0.0
        ),
        "mean_invested_weight": (
            float(
                np.mean(
                    invested_weights
                )
            )
            if invested_weights
            else 0.0
        ),
        "median_invested_weight": (
            float(
                np.median(
                    invested_weights
                )
            )
            if invested_weights
            else 0.0
        ),
        "gross_compounded_return": (
            gross_compounded
        ),
        "net_compounded_return": (
            net_compounded
        ),
        "benchmark_compounded_return": (
            benchmark_compounded
        ),
        "excess_return_vs_benchmark": (
            float(
                net_compounded
                - benchmark_compounded
            )
        ),
        "mean_period_return": (
            float(
                np.mean(
                    period_returns
                )
            )
            if period_returns
            else None
        ),
        "median_period_return": (
            float(
                np.median(
                    period_returns
                )
            )
            if period_returns
            else None
        ),
        "max_drawdown": _max_drawdown(
            period_returns
        ),
        "periods_detail": periods,
    }


def _build_yearly_results(
    predictions: pd.DataFrame,
    fraction: float,
    direction: str,
    transaction_cost_bps: float,
    rebalance_days: int | None = None,
) -> list[dict[str, Any]]:
    """
    Kör samma ekonomiska strategi separat per kalenderår.

    Detta används diagnostiskt för att skilja 2025 från 2026
    och undvika att ett starkt år döljer ett svagt år.
    """
    yearly_results: list[
        dict[str, Any]
    ] = []

    years = sorted(
        predictions[
            "snapshot_date"
        ]
        .dt.year
        .unique()
        .tolist()
    )

    for year in years:
        year_predictions = predictions.loc[
            predictions[
                "snapshot_date"
            ].dt.year
            == year
        ].copy()

        if year_predictions.empty:
            continue

        result = _build_strategy(
            year_predictions,
            fraction,
            direction,
            transaction_cost_bps,
            rebalance_days,
        )

        result["year"] = int(
            year
        )

        yearly_results.append(
            result
        )

    return yearly_results


def _build_cost_sensitivity(
    predictions: pd.DataFrame,
    fraction: float,
    direction: str,
    rebalance_days: int | None = None,
) -> list[dict[str, Any]]:
    """
    Kör samma strategi vid flera transaktionskostnader.
    """
    return [
        _build_strategy(
            predictions,
            fraction,
            direction,
            transaction_cost_bps,
            rebalance_days,
        )
        for transaction_cost_bps
        in ECONOMIC_TRANSACTION_COST_BPS
    ]


def _build_rebalance_sensitivity(
    predictions: pd.DataFrame,
    fraction: float,
    direction: str,
    transaction_cost_bps: float,
) -> list[dict[str, Any]]:
    """
    Kör samma strategi vid flera rebalance-intervall.

    OBS:
        Targeten är fortfarande 5 dagar.

    Detta är därför en känslighetsanalys av hur ofta modellen
    omsätts, inte ett test av olika holding-period targets.
    """
    return [
        _build_strategy(
            predictions,
            fraction,
            direction,
            transaction_cost_bps,
            rebalance_days,
        )
        for rebalance_days
        in ECONOMIC_REBALANCE_DAYS_SENSITIVITY
    ]


def _selected_security_counts(
    predictions: pd.DataFrame,
    fraction: float,
    rebalance_days: int,
) -> Counter[str]:
    """
    Räkna hur ofta varje värdepapper väljs av baseline-strategin.
    """
    dates = sorted(
        predictions[
            "snapshot_date"
        ]
        .dt.normalize()
        .unique()
    )

    rebalance_dates = dates[
        ::rebalance_days
    ]

    counts: Counter[str] = Counter()

    for date in rebalance_dates:
        day = predictions.loc[
            predictions[
                "snapshot_date"
            ].dt.normalize()
            == date
        ].copy()

        if day.empty:
            continue

        day = day.sort_values(
            [
                "probability",
                "security_key",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )

        count = max(
            1,
            int(
                np.ceil(
                    len(day)
                    * fraction
                )
            ),
        )

        selected = day.iloc[
            :count
        ]

        for security in selected[
            "security_key"
        ].astype(str):
            counts[security] += 1

    return counts


def _build_concentration_sensitivity(
    predictions: pd.DataFrame,
    fraction: float,
    direction: str,
    transaction_cost_bps: float,
    rebalance_days: int,
    top_n: int = 10,
) -> dict[str, Any]:
    """
    Testa hur beroende baseline-strategin är av de mest frekvent
    valda värdepappren.

    För varje av de top_n mest valda värdepappren körs ett
    leave-one-out-test där värdepappret tas bort ur hela
    prediktionsuniversumet innan rangordningen görs om.

    Dessutom görs ett test där alla top_n tas bort samtidigt.

    VIKTIGT:

    Benchmarken kommer från baseline-universumet och hålls därför
    konstant när ett eller flera värdepapper exkluderas.

    Detta är nödvändigt för att delta-excess ska mäta effekten
    av koncentrationen och inte en samtidig förändring av benchmark.
    """
    baseline = _build_strategy(
        predictions,
        fraction,
        direction,
        transaction_cost_bps,
        rebalance_days,
    )

    baseline_net = float(
        baseline[
            "net_compounded_return"
        ]
    )

    baseline_benchmark = float(
        baseline[
            "benchmark_compounded_return"
        ]
    )

    baseline_excess = (
        baseline_net
        - baseline_benchmark
    )

    baseline_max_drawdown = float(
        baseline[
            "max_drawdown"
        ]
    )

    selection_counts = _selected_security_counts(
        predictions,
        fraction,
        rebalance_days,
    )

    top_securities = [
        security
        for security, _count
        in selection_counts.most_common(
            top_n
        )
    ]

    tests: list[
        dict[str, Any]
    ] = []

    def run_exclusion_test(
        label: str,
        excluded_securities: set[str],
        selection_count: int,
    ) -> None:
        filtered = predictions.loc[
            ~predictions[
                "security_key"
            ].astype(str).isin(
                excluded_securities
            )
        ].copy()

        result = _build_strategy(
            filtered,
            fraction,
            direction,
            transaction_cost_bps,
            rebalance_days,
        )

        result_net = float(
            result[
                "net_compounded_return"
            ]
        )

        result_max_drawdown = float(
            result[
                "max_drawdown"
            ]
        )

        # Benchmarken ska INTE tas från den filtrerade körningen.
        # Vi använder baseline-benchmarken för alla jämförelser.
        result_excess_vs_baseline_benchmark = (
            result_net
            - baseline_benchmark
        )

        delta_net_return = (
            result_net
            - baseline_net
        )

        delta_excess_return = (
            result_excess_vs_baseline_benchmark
            - baseline_excess
        )

        tests.append(
            {
                "label": label,
                "excluded_securities": sorted(
                    excluded_securities
                ),
                "selection_count": int(
                    selection_count
                ),
                "net_compounded_return": (
                    result_net
                ),
                "delta_net_return": (
                    delta_net_return
                ),
                "excess_return_vs_baseline_benchmark": (
                    result_excess_vs_baseline_benchmark
                ),
                "delta_excess_return": (
                    delta_excess_return
                ),
                "max_drawdown": (
                    result_max_drawdown
                ),
                "delta_max_drawdown": (
                    result_max_drawdown
                    - baseline_max_drawdown
                ),
            }
        )

    for security in top_securities:
        run_exclusion_test(
            label=security,
            excluded_securities={
                security
            },
            selection_count=selection_counts[
                security
            ],
        )

    top_n_set = set(
        top_securities
    )

    run_exclusion_test(
        label=f"top_{top_n}_excluded",
        excluded_securities=top_n_set,
        selection_count=sum(
            selection_counts[
                security
            ]
            for security in top_n_set
        ),
    )

    return {
        "fraction": float(
            fraction
        ),
        "rebalance_days": int(
            rebalance_days
        ),
        "transaction_cost_bps": float(
            transaction_cost_bps
        ),
        "top_n": int(
            top_n
        ),
        "baseline": {
            "net_compounded_return": (
                baseline_net
            ),
            "benchmark_compounded_return": (
                baseline_benchmark
            ),
            "excess_return_vs_benchmark": (
                baseline_excess
            ),
            "max_drawdown": (
                baseline_max_drawdown
            ),
            "selection_counts": {
                security: int(count)
                for security, count
                in selection_counts.items()
            },
        },
        "tests": tests,
    }


def run_economic_backtest(
    predictions: pd.DataFrame,
    target_name: str,
) -> dict[str, Any]:
    """
    Kör ett OOS-ekonomiskt backtest.

    Portföljen är lika viktad inom urvalet men med ett hårt
    maxviktstak per värdepapper.

    Kapital som inte kan investeras på grund av taket
    ligger som cash.

    För varje urvalsnivå produceras:

        - huvudresultat vid 10 bps
        - separata årsresultat
        - kostnadskänslighet vid 5/10/20 bps
        - rebalance-känslighet vid 3/5/10 observationsdagar

    Viktigt:

        Detta är fortfarande ett syntetiskt ekonomiskt backtest.

        För short-strategier används -target_return.

        Borrow cost, locate constraints, borrow availability
        och faktisk short execution modelleras ännu inte.

        Rebalance-sensitiviteten ändrar inte targetens längd.
        Targeten är fortfarande 5 dagar.

        Koncentrationsanalysen körs på 1 %-urvalet och testar
        hur känsligt resultatet är för de mest frekvent valda
        värdepappren.
    """
    clean = _clean_predictions(
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
        primary = _build_strategy(
            clean,
            fraction,
            direction,
            primary_transaction_cost_bps,
            ECONOMIC_REBALANCE_DAYS,
        )

        yearly = _build_yearly_results(
            clean,
            fraction,
            direction,
            primary_transaction_cost_bps,
            ECONOMIC_REBALANCE_DAYS,
        )

        cost_sensitivity = (
            _build_cost_sensitivity(
                clean,
                fraction,
                direction,
                ECONOMIC_REBALANCE_DAYS,
            )
        )

        rebalance_sensitivity = (
            _build_rebalance_sensitivity(
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
        _build_concentration_sensitivity(
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
    }
