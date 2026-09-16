"""Ekonomiskt backtest av Blankdiss OOS-signaler."""
from __future__ import annotations

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


def _selected_security_counts(
    predictions: pd.DataFrame,
    fraction: float,
    rebalance_days: int,
) -> pd.Series:
    """
    Räkna hur ofta varje värdepapper väljs vid varje rebalance.

    Detta använder exakt samma rankinglogik som den ekonomiska
    strategin men utan att beräkna avkastning.

    Returnerar en Series sorterad efter antal val.
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

    selected_securities: list[str] = []

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

        selected_securities.extend(
            selected[
                "security_key"
            ]
            .astype(str)
            .tolist()
        )

    if not selected_securities:
        return pd.Series(
            dtype="int64"
        )

    return (
        pd.Series(
            selected_securities,
            dtype="string",
        )
        .value_counts()
    )


def _build_concentration_sensitivity(
    predictions: pd.DataFrame,
    direction: str,
    transaction_cost_bps: float,
    rebalance_days: int,
    fraction: float = 0.01,
    top_n: int = 10,
) -> dict[str, Any]:
    """
    Analysera hur beroende top-1%-strategin är av enskilda
    värdepapper.

    Baseline är den normala top-1%-strategin.

    Därefter görs:

        1. Leave-one-out för de 10 mest frekvent valda
           värdepapperen.
        2. Ett test där alla 10 tas bort samtidigt.

    Ett värdepapper tas bort från hela prediktionsuniversumet
    innan rankingen görs om. Det innebär att nästa högst
    rankade värdepapper får möjlighet att ersätta det borttagna.

    Detta är en robusthetsdiagnostik, inte en optimering.
    """
    baseline = _build_strategy(
        predictions,
        fraction,
        direction,
        transaction_cost_bps,
        rebalance_days,
    )

    counts = _selected_security_counts(
        predictions,
        fraction,
        rebalance_days,
    )

    top = counts.head(
        top_n
    )

    leave_one_out: list[
        dict[str, Any]
    ] = []

    baseline_net = float(
        baseline[
            "net_compounded_return"
        ]
    )

    baseline_excess = float(
        baseline[
            "excess_return_vs_benchmark"
        ]
    )

    baseline_drawdown = float(
        baseline[
            "max_drawdown"
        ]
    )

    for rank, security in enumerate(
        top.index.tolist(),
        start=1,
    ):
        security = str(
            security
        )

        filtered = predictions.loc[
            predictions[
                "security_key"
            ].astype(str)
            != security
        ].copy()

        if filtered.empty:
            continue

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

        result_excess = float(
            result[
                "excess_return_vs_benchmark"
            ]
        )

        result_drawdown = float(
            result[
                "max_drawdown"
            ]
        )

        leave_one_out.append(
            {
                "rank": int(rank),
                "security_key": security,
                "selection_count": int(
                    counts.loc[
                        security
                    ]
                ),
                "selection_share": float(
                    counts.loc[
                        security
                    ]
                    / counts.sum()
                ),
                "net_compounded_return": (
                    result_net
                ),
                "delta_net_return": (
                    result_net
                    - baseline_net
                ),
                "excess_return_vs_benchmark": (
                    result_excess
                ),
                "delta_excess_return": (
                    result_excess
                    - baseline_excess
                ),
                "max_drawdown": (
                    result_drawdown
                ),
                "delta_max_drawdown": (
                    result_drawdown
                    - baseline_drawdown
                ),
            }
        )

    top_security_keys = [
        str(security)
        for security in top.index.tolist()
    ]

    filtered_top = predictions.loc[
        ~predictions[
            "security_key"
        ].astype(str).isin(
            top_security_keys
        )
    ].copy()

    top_n_result: dict[str, Any] | None = None

    if not filtered_top.empty:
        result = _build_strategy(
            filtered_top,
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

        result_excess = float(
            result[
                "excess_return_vs_benchmark"
            ]
        )

        result_drawdown = float(
            result[
                "max_drawdown"
            ]
        )

        top_n_result = {
            "excluded_count": int(
                len(top_security_keys)
            ),
            "excluded_securities": (
                top_security_keys
            ),
            "net_compounded_return": (
                result_net
            ),
            "delta_net_return": (
                result_net
                - baseline_net
            ),
            "excess_return_vs_benchmark": (
                result_excess
            ),
            "delta_excess_return": (
                result_excess
                - baseline_excess
            ),
            "max_drawdown": (
                result_drawdown
            ),
            "delta_max_drawdown": (
                result_drawdown
                - baseline_drawdown
            ),
        }

    return {
        "fraction": float(
            fraction
        ),
        "percentage": float(
            fraction * 100
        ),
        "rebalance_days": int(
            rebalance_days
        ),
        "transaction_cost_bps": float(
            transaction_cost_bps
        ),
        "baseline": {
            "net_compounded_return": (
                baseline_net
            ),
            "excess_return_vs_benchmark": (
                baseline_excess
            ),
            "max_drawdown": (
                baseline_drawdown
            ),
        },
        "selection_counts": [
            {
                "rank": int(rank),
                "security_key": str(
                    security
                ),
                "selection_count": int(
                    counts.loc[
                        security
                    ]
                ),
            }
            for rank, security in enumerate(
                top.index.tolist(),
                start=1,
            )
        ],
        "leave_one_out": leave_one_out,
        "exclude_top_n": top_n_result,
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

    För top-1%-nivån produceras dessutom:

        - frekvens för de mest valda värdepapperen
        - leave-one-out-koncentrationstest
        - test där de 10 mest valda tas bort samtidigt

    Viktigt:

        Detta är fortfarande ett syntetiskt ekonomiskt backtest.

        För short-strategier används -target_return.

        Borrow cost, locate constraints, borrow availability
        och faktisk short execution modelleras ännu inte.

        Rebalance-sensitiviteten ändrar inte targetens längd.
        Targeten är fortfarande 5 dagar.

        Koncentrationsanalysen är en robusthetsdiagnostik.
        Den är inte en modelloptimering.
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

    concentration_sensitivity: dict[
        str, Any
    ] | None = None

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

        # Koncentrationsanalysen görs specifikt för top 1 %.
        # Den körs för både UP och DOWN eftersom den ligger
        # inne i det generella ekonomiska backtestet.
        if np.isclose(
            fraction,
            0.01,
        ):
            concentration_sensitivity = (
                _build_concentration_sensitivity(
                    clean,
                    direction,
                    primary_transaction_cost_bps,
                    ECONOMIC_REBALANCE_DAYS,
                    fraction=0.01,
                    top_n=10,
                )
            )

        strategies.append(
            primary
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
        "concentration_sensitivity": (
            concentration_sensitivity
        ),
        "strategies": strategies,
    }
