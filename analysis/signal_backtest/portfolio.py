"""Grundläggande portföljberäkningar för Blankdiss."""
from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_weights(
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


def portfolio_return(
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

    weights = portfolio_weights(
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


def invested_weight(
    weights: dict[str, float],
) -> float:
    """Total investerad vikt; resterande vikt är cash."""
    return float(
        sum(weights.values())
    )


def turnover(
    previous_weights: dict[str, float],
    current_weights: dict[str, float],
) -> float:
    """
    Total portföljomsättning som halv-L1-avstånd.

    Cash ingår implicit som resterande vikt. Eftersom cashvikten
    kan förändras måste även den förändringen räknas med.
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

    previous_cash = 1.0 - invested_weight(
        previous_weights
    )

    current_cash = 1.0 - invested_weight(
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


def compound(
    returns: list[float],
) -> float:
    """Beräkna sammansatt avkastning."""
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


def max_drawdown(
    returns: list[float],
) -> float:
    """Beräkna maximal drawdown från periodavkastningar."""
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
