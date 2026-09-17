"""
Definition av Blankdiss research matrix.

Experimenten definieras deklarativt så att hundratals tester
kan genereras utan hundratals manuellt skrivna experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class Experiment:
    name: str
    signal: str
    tail: str
    description: str


SIGNAL_TAILS = (
    ("top_20", 0.80),
    ("top_10", 0.90),
    ("top_5", 0.95),
    ("top_2_5", 0.975),
    ("top_1", 0.99),
)


TARGETS = (
    "up_5pct_5d",
    "up_5pct_20d",
    "up_10pct_60d",
    "down_3pct_5d",
    "down_5pct_5d",
    "down_7pct_5d",
    "down_10pct_5d",
    "down_5pct_20d",
    "down_10pct_60d",
)


BASE_SIGNALS = (
    "short_interest_level",
    "short_interest_change",
    "short_interest_acceleration",
    "event_risk",
    "price_momentum",
    "price_volatility",
    "distance_from_20d_high",
    "distance_from_60d_high",
)


INTERACTION_SIGNALS = (
    "event_risk_x_short_interest_change",
    "event_risk_x_short_interest_acceleration",
    "event_risk_x_price_volatility",
    "event_risk_x_price_momentum",
    "short_interest_change_x_price_volatility",
    "short_interest_change_x_price_momentum",
)


def build_experiments() -> list[Experiment]:
    experiments: list[Experiment] = []

    for signal in BASE_SIGNALS:
        for tail, _ in SIGNAL_TAILS:
            experiments.append(
                Experiment(
                    name=f"{signal}_{tail}",
                    signal=signal,
                    tail=tail,
                    description=(
                        f"{signal} {tail}"
                    ),
                )
            )

    for signal in INTERACTION_SIGNALS:
        for tail, _ in SIGNAL_TAILS:
            experiments.append(
                Experiment(
                    name=f"{signal}_{tail}",
                    signal=signal,
                    tail=tail,
                    description=(
                        f"{signal} {tail}"
                    ),
                )
            )

    return experiments


EXPERIMENTS = tuple(
    build_experiments()
)


def tail_fraction(
    tail: str,
) -> float:
    mapping = dict(SIGNAL_TAILS)

    if tail not in mapping:
        raise ValueError(
            f"Okänd tail: {tail}"
        )

    return mapping[tail]


def target_names() -> tuple[str, ...]:
    return TARGETS
