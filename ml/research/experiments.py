"""Deklarativ definition av Blankdiss research-matris."""

from __future__ import annotations

from dataclasses import dataclass


TAIL_FRACTIONS = (
    0.20,
    0.10,
    0.05,
    0.025,
    0.01,
)


TARGET_NAMES = (
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


SIGNAL_SPECS = (
    (
        "short_interest_level",
        ("upper",),
    ),
    (
        "short_interest_change",
        ("upper",),
    ),
    (
        "short_interest_acceleration",
        ("upper",),
    ),
    (
        "price_momentum_5d",
        ("upper", "lower"),
    ),
    (
        "price_momentum_20d",
        ("upper", "lower"),
    ),
    (
        "price_momentum_60d",
        ("upper", "lower"),
    ),
    (
        "price_volatility_20d",
        ("upper",),
    ),
    (
        "distance_from_20d_high",
        ("upper", "lower"),
    ),
    (
        "distance_from_60d_high",
        ("upper", "lower"),
    ),
)


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    signal_name: str
    target_name: str
    tail_fraction: float
    tail_direction: str


def _fraction_name(
    fraction: float,
) -> str:
    if fraction == 0.20:
        return "20pct"

    if fraction == 0.10:
        return "10pct"

    if fraction == 0.05:
        return "5pct"

    if fraction == 0.025:
        return "2_5pct"

    if fraction == 0.01:
        return "1pct"

    raise ValueError(
        f"Okänd tail-fraktion: {fraction}"
    )


def build_experiment_matrix() -> list[Experiment]:
    experiments: list[Experiment] = []

    for signal_name, directions in SIGNAL_SPECS:
        for target_name in TARGET_NAMES:
            for direction in directions:
                for fraction in TAIL_FRACTIONS:
                    experiment_id = (
                        f"{signal_name}"
                        f"__{direction}"
                        f"__{_fraction_name(fraction)}"
                        f"__{target_name}"
                    )

                    experiments.append(
                        Experiment(
                            experiment_id=experiment_id,
                            signal_name=signal_name,
                            target_name=target_name,
                            tail_fraction=fraction,
                            tail_direction=direction,
                        )
                    )

    return experiments
