from __future__ import annotations

import numpy as np

from .bootstrap import MIN_ROWS
from .cache import _tail_key


def _stable_seed(
    *parts: object,
) -> int:
    import hashlib

    payload = "|".join(
        str(part)
        for part in parts
    ).encode("utf-8")

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    ) % (2**32 - 1)


def _rate(
    target: np.ndarray,
    mask: np.ndarray,
) -> tuple[int, int, float | None]:
    selected = target[mask]

    n = int(selected.shape[0])

    if n == 0:
        return 0, 0, None

    events = int(
        (selected > 0).sum()
    )

    return (
        n,
        events,
        events / n,
    )


def _bootstrap_additive_interaction(
    target: np.ndarray,
    signal_1_selected: np.ndarray,
    signal_2_selected: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float | None, float | None]:
    """
    Bootstrap 95% CI for the additive interaction effect:

        p11 - p10 - p01 + p00

    where:

        p11 = event rate when both signals are selected
        p10 = event rate for signal 1 only
        p01 = event rate for signal 2 only
        p00 = event rate when neither signal is selected

    A value around zero is consistent with no additive
    interaction. A positive value means that the joint
    event rate exceeds the additive combination of the
    individual regime effects.
    """
    target = np.asarray(
        target,
        dtype=np.float64,
    )

    signal_1_selected = np.asarray(
        signal_1_selected,
        dtype=bool,
    )

    signal_2_selected = np.asarray(
        signal_2_selected,
        dtype=bool,
    )

    valid = np.isfinite(
        target
    )

    if int(valid.sum()) < MIN_ROWS:
        return None, None

    y = target[valid]
    signal_1 = signal_1_selected[valid]
    signal_2 = signal_2_selected[valid]

    n = len(y)

    if n < MIN_ROWS:
        return None, None

    rng = np.random.default_rng(
        seed
    )

    interactions: list[float] = []

    chunk_size = 100

    offset = 0

    while offset < iterations:
        current = min(
            chunk_size,
            iterations - offset,
        )

        indices = rng.integers(
            0,
            n,
            size=(
                current,
                n,
            ),
        )

        sampled_y = y[
            indices
        ]

        sampled_signal_1 = (
            signal_1[
                indices
            ]
        )

        sampled_signal_2 = (
            signal_2[
                indices
            ]
        )

        events = (
            sampled_y > 0
        )

        both = (
            sampled_signal_1
            & sampled_signal_2
        )

        signal_1_only = (
            sampled_signal_1
            & ~sampled_signal_2
        )

        signal_2_only = (
            ~sampled_signal_1
            & sampled_signal_2
        )

        neither = (
            ~sampled_signal_1
            & ~sampled_signal_2
        )

        both_n = both.sum(
            axis=1
        )

        signal_1_only_n = (
            signal_1_only.sum(
                axis=1
            )
        )

        signal_2_only_n = (
            signal_2_only.sum(
                axis=1
            )
        )

        neither_n = neither.sum(
            axis=1
        )

        both_events = (
            both & events
        ).sum(
            axis=1
        )

        signal_1_only_events = (
            signal_1_only & events
        ).sum(
            axis=1
        )

        signal_2_only_events = (
            signal_2_only & events
        ).sum(
            axis=1
        )

        neither_events = (
            neither & events
        ).sum(
            axis=1
        )

        p11 = np.divide(
            both_events,
            both_n,
            out=np.full(
                current,
                np.nan,
                dtype=np.float64,
            ),
            where=both_n > 0,
        )

        p10 = np.divide(
            signal_1_only_events,
            signal_1_only_n,
            out=np.full(
                current,
                np.nan,
                dtype=np.float64,
            ),
            where=signal_1_only_n > 0,
        )

        p01 = np.divide(
            signal_2_only_events,
            signal_2_only_n,
            out=np.full(
                current,
                np.nan,
                dtype=np.float64,
            ),
            where=signal_2_only_n > 0,
        )

        p00 = np.divide(
            neither_events,
            neither_n,
            out=np.full(
                current,
                np.nan,
                dtype=np.float64,
            ),
            where=neither_n > 0,
        )

        values = (
            p11
            - p10
            - p01
            + p00
        )

        interactions.extend(
            values[
                np.isfinite(values)
            ].tolist()
        )

        offset += current

    if len(interactions) < MIN_ROWS:
        return None, None

    values = np.asarray(
        interactions,
        dtype=np.float64,
    )

    lower, upper = np.quantile(
        values,
        [0.025, 0.975],
    )

    return (
        float(lower),
        float(upper),
    )


def analyse_interaction(
    cache,
    signals,
    fractions,
    target_name: str,
    window_name: str,
    split_name: str,
    *,
    bootstrap: bool,
    bootstrap_iterations: int,
    spec_id: str,
) -> dict:
    """
    Analyse a two-signal interaction.

    The analysis reports:

    - signal 1 regime
    - signal 2 regime
    - joint regime
    - signal 1 only
    - signal 2 only
    - neither signal
    - additive interaction effect

    Additive interaction:

        p11 - p10 - p01 + p00

    This is different from simply comparing the joint
    event rate with either individual signal.
    """
    if len(signals) != 2:
        raise ValueError(
            "interaction requires exactly two signals."
        )

    if len(fractions) != 2:
        raise ValueError(
            "interaction requires exactly two fractions."
        )

    target = cache.targets[
        target_name
    ]

    window_mask = cache.window_masks[
        window_name
    ][split_name]

    first_mask = cache.tail_masks[
        _tail_key(
            signals[0].name,
            signals[0].direction,
            fractions[0],
        )
    ]

    second_mask = cache.tail_masks[
        _tail_key(
            signals[1].name,
            signals[1].direction,
            fractions[1],
        )
    ]

    first = (
        window_mask
        & first_mask
    )

    second = (
        window_mask
        & second_mask
    )

    combined = (
        first
        & second
    )

    first_only = (
        first
        & ~second
    )

    second_only = (
        ~first
        & second
    )

    neither = (
        ~first
        & ~second
        & window_mask
    )

    first_metrics = _rate(
        target,
        first,
    )

    second_metrics = _rate(
        target,
        second,
    )

    combined_metrics = _rate(
        target,
        combined,
    )

    first_only_metrics = _rate(
        target,
        first_only,
    )

    second_only_metrics = _rate(
        target,
        second_only,
    )

    neither_metrics = _rate(
        target,
        neither,
    )

    p11 = combined_metrics[2]
    p10 = first_only_metrics[2]
    p01 = second_only_metrics[2]
    p00 = neither_metrics[2]

    additive_interaction = None

    if (
        p11 is not None
        and p10 is not None
        and p01 is not None
        and p00 is not None
    ):
        additive_interaction = (
            p11
            - p10
            - p01
            + p00
        )

    seed = _stable_seed(
        spec_id,
        signals[0].name,
        signals[0].direction,
        fractions[0],
        signals[1].name,
        signals[1].direction,
        fractions[1],
        target_name,
        window_name,
        split_name,
    )

    ci_low = None
    ci_high = None

    if bootstrap:
        (
            ci_low,
            ci_high,
        ) = _bootstrap_additive_interaction(
            target[window_mask],
            first_mask[window_mask],
            second_mask[window_mask],
            iterations=bootstrap_iterations,
            seed=seed,
        )

    return {
        "analysis": "interaction",
        "signal_1": signals[0].name,
        "signal_1_direction": (
            signals[0].direction
        ),
        "signal_1_fraction": fractions[0],
        "signal_2": signals[1].name,
        "signal_2_direction": (
            signals[1].direction
        ),
        "signal_2_fraction": fractions[1],
        "target": target_name,
        "window": window_name,
        "split": split_name,
        "signal_1_n": first_metrics[0],
        "signal_1_events": first_metrics[1],
        "signal_1_event_rate": first_metrics[2],
        "signal_2_n": second_metrics[0],
        "signal_2_events": second_metrics[1],
        "signal_2_event_rate": second_metrics[2],
        "combined_n": combined_metrics[0],
        "combined_events": combined_metrics[1],
        "combined_event_rate": combined_metrics[2],
        "signal_1_only_n": (
            first_only_metrics[0]
        ),
        "signal_1_only_events": (
            first_only_metrics[1]
        ),
        "signal_1_only_event_rate": (
            first_only_metrics[2]
        ),
        "signal_2_only_n": (
            second_only_metrics[0]
        ),
        "signal_2_only_events": (
            second_only_metrics[1]
        ),
        "signal_2_only_event_rate": (
            second_only_metrics[2]
        ),
        "neither_n": (
            neither_metrics[0]
        ),
        "neither_events": (
            neither_metrics[1]
        ),
        "neither_event_rate": (
            neither_metrics[2]
        ),
        "additive_interaction": (
            additive_interaction
        ),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
    }
