"""Bootstrap statistics for Blankdiss research."""

from __future__ import annotations

import numpy as np


DEFAULT_ITERATIONS = 2000
MIN_ROWS = 20
CHUNK_SIZE = 100


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int,
) -> tuple[float | None, float | None]:
    """
    Bootstrap 95% confidence interval for the mean.

    NumPy arrays are used throughout the bootstrap loop.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[np.isfinite(values)]

    if len(values) < MIN_ROWS:
        return None, None

    rng = np.random.default_rng(seed)

    n = len(values)

    means = np.empty(
        iterations,
        dtype=np.float64,
    )

    offset = 0

    while offset < iterations:
        current = min(
            CHUNK_SIZE,
            iterations - offset,
        )

        indices = rng.integers(
            0,
            n,
            size=(current, n),
        )

        means[
            offset:offset + current
        ] = values[indices].mean(axis=1)

        offset += current

    lower, upper = np.quantile(
        means,
        [0.025, 0.975],
    )

    return (
        float(lower),
        float(upper),
    )


def bootstrap_mean_difference(
    tail_returns: np.ndarray,
    rest_returns: np.ndarray,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int,
) -> tuple[float | None, float | None]:
    """
    Bootstrap 95% CI for:

        mean(tail) - mean(rest)
    """
    tail = np.asarray(
        tail_returns,
        dtype=np.float64,
    )

    rest = np.asarray(
        rest_returns,
        dtype=np.float64,
    )

    tail = tail[np.isfinite(tail)]
    rest = rest[np.isfinite(rest)]

    if (
        len(tail) < MIN_ROWS
        or len(rest) < MIN_ROWS
    ):
        return None, None

    rng = np.random.default_rng(seed)

    tail_n = len(tail)
    rest_n = len(rest)

    differences = np.empty(
        iterations,
        dtype=np.float64,
    )

    offset = 0

    while offset < iterations:
        current = min(
            CHUNK_SIZE,
            iterations - offset,
        )

        tail_indices = rng.integers(
            0,
            tail_n,
            size=(current, tail_n),
        )

        rest_indices = rng.integers(
            0,
            rest_n,
            size=(current, rest_n),
        )

        tail_means = (
            tail[tail_indices]
            .mean(axis=1)
        )

        rest_means = (
            rest[rest_indices]
            .mean(axis=1)
        )

        differences[
            offset:offset + current
        ] = tail_means - rest_means

        offset += current

    lower, upper = np.quantile(
        differences,
        [0.025, 0.975],
    )

    return (
        float(lower),
        float(upper),
    )
