from __future__ import annotations

import numpy as np

from ml.config import RANDOM_STATE


DEFAULT_ITERATIONS = 2000
MIN_ROWS = 20


def bootstrap_mean_difference(
    tail_returns: np.ndarray,
    rest_returns: np.ndarray,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = RANDOM_STATE,
) -> tuple[float | None, float | None]:
    """
    Bootstrap 95% CI for:

        mean(tail_returns) - mean(rest_returns)

    Returns:
        (lower, upper)

    The implementation uses NumPy arrays rather than pandas operations
    inside the bootstrap loop.
    """

    tail_returns = np.asarray(
        tail_returns,
        dtype=np.float64,
    )

    rest_returns = np.asarray(
        rest_returns,
        dtype=np.float64,
    )

    tail_returns = tail_returns[
        np.isfinite(tail_returns)
    ]

    rest_returns = rest_returns[
        np.isfinite(rest_returns)
    ]

    if (
        len(tail_returns) < MIN_ROWS
        or len(rest_returns) < MIN_ROWS
    ):
        return None, None

    rng = np.random.default_rng(seed)

    tail_n = len(tail_returns)
    rest_n = len(rest_returns)

    # Process in chunks to avoid creating one enormous
    # iterations × n array.
    chunk_size = 100

    differences = np.empty(
        iterations,
        dtype=np.float64,
    )

    offset = 0

    while offset < iterations:
        current = min(
            chunk_size,
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

        tail_means = tail_returns[
            tail_indices
        ].mean(axis=1)

        rest_means = rest_returns[
            rest_indices
        ].mean(axis=1)

        differences[
            offset:offset + current
        ] = tail_means - rest_means

        offset += current

    lower, upper = np.quantile(
        differences,
        [0.025, 0.975],
    )

    return float(lower), float(upper)
