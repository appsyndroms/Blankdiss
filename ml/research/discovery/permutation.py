from __future__ import annotations

from dataclasses import replace

import numpy as np

from .engine import DiscoveryData


def _permutation_indices(
    indices: np.ndarray,
    rng: np.random.Generator,
    block_size: int,
) -> np.ndarray:
    if indices.size <= 1:
        return indices.copy()

    if block_size <= 1:
        return rng.permutation(indices)

    blocks = [
        indices[start:start + block_size]
        for start in range(
            0,
            indices.size,
            block_size,
        )
    ]

    order = rng.permutation(
        len(blocks)
    )

    return np.concatenate(
        [blocks[index] for index in order]
    )


def build_permutation_index(
    data: DiscoveryData,
    rng: np.random.Generator,
    block_size: int = 1,
) -> np.ndarray:
    permutation = np.arange(
        len(data.frame),
        dtype=np.int64,
    )

    for window in data.windows.values():
        test_indices = np.flatnonzero(
            window["test"]
        )

        if test_indices.size <= 1:
            continue

        permutation[test_indices] = (
            _permutation_indices(
                test_indices,
                rng,
                block_size,
            )
        )

    return permutation


def permute_array(
    values: np.ndarray,
    permutation: np.ndarray,
) -> np.ndarray:
    result = values.copy()

    for start in range(
        len(values)
    ):
        source = permutation[start]

        if source == start:
            continue

        result[start] = values[source]

    return result


def permute_discovery_data(
    data: DiscoveryData,
    rng: np.random.Generator,
    block_size: int = 1,
) -> DiscoveryData:
    permutation = build_permutation_index(
        data=data,
        rng=rng,
        block_size=block_size,
    )

    targets = {
        name: permute_array(
            values,
            permutation,
        )
        for name, values
        in data.targets.items()
    }

    returns = {
        name: permute_array(
            values,
            permutation,
        )
        for name, values
        in data.returns.items()
    }

    return replace(
        data,
        targets=targets,
        returns=returns,
    )
