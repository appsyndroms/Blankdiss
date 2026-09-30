from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def stable_seed(
    *parts: object,
) -> int:
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


def regime_rate(
    target,
    mask,
) -> dict[str, Any]:
    selected = target[mask]

    valid = np.isfinite(selected)
    selected = selected[valid]

    n = int(selected.shape[0])

    if n == 0:
        return {
            "n": 0,
            "events": 0,
            "event_rate": None,
        }

    events = int(
        (selected > 0).sum()
    )

    return {
        "n": n,
        "events": events,
        "event_rate": events / n,
    }
