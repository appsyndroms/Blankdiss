from __future__ import annotations

import hashlib
from typing import Any


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
    n = int(selected.shape[0])

    if n == 0:
        return {
            "n": 0,
            "events": 0,
            "event_rate": None,
        }

    events = int(selected.sum())

    return {
        "n": n,
        "events": events,
        "event_rate": events / n,
    }
