import numpy as np

from ml.research.analysis_utils import (
    regime_rate,
)


def test_regime_rate_ignores_nan_targets():
    target = np.array(
        [1.0, 0.0, np.nan, 1.0, np.nan]
    )
    mask = np.array(
        [True, True, True, True, False]
    )

    result = regime_rate(
        target,
        mask,
    )

    assert result == {
        "n": 3,
        "events": 2,
        "event_rate": 2 / 3,
    }


def test_regime_rate_returns_empty_result_when_all_targets_are_nan():
    target = np.array(
        [np.nan, np.nan, np.nan]
    )
    mask = np.array(
        [True, True, True]
    )

    result = regime_rate(
        target,
        mask,
    )

    assert result == {
        "n": 0,
        "events": 0,
        "event_rate": None,
    }
