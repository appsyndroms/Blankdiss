from __future__ import annotations

import numpy as np

from ml.research.cache import (
    ResearchCache,
    _tail_key,
)
from ml.research.evaluator import evaluate_experiment
from ml.research.experiments import Experiment
from ml.research.spec import (
    AnalysisSpec,
    ResearchSpec,
    SignalSpec,
)
from ml.research.engine import run_spec


class _TargetConfig:
    return_column = "forward_return_5d"


def _build_cache() -> ResearchCache:
    signal = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        dtype=float,
    )

    target = np.array(
        [0.0, 1.0, 0.0, 1.0, 1.0],
        dtype=float,
    )

    returns = np.array(
        [0.01, 0.02, -0.03, 0.04, 0.05],
        dtype=float,
    )

    upper_tail = np.array(
        [False, False, False, True, True],
        dtype=bool,
    )

    lower_tail = np.array(
        [True, True, False, False, False],
        dtype=bool,
    )

    window = np.array(
        [True, True, True, True, True],
        dtype=bool,
    )

    return ResearchCache(
        signals={
            "test_signal": signal,
        },
        signal_ranks={
            "test_signal": np.array(
                [0.2, 0.4, 0.6, 0.8, 1.0],
                dtype=float,
            ),
        },
        targets={
            "test_target": target,
        },
        returns={
            "forward_return_5d": returns,
        },
        tail_masks={
            _tail_key(
                "test_signal",
                "upper",
                0.40,
            ): upper_tail,
            _tail_key(
                "test_signal",
                "lower",
                0.40,
            ): lower_tail,
        },
        window_masks={
            "window_1": {
                "test": window,
            },
        },
        target_configs={
            "test_target": _TargetConfig(),
        },
    )


def _run_old(
    cache: ResearchCache,
    direction: str,
) -> dict:
    experiment = Experiment(
        experiment_id=(
            f"test_signal__{direction}"
            "__40pct__test_target"
        ),
        signal_name="test_signal",
        target_name="test_target",
        tail_fraction=0.40,
        tail_direction=direction,
    )

    return evaluate_experiment(
        frame=None,
        cache=cache,
        experiment=experiment,
        window_name="window_1",
        split_name="test",
    )


def _run_new(
    cache: ResearchCache,
    direction: str,
) -> dict:
    spec = ResearchSpec(
        id="migration_test",
        question="test",
        signals=(
            SignalSpec(
                name="test_signal",
                direction=direction,
                bins=(0.40,),
            ),
        ),
        targets=("test_target",),
        analysis=AnalysisSpec(
            type="tail",
            bootstrap=False,
        ),
        windows=("window_1",),
        splits=("test",),
    )

    result = run_spec(
        cache,
        spec,
    )

    return result["results"][0]


def _assert_common_metrics_match(
    old: dict,
    new: dict,
) -> None:
    assert old["n"] == new["n"]
    assert old["events"] == new["events"]

    assert np.isclose(
        old["event_rate"],
        new["event_rate"],
    )

    assert np.isclose(
        old["lift"],
        new["lift"],
    )

    assert np.isclose(
        old["mean_return"],
        new["mean_return"],
    )


def test_upper_tail_matches_old_and_new() -> None:
    cache = _build_cache()

    old = _run_old(
        cache,
        "upper",
    )

    new = _run_new(
        cache,
        "upper",
    )

    _assert_common_metrics_match(
        old,
        new,
    )


def test_lower_tail_matches_old_and_new() -> None:
    cache = _build_cache()

    old = _run_old(
        cache,
        "lower",
    )

    new = _run_new(
        cache,
        "lower",
    )

    _assert_common_metrics_match(
        old,
        new,
    )


def test_new_engine_adds_return_difference() -> None:
    cache = _build_cache()

    new = _run_new(
        cache,
        "upper",
    )

    assert "return_difference" in new

    selected_returns = np.array(
        [0.04, 0.05],
        dtype=float,
    )

    rest_returns = np.array(
        [0.01, 0.02, -0.03],
        dtype=float,
    )

    expected = (
        selected_returns.mean()
        - rest_returns.mean()
    )

    assert np.isclose(
        new["return_difference"],
        expected,
    )
