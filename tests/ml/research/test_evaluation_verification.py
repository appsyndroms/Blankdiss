from __future__ import annotations

import pytest

from research.candidates.spec import (
    CandidateFeature,
    CandidatePeriod,
    CandidateSpec,
    CandidateTarget,
)
from research.evaluation.spec import (
    EvaluationPeriod,
    EvaluationSpec,
    WalkForwardSpec,
)
from research.evaluation.verification import (
    verify_evaluation,
    verify_temporal_separation,
    verify_walk_forward_window,
)


def make_candidate() -> CandidateSpec:
    return CandidateSpec(
        id="candidate_test_001",
        version=1,
        question="Test candidate",
        created_at="2026-01-01T00:00:00Z",
        discovery_cutoff="2026-06-30T00:00:00Z",
        freeze_at="2026-07-15T00:00:00Z",
        status="frozen",
        features=(
            CandidateFeature(
                name="price_momentum_60d"
            ),
        ),
        parameters={
            "quantile": 0.10,
        },
        target=CandidateTarget(
            name="forward_return_20d"
        ),
        training_period=CandidatePeriod(
            start="2022-01-01T00:00:00Z",
            end="2026-06-30T00:00:00Z",
        ),
        provenance={
            "source": "test",
        },
    )


def make_evaluation(
    *,
    start: str = "2026-07-16T00:00:00Z",
    end: str = "2026-09-30T00:00:00Z",
    candidate_id: str = "candidate_test_001",
    candidate_version: int = 1,
    targets: tuple[str, ...] = (
        "forward_return_20d",
    ),
    walk_forward: WalkForwardSpec | None = None,
) -> EvaluationSpec:
    if walk_forward is None:
        walk_forward = WalkForwardSpec(
            enabled=False,
            windows=(),
        )

    return EvaluationSpec(
        id="evaluation_test_001",
        version=1,
        candidate_id=candidate_id,
        candidate_version=candidate_version,
        evaluation_period=EvaluationPeriod(
            start=start,
            end=end,
        ),
        targets=targets,
        metrics=(
            "sample_size",
            "mean_return",
        ),
        walk_forward=walk_forward,
        metadata={},
    )


def test_future_evaluation_is_accepted():
    candidate = make_candidate()
    evaluation = make_evaluation()

    verify_evaluation(
        candidate,
        evaluation,
    )


def test_evaluation_before_freeze_is_rejected():
    candidate = make_candidate()

    evaluation = make_evaluation(
        start="2026-07-01T00:00:00Z",
        end="2026-07-14T00:00:00Z",
    )

    with pytest.raises(
        ValueError,
        match="Temporal separation",
    ):
        verify_temporal_separation(
            candidate,
            evaluation,
        )


def test_evaluation_starting_at_freeze_is_rejected():
    candidate = make_candidate()

    evaluation = make_evaluation(
        start="2026-07-15T00:00:00Z",
        end="2026-08-01T00:00:00Z",
    )

    with pytest.raises(
        ValueError,
        match="Temporal separation",
    ):
        verify_temporal_separation(
            candidate,
            evaluation,
        )


def test_evaluation_after_freeze_is_accepted():
    candidate = make_candidate()

    evaluation = make_evaluation(
        start="2026-07-15T00:00:01Z",
        end="2026-08-01T00:00:00Z",
    )

    verify_temporal_separation(
        candidate,
        evaluation,
    )


def test_reversed_evaluation_period_is_rejected():
    candidate = make_candidate()

    evaluation = make_evaluation(
        start="2026-09-30T00:00:00Z",
        end="2026-07-16T00:00:00Z",
    )

    with pytest.raises(
        ValueError,
        match="slut ligger före",
    ):
        verify_temporal_separation(
            candidate,
            evaluation,
        )


def test_wrong_candidate_id_is_rejected():
    candidate = make_candidate()

    evaluation = make_evaluation(
        candidate_id="candidate_other_001"
    )

    with pytest.raises(
        ValueError,
        match="candidate_id",
    ):
        verify_evaluation(
            candidate,
            evaluation,
        )


def test_wrong_candidate_version_is_rejected():
    candidate = make_candidate()

    evaluation = make_evaluation(
        candidate_version=2
    )

    with pytest.raises(
        ValueError,
        match="candidate_version",
    ):
        verify_evaluation(
            candidate,
            evaluation,
        )


def test_missing_candidate_target_is_rejected():
    candidate = make_candidate()

    evaluation = make_evaluation(
        targets=(
            "another_target",
        )
    )

    with pytest.raises(
        ValueError,
        match="target",
    ):
        verify_evaluation(
            candidate,
            evaluation,
        )


def test_walk_forward_before_freeze_is_rejected():
    candidate = make_candidate()

    with pytest.raises(
        ValueError,
        match="candidate.freeze_at",
    ):
        verify_walk_forward_window(
            candidate,
            "2026-07-01T00:00:00Z",
            "2026-07-31T00:00:00Z",
        )


def test_walk_forward_after_freeze_is_accepted():
    candidate = make_candidate()

    verify_walk_forward_window(
        candidate,
        "2026-07-16T00:00:00Z",
        "2026-08-31T00:00:00Z",
    )


def test_enabled_walk_forward_requires_windows():
    evaluation = EvaluationSpec(
        id="evaluation_test_002",
        version=1,
        candidate_id="candidate_test_001",
        candidate_version=1,
        evaluation_period=EvaluationPeriod(
            start="2026-07-16T00:00:00Z",
            end="2026-09-30T00:00:00Z",
        ),
        targets=("forward_return_20d",),
        metrics=("sample_size",),
        walk_forward=WalkForwardSpec(
            enabled=True,
            windows=(),
        ),
        metadata={},
    )

    assert evaluation.walk_forward.enabled is True
    assert evaluation.walk_forward.windows == ()
