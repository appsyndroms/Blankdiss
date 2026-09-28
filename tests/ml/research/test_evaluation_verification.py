from __future__ import annotations

from dataclasses import replace

import pytest

from ml.research.candidates.spec import (
    CandidateAnalysis,
    CandidateFeature,
    CandidatePeriod,
    CandidateSpec,
    CandidateTarget,
)
from ml.research.candidates.verification import (
    candidate_fingerprint,
)
from ml.research.evaluation.spec import (
    EvaluationPeriod,
    EvaluationSpec,
    WalkForwardSpec,
    WalkForwardWindow,
)
from ml.research.evaluation.verification import (
    verify_evaluation,
    verify_temporal_separation,
    verify_walk_forward_window,
)


def make_candidate() -> CandidateSpec:
    candidate = CandidateSpec(
        schema_version=1,
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
        analysis=CandidateAnalysis(
            type="tail",
            bootstrap=True,
            bootstrap_iterations=2000,
        ),
        training_period=CandidatePeriod(
            start="2022-01-01T00:00:00Z",
            end="2026-06-30T00:00:00Z",
        ),
        provenance={
            "source": "test",
        },
        fingerprint_algorithm=None,
        fingerprint_value=None,
    )

    return replace(
        candidate,
        fingerprint_algorithm="sha256",
        fingerprint_value=(
            candidate_fingerprint(
                candidate
            )
        ),
    )


def make_evaluation(
    *,
    start: str = "2026-07-16T00:00:00Z",
    end: str = "2026-09-30T00:00:00Z",
    candidate_id: str = "candidate_test_001",
    candidate_version: int = 1,
    candidate_fingerprint: str | None = None,
    walk_forward: WalkForwardSpec | None = None,
) -> EvaluationSpec:
    if walk_forward is None:
        walk_forward = WalkForwardSpec(
            enabled=False,
            windows=(),
        )

    if candidate_fingerprint is None:
        candidate_fingerprint = (
            candidate_fingerprint_for_test()
        )

    return EvaluationSpec(
        schema_version=1,
        id="evaluation_test_001",
        version=1,
        candidate_id=candidate_id,
        candidate_version=candidate_version,
        candidate_fingerprint=(
            candidate_fingerprint
        ),
        evaluation_period=EvaluationPeriod(
            start=start,
            end=end,
        ),
        metrics=(
            "sample_size",
            "mean_return",
        ),
        walk_forward=walk_forward,
        metadata={},
    )


def candidate_fingerprint_for_test() -> str:
    return candidate_fingerprint(
        make_candidate()
    )


def test_future_evaluation_is_accepted():
    candidate = make_candidate()

    evaluation = make_evaluation(
        candidate_fingerprint=(
            candidate.fingerprint_value
        )
    )

    verify_evaluation(
        candidate,
        evaluation,
    )


def test_evaluation_before_freeze_is_rejected():
    candidate = make_candidate()

    evaluation = make_evaluation(
        start="2026-07-01T00:00:00Z",
        end="2026-07-14T00:00:00Z",
        candidate_fingerprint=(
            candidate.fingerprint_value
        ),
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
        candidate_fingerprint=(
            candidate.fingerprint_value
        ),
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
        candidate_fingerprint=(
            candidate.fingerprint_value
        ),
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
        candidate_fingerprint=(
            candidate.fingerprint_value
        ),
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
        candidate_id="candidate_other_001",
        candidate_fingerprint=(
            candidate.fingerprint_value
        ),
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
        candidate_version=2,
        candidate_fingerprint=(
            candidate.fingerprint_value
        ),
    )

    with pytest.raises(
        ValueError,
        match="candidate_version",
    ):
        verify_evaluation(
            candidate,
            evaluation,
        )


def test_wrong_candidate_fingerprint_is_rejected():
    candidate = make_candidate()

    evaluation = make_evaluation(
        candidate_fingerprint=(
            "0" * 64
        ),
    )

    with pytest.raises(
        ValueError,
        match="fingerprint",
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


def test_walk_forward_must_fit_evaluation_period():
    candidate = make_candidate()

    evaluation = make_evaluation(
        start="2026-07-16T00:00:00Z",
        end="2026-08-31T00:00:00Z",
        candidate_fingerprint=(
            candidate.fingerprint_value
        ),
    )

    with pytest.raises(
        ValueError,
        match="inom evaluation_period",
    ):
        verify_walk_forward_window(
            candidate,
            "2026-09-01T00:00:00Z",
            "2026-09-10T00:00:00Z",
            evaluation,
        )


def test_overlapping_walk_forward_windows_are_rejected():
    candidate = make_candidate()

    evaluation = make_evaluation(
        end="2026-09-30T00:00:00Z",
        candidate_fingerprint=(
            candidate.fingerprint_value
        ),
        walk_forward=WalkForwardSpec(
            enabled=True,
            windows=(
                WalkForwardWindow(
                    name="window_1",
                    start="2026-07-16T00:00:00Z",
                    end="2026-08-15T00:00:00Z",
                ),
                WalkForwardWindow(
                    name="window_2",
                    start="2026-08-15T00:00:00Z",
                    end="2026-09-30T00:00:00Z",
                ),
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="överlappa",
    ):
        verify_evaluation(
            candidate,
            evaluation,
        )


def test_enabled_walk_forward_requires_windows():
    evaluation = EvaluationSpec(
        schema_version=1,
        id="evaluation_test_002",
        version=1,
        candidate_id="candidate_test_001",
        candidate_version=1,
        candidate_fingerprint="0" * 64,
        evaluation_period=EvaluationPeriod(
            start="2026-07-16T00:00:00Z",
            end="2026-09-30T00:00:00Z",
        ),
        metrics=(
            "sample_size",
        ),
        walk_forward=WalkForwardSpec(
            enabled=True,
            windows=(),
        ),
        metadata={},
    )

    assert evaluation.walk_forward.enabled is True
    assert evaluation.walk_forward.windows == ()
