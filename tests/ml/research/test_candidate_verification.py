from __future__ import annotations

from dataclasses import replace

import pytest

from research.candidates.spec import (
    CandidateFeature,
    CandidatePeriod,
    CandidateSpec,
    CandidateTarget,
)
from research.candidates.verification import (
    candidate_fingerprint,
    candidate_snapshot,
    verify_candidate_fingerprint,
    verify_candidate_frozen,
    verify_candidate_identity,
)


def make_candidate(
    *,
    status: str = "frozen",
) -> CandidateSpec:
    return CandidateSpec(
        id="candidate_test_001",
        version=1,
        question="Test candidate",
        created_at="2026-01-01T00:00:00Z",
        discovery_cutoff="2026-06-30T00:00:00Z",
        freeze_at="2026-07-15T00:00:00Z",
        status=status,
        features=(
            CandidateFeature(
                name="price_momentum_60d"
            ),
            CandidateFeature(
                name="short_interest_change"
            ),
        ),
        parameters={
            "momentum_quantile": 0.10,
            "short_interest_quantile": 0.10,
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


def test_frozen_candidate_is_accepted():
    candidate = make_candidate()

    verify_candidate_frozen(
        candidate
    )


def test_non_frozen_candidate_is_rejected():
    candidate = make_candidate(
        status="tested"
    )

    with pytest.raises(
        ValueError,
        match="frozen",
    ):
        verify_candidate_frozen(
            candidate
        )


def test_identity_is_verified():
    candidate = make_candidate()

    verify_candidate_identity(
        candidate,
        expected_id="candidate_test_001",
        expected_version=1,
    )


def test_wrong_candidate_id_is_rejected():
    candidate = make_candidate()

    with pytest.raises(
        ValueError,
        match="identity",
    ):
        verify_candidate_identity(
            candidate,
            expected_id="candidate_other",
            expected_version=1,
        )


def test_wrong_candidate_version_is_rejected():
    candidate = make_candidate()

    with pytest.raises(
        ValueError,
        match="version",
    ):
        verify_candidate_identity(
            candidate,
            expected_id="candidate_test_001",
            expected_version=2,
        )


def test_same_candidate_has_same_fingerprint():
    first = make_candidate()
    second = make_candidate()

    assert (
        candidate_fingerprint(first)
        == candidate_fingerprint(second)
    )


def test_parameter_change_changes_fingerprint():
    original = make_candidate()

    changed = replace(
        original,
        parameters={
            "momentum_quantile": 0.15,
            "short_interest_quantile": 0.10,
        },
    )

    assert (
        candidate_fingerprint(original)
        != candidate_fingerprint(changed)
    )


def test_changed_candidate_fails_fingerprint_verification():
    original = make_candidate()

    fingerprint = candidate_fingerprint(
        original
    )

    changed = replace(
        original,
        parameters={
            "momentum_quantile": 0.15,
            "short_interest_quantile": 0.10,
        },
    )

    with pytest.raises(
        ValueError,
        match="fingerprint",
    ):
        verify_candidate_fingerprint(
            changed,
            fingerprint,
        )


def test_candidate_snapshot_contains_identity_and_fingerprint():
    candidate = make_candidate()

    snapshot = candidate_snapshot(
        candidate
    )

    assert snapshot["candidate_id"] == (
        "candidate_test_001"
    )

    assert snapshot["candidate_version"] == 1

    assert snapshot[
        "candidate_fingerprint"
    ] == candidate_fingerprint(
        candidate
    )

    assert snapshot[
        "candidate"
    ]["parameters"]["momentum_quantile"] == 0.10
