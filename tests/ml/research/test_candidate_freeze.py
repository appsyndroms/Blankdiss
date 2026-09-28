from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ml.research.candidates.freeze import (
    freeze_candidate,
)
from ml.research.candidates.spec import (
    load_candidate,
)
from ml.research.candidates.verification import (
    candidate_fingerprint,
    verify_candidate,
)


def _write_candidate(
    path: Path,
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "candidate_freeze_test",
                "version": 1,
                "question": "Freeze test",
                "created_at": (
                    "2026-01-01T00:00:00Z"
                ),
                "discovery_cutoff": (
                    "2026-06-30T00:00:00Z"
                ),
                "freeze_at": (
                    "2026-07-15T00:00:00Z"
                ),
                "candidate_status": "draft",
                "features": [
                    {
                        "name": "price_momentum_60d"
                    }
                ],
                "parameters": {
                    "quantile": 0.10,
                },
                "target": {
                    "name": "down_10pct_5d"
                },
                "analysis": {
                    "type": "tail",
                    "bootstrap": False,
                    "bootstrap_iterations": 2000,
                },
                "training_period": {
                    "start": (
                        "2022-01-01T00:00:00Z"
                    ),
                    "end": (
                        "2026-06-30T00:00:00Z"
                    ),
                },
                "provenance": {
                    "source": "test",
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_freeze_creates_verified_frozen_candidate(
    tmp_path: Path,
):
    draft = tmp_path / "candidate.yaml"

    _write_candidate(
        draft
    )

    frozen_path = freeze_candidate(
        draft
    )

    frozen = load_candidate(
        frozen_path
    )

    assert frozen.status == "frozen"

    assert (
        frozen.fingerprint_algorithm
        == "sha256"
    )

    assert (
        frozen.fingerprint_value
        == candidate_fingerprint(frozen)
    )

    verify_candidate(
        frozen
    )


def test_existing_frozen_output_is_never_overwritten(
    tmp_path: Path,
):
    draft = tmp_path / "candidate.yaml"

    _write_candidate(
        draft
    )

    output = tmp_path / "candidate.frozen.yaml"

    freeze_candidate(
        draft,
        output,
    )

    with pytest.raises(
        FileExistsError
    ):
        freeze_candidate(
            draft,
            output,
        )


def test_freezing_already_frozen_candidate_is_rejected(
    tmp_path: Path,
):
    draft = tmp_path / "candidate.yaml"

    _write_candidate(
        draft
    )

    frozen_path = freeze_candidate(
        draft
    )

    with pytest.raises(
        ValueError,
        match="redan frozen",
    ):
        freeze_candidate(
            frozen_path
        )
