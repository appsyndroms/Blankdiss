from __future__ import annotations

import pytest

from ml.research import entrypoint


def test_scan_command_passes_scan_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def fake_run_research(
        specs,
        *,
        mode,
        output_dir,
    ):
        calls.append(
            {
                "specs": specs,
                "mode": mode,
                "output_dir": output_dir,
            }
        )

        return "/tmp/scan"


    monkeypatch.setattr(
        entrypoint,
        "run_research",
        fake_run_research,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "blankdiss-research",
            "scan",
            "scan.yaml",
            "--output-dir",
            "/tmp/output",
        ],
    )

    assert entrypoint.main() == 0

    assert calls == [
        {
            "specs": ["scan.yaml"],
            "mode": "scan",
            "output_dir": "/tmp/output",
        }
    ]


def test_deep_command_passes_deep_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def fake_run_research(
        specs,
        *,
        mode,
        output_dir,
    ):
        calls.append(
            {
                "specs": specs,
                "mode": mode,
                "output_dir": output_dir,
            }
        )

        return "/tmp/deep"

    monkeypatch.setattr(
        entrypoint,
        "run_research",
        fake_run_research,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "blankdiss-research",
            "deep",
            "deep.yaml",
        ],
    )

    assert entrypoint.main() == 0

    assert calls == [
        {
            "specs": ["deep.yaml"],
            "mode": "deep",
            "output_dir": None,
        }
    ]


def test_pipeline_requires_candidate(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "blankdiss-research",
            "pipeline",
        ],
    )

    assert entrypoint.main() == 1


def test_pipeline_requires_evaluation(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "blankdiss-research",
            "pipeline",
            "--candidate",
            "candidate.yaml",
        ],
    )

    assert entrypoint.main() == 1
