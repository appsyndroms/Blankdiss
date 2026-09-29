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


def test_pipeline_runs_scan_deep_and_evaluation_in_order(
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
            (
                "research",
                specs,
                mode,
                output_dir,
            )
        )

        return f"/tmp/{mode}"

    def fake_run_evaluation(
        candidate,
        evaluation,
        *,
        output_dir,
    ):
        calls.append(
            (
                "evaluation",
                candidate,
                evaluation,
                output_dir,
            )
        )

        return "/tmp/evaluation/evaluation.json"

    monkeypatch.setattr(
        entrypoint,
        "run_research",
        fake_run_research,
    )

    monkeypatch.setattr(
        entrypoint,
        "run_evaluation",
        fake_run_evaluation,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "blankdiss-research",
            "pipeline",
            "--scan",
            "--deep",
            "--scan-spec",
            "scan.yaml",
            "--deep-spec",
            "deep.yaml",
            "--candidate",
            "candidate.yaml",
            "--evaluation",
            "evaluation.yaml",
            "--output-dir",
            "/tmp/evaluation",
        ],
    )

    assert entrypoint.main() == 0

    assert calls == [
        (
            "research",
            ["scan.yaml"],
            "scan",
            None,
        ),
        (
            "research",
            ["deep.yaml"],
            "deep",
            None,
        ),
        (
            "evaluation",
            "candidate.yaml",
            "evaluation.yaml",
            "/tmp/evaluation",
        ),
    ]


def test_pipeline_can_run_evaluation_without_scan_or_deep(
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
            (
                "research",
                specs,
                mode,
                output_dir,
            )
        )

        return "/tmp/research"

    def fake_run_evaluation(
        candidate,
        evaluation,
        *,
        output_dir,
    ):
        calls.append(
            (
                "evaluation",
                candidate,
                evaluation,
                output_dir,
            )
        )

        return "/tmp/evaluation/evaluation.json"

    monkeypatch.setattr(
        entrypoint,
        "run_research",
        fake_run_research,
    )

    monkeypatch.setattr(
        entrypoint,
        "run_evaluation",
        fake_run_evaluation,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "blankdiss-research",
            "pipeline",
            "--candidate",
            "candidate.yaml",
            "--evaluation",
            "evaluation.yaml",
        ],
    )

    assert entrypoint.main() == 0

    assert calls == [
        (
            "evaluation",
            "candidate.yaml",
            "evaluation.yaml",
            None,
        )
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
