from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.research import runner


def _write_spec(
    path: Path,
    *,
    spec_id: str,
    mode: str,
) -> None:
    path.write_text(
        f"""
id: {spec_id}
question: Test {spec_id}
mode: {mode}

signals:
  - name: price_momentum_60d
    direction: upper
    bins:
      - 0.10

targets:
  - down_10pct_5d

analysis:
  type: tail
  bootstrap: false
  bootstrap_iterations: 100

windows:
  - window_1

splits:
  - test
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_run_research_filters_default_specs_by_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()

    scan_path = spec_dir / "scan.yaml"
    deep_path = spec_dir / "deep.yaml"

    _write_spec(
        scan_path,
        spec_id="scan_spec",
        mode="scan",
    )

    _write_spec(
        deep_path,
        spec_id="deep_spec",
        mode="deep",
    )

    output_dir = tmp_path / "output"

    monkeypatch.setattr(
        runner,
        "DEFAULT_SPEC_DIR",
        spec_dir,
    )

    monkeypatch.setattr(
        runner,
        "build_session",
        lambda specs: type(
            "Session",
            (),
            {
                "frame": [],
                "cache": object(),
            },
        )(),
    )

    executed: list[str] = []

    def fake_run_spec(cache, spec):
        executed.append(spec.id)

        return {
            "spec_id": spec.id,
            "question": spec.question,
            "mode": spec.mode,
            "analysis": spec.analysis.type,
            "results": [],
        }

    monkeypatch.setattr(
        runner,
        "run_spec",
        fake_run_spec,
    )

    monkeypatch.setattr(
        runner,
        "write_json",
        lambda path, value: (
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            ),
            path.write_text(
                json.dumps(value),
                encoding="utf-8",
            ),
        )[-1],
    )

    runner.run_research(
        mode="scan",
        output_dir=output_dir,
    )

    assert executed == [
        "scan_spec",
    ]


def test_run_research_deep_only_runs_deep_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()

    scan_path = spec_dir / "scan.yaml"
    deep_path = spec_dir / "deep.yaml"

    _write_spec(
        scan_path,
        spec_id="scan_spec",
        mode="scan",
    )

    _write_spec(
        deep_path,
        spec_id="deep_spec",
        mode="deep",
    )

    output_dir = tmp_path / "output"

    monkeypatch.setattr(
        runner,
        "DEFAULT_SPEC_DIR",
        spec_dir,
    )

    monkeypatch.setattr(
        runner,
        "build_session",
        lambda specs: type(
            "Session",
            (),
            {
                "frame": [],
                "cache": object(),
            },
        )(),
    )

    executed: list[str] = []

    def fake_run_spec(cache, spec):
        executed.append(spec.id)

        return {
            "spec_id": spec.id,
            "question": spec.question,
            "mode": spec.mode,
            "analysis": spec.analysis.type,
            "results": [],
        }

    monkeypatch.setattr(
        runner,
        "run_spec",
        fake_run_spec,
    )

    monkeypatch.setattr(
        runner,
        "write_json",
        lambda path, value: (
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            ),
            path.write_text(
                json.dumps(value),
                encoding="utf-8",
            ),
        )[-1],
    )

    runner.run_research(
        mode="deep",
        output_dir=output_dir,
    )

    assert executed == [
        "deep_spec",
    ]


def test_explicit_specs_must_match_requested_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    spec_path = tmp_path / "scan.yaml"

    _write_spec(
        spec_path,
        spec_id="scan_spec",
        mode="scan",
    )

    monkeypatch.setattr(
        runner,
        "build_session",
        lambda specs: type(
            "Session",
            (),
            {
                "frame": [],
                "cache": object(),
            },
        )(),
    )

    with pytest.raises(
        ValueError,
        match="fel mode",
    ):
        runner.run_research(
            [spec_path],
            mode="deep",
            output_dir=tmp_path / "output",
        )


def test_duplicate_spec_ids_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"

    _write_spec(
        first,
        spec_id="duplicate",
        mode="scan",
    )

    _write_spec(
        second,
        spec_id="duplicate",
        mode="scan",
    )

    monkeypatch.setattr(
        runner,
        "build_session",
        lambda specs: type(
            "Session",
            (),
            {
                "frame": [],
                "cache": object(),
            },
        )(),
    )

    with pytest.raises(
        ValueError,
        match="Duplicate research spec id",
    ):
        runner.run_research(
            [first, second],
            mode="scan",
            output_dir=tmp_path / "output",
        )


def test_run_research_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    spec_path = tmp_path / "scan.yaml"

    _write_spec(
        spec_path,
        spec_id="manifest_test",
        mode="scan",
    )

    output_dir = tmp_path / "output"

    monkeypatch.setattr(
        runner,
        "build_session",
        lambda specs: type(
            "Session",
            (),
            {
                "frame": [1, 2, 3],
                "cache": object(),
            },
        )(),
    )

    monkeypatch.setattr(
        runner,
        "run_spec",
        lambda cache, spec: {
            "spec_id": spec.id,
            "question": spec.question,
            "mode": spec.mode,
            "analysis": spec.analysis.type,
            "results": [
                {
                    "n": 10,
                }
            ],
        },
    )

    written: dict[str, object] = {}

    def fake_write_json(path, value):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(value),
            encoding="utf-8",
        )

        written[str(path)] = value

    monkeypatch.setattr(
        runner,
        "write_json",
        fake_write_json,
    )

    run_dir = runner.run_research(
        [spec_path],
        mode="scan",
        output_dir=output_dir,
    )

    manifest_path = (
        run_dir / "manifest.json"
    )

    assert manifest_path.exists()

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert manifest["mode"] == "scan"
    assert manifest["feature_rows"] == 3

    assert manifest["specs"][0]["id"] == (
        "manifest_test"
    )

    assert manifest["specs"][0]["mode"] == (
        "scan"
    )
