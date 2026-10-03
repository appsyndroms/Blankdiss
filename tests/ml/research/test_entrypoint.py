def test_pipeline_uses_config_when_candidate_and_evaluation_are_omitted(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def fake_load_prospective_evaluation_config():
        return (
            True,
            "/tmp/candidate.yaml",
            "/tmp/evaluation.yaml",
        )

    def fake_run_evaluation(
        candidate,
        evaluation,
        *,
        output_dir,
    ):
        calls.append(
            (
                candidate,
                evaluation,
                output_dir,
            )
        )

        return "/tmp/evaluation/evaluation.json"

    monkeypatch.setattr(
        entrypoint,
        "_load_prospective_evaluation_config",
        fake_load_prospective_evaluation_config,
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
        ],
    )

    assert entrypoint.main() == 0

    assert calls == [
        (
            "/tmp/candidate.yaml",
            "/tmp/evaluation.yaml",
            None,
        )
    ]
