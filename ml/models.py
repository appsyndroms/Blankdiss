from ml.research.derived_metrics import (
    apply_derived_metrics,
)
from ml.research.spec import (
    ResearchSpec,
    SignalSpec,
)


def _spec(
    metadata,
) -> ResearchSpec:
    return ResearchSpec(
        id="test",
        question="test",
        signals=(
            SignalSpec(
                name="test_signal",
            ),
        ),
        targets=(
            "down_7pct_5d",
            "down_10pct_5d",
        ),
        metadata=metadata,
    )


def _results():
    return [
        {
            "analysis": (
                "nested_regime_comparison"
            ),
            "window": "window_1",
            "split": "test",
            "target": "down_7pct_5d",
            "baseline_signals": [],
            "events": 100,
            "n": 1000,
        },
        {
            "analysis": (
                "nested_regime_comparison"
            ),
            "window": "window_1",
            "split": "test",
            "target": "down_10pct_5d",
            "baseline_signals": [],
            "events": 25,
            "n": 1000,
        },
        {
            "analysis": (
                "nested_regime_comparison"
            ),
            "window": "window_1",
            "split": "test",
            "target": "down_7pct_5d",
            "incremental_signal": {},
            "events": 120,
            "n": 500,
        },
        {
            "analysis": (
                "nested_regime_comparison"
            ),
            "window": "window_1",
            "split": "test",
            "target": "down_10pct_5d",
            "incremental_signal": {},
            "events": 48,
            "n": 500,
        },
    ]


def test_no_derived_metrics_keeps_existing_results_unchanged():
    results = _results()

    output = apply_derived_metrics(
        _spec({}),
        results,
    )

    assert output == results


def test_legacy_severity_yaml_shape_is_supported():
    metadata = {
        "evaluation": {
            "severity": {
                "conditioning_target": (
                    "down_7pct_5d"
                ),
                "target": (
                    "down_10pct_5d"
                ),
            }
        },
        "derived_metrics": {
            "p_down_10_given_down_7": {
                "formula": (
                    "down_10pct_5d_events / "
                    "down_7pct_5d_events"
                )
            },
            "conditional_severity_difference": {
                "formula": (
                    "p_down_10_given_down_7_incremental "
                    "- p_down_10_given_down_7_baseline"
                )
            },
            "conditional_severity_lift": {
                "formula": (
                    "p_down_10_given_down_7_incremental "
                    "/ p_down_10_given_down_7_baseline"
                )
            },
        },
    }

    output = apply_derived_metrics(
        _spec(metadata),
        _results(),
    )

    baseline = output[0]
    incremental = output[2]

    assert (
        baseline["derived_metrics"][
            "baseline"
        ]["p_down_10_given_down_7"]
        == 0.25
    )

    assert (
        incremental["derived_metrics"][
            "incremental"
        ]["p_down_10_given_down_7"]
        == 0.4
    )

    assert (
        baseline["derived_metrics"][
            "result"
        ]["conditional_severity_difference"]
        == 0.15000000000000002
    )

    assert (
        baseline["derived_metrics"][
            "result"
        ]["conditional_severity_lift"]
        == 1.6
    )


def test_structured_derived_metrics_are_supported():
    metadata = {
        "derived_metrics": [
            {
                "name": "severity",
                "formula": (
                    "down_10pct_5d_events / "
                    "down_7pct_5d_events"
                ),
            },
            {
                "name": (
                    "severity_difference"
                ),
                "formula": (
                    "severity_incremental "
                    "- severity_baseline"
                ),
            },
            {
                "name": "severity_lift",
                "formula": (
                    "severity_incremental "
                    "/ severity_baseline"
                ),
            },
        ]
    }

    output = apply_derived_metrics(
        _spec(metadata),
        _results(),
    )

    assert (
        output[0][
            "derived_metrics"
        ]["baseline"]["severity"]
        == 0.25
    )

    assert (
        output[2][
            "derived_metrics"
        ]["incremental"]["severity"]
        == 0.4
    )

    assert (
        output[0][
            "derived_metrics"
        ]["result"]["severity_difference"]
        == 0.15000000000000002
    )

    assert (
        output[0][
            "derived_metrics"
        ]["result"]["severity_lift"]
        == 1.6
    )
