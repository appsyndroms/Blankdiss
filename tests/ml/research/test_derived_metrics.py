from ml.research.derived_metrics import (
    apply_derived_metrics,
)
from ml.research.spec import ResearchSpec


def _spec(
    metadata,
) -> ResearchSpec:
    return ResearchSpec(
        id="test",
        question="test",
        signals=(),
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
                    "incremental - baseline"
                )
            },
            "conditional_severity_lift": {
                "formula": (
                    "incremental / baseline"
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
        baseline["regime"]
        == "baseline"
    )

    assert (
        incremental["regime"]
        == "incremental"
    )

    assert (
        baseline["derived_metrics"][
            "p_down_10_given_down_7"
        ]
        == 0.25
    )

    assert (
        incremental["derived_metrics"][
            "p_down_10_given_down_7"
        ]
        == 0.4
    )

    assert (
        baseline["derived_metrics"][
            "conditional_severity_difference"
        ]
        == 0.15000000000000002
    )

    assert (
        baseline["derived_metrics"][
            "conditional_severity_lift"
        ]
        == 1.6
    )


def test_structured_derived_metrics_are_supported():
    metadata = {
        "derived_metrics": [
            {
                "name": "severity",
                "type": (
                    "conditional_event_rate"
                ),
                "numerator_target": (
                    "down_10pct_5d"
                ),
                "denominator_target": (
                    "down_7pct_5d"
                ),
            },
            {
                "name": (
                    "severity_difference"
                ),
                "type": "rate_difference",
                "left": (
                    "incremental.severity"
                ),
                "right": (
                    "baseline.severity"
                ),
            },
            {
                "name": "severity_lift",
                "type": "rate_ratio",
                "numerator": (
                    "incremental.severity"
                ),
                "denominator": (
                    "baseline.severity"
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
        ]["severity"]
        == 0.25
    )

    assert (
        output[2][
            "derived_metrics"
        ]["severity"]
        == 0.4
    )

    assert (
        output[0][
            "derived_metrics"
        ]["severity_difference"]
        == 0.15000000000000002
    )

    assert (
        output[0][
            "derived_metrics"
        ]["severity_lift"]
        == 1.6
    )
