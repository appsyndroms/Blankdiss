from __future__ import annotations

from ml.research.walk_forward import (
    aggregate_walk_forward,
)


def test_walk_forward_aggregation_keeps_windows_separate():
    results = [
        {
            "window": {
                "name": "window_1",
            },
            "results": [
                {
                    "n": 100,
                    "events": 20,
                    "event_rate": 0.20,
                    "lift": 1.5,
                }
            ],
        },
        {
            "window": {
                "name": "window_2",
            },
            "results": [
                {
                    "n": 200,
                    "events": 30,
                    "event_rate": 0.15,
                    "lift": 1.2,
                }
            ],
        },
    ]

    aggregated = aggregate_walk_forward(
        results
    )

    assert aggregated[
        "window_count"
    ] == 2

    assert aggregated[
        "result_count"
    ] == 2

    assert [
        window["name"]
        for window in aggregated["windows"]
    ] == [
        "window_1",
        "window_2",
    ]

    assert (
        aggregated["windows"][0]["metrics"]["n"]["mean"]
        == 100.0
    )

    assert (
        aggregated["windows"][1]["metrics"]["n"]["mean"]
        == 200.0
    )


def test_walk_forward_reports_window_stability():
    results = [
        {
            "window": {
                "name": "window_1",
            },
            "results": [
                {
                    "event_rate": 0.20,
                    "lift": 1.5,
                    "mean_return": 0.10,
                }
            ],
        },
        {
            "window": {
                "name": "window_2",
            },
            "results": [
                {
                    "event_rate": 0.15,
                    "lift": 1.2,
                    "mean_return": 0.05,
                }
            ],
        },
    ]

    aggregated = aggregate_walk_forward(
        results
    )

    stability = aggregated[
        "windows"
    ][0]["stability"]

    assert (
        stability["lift"]["positive"] == 1
    )

    assert (
        stability["mean_return"]["positive"]
        == 1
    )


def test_walk_forward_reports_cross_window_baseline_consistency():
    results = [
        {
            "window": {
                "name": "window_1",
            },
            "results": [
                {
                    "auc": 0.60,
                    "lift": 1.50,
                    "mean_return": 0.10,
                }
            ],
        },
        {
            "window": {
                "name": "window_2",
            },
            "results": [
                {
                    "auc": 0.55,
                    "lift": 1.20,
                    "mean_return": 0.05,
                }
            ],
        },
        {
            "window": {
                "name": "window_3",
            },
            "results": [
                {
                    "auc": 0.48,
                    "lift": 0.90,
                    "mean_return": -0.02,
                }
            ],
        },
    ]

    aggregated = aggregate_walk_forward(
        results
    )

    stability = aggregated[
        "overall"
    ]["stability"]

    baseline = aggregated[
        "overall"
    ]["baseline_consistency"]

    assert (
        stability["lift"]["above"] == 2
    )

    assert (
        stability["lift"]["below"] == 1
    )

    assert (
        stability["mean_return"]["above"]
        == 2
    )

    assert (
        stability["mean_return"]["below"]
        == 1
    )

    assert (
        baseline["auc"]["above"] == 2
    )

    assert (
        baseline["auc"]["below"] == 1
    )

    assert (
        baseline["lift"]["above"] == 2
    )

    assert (
        baseline["lift"]["below"] == 1
    )

    assert (
        baseline["mean_return"]["above"]
        == 2
    )

    assert (
        baseline["mean_return"]["below"]
        == 1
    )


def test_walk_forward_uses_window_means_for_cross_window_analysis():
    results = [
        {
            "window": {
                "name": "window_1",
            },
            "results": [
                {
                    "auc": 0.60,
                    "lift": 1.20,
                    "mean_return": 0.10,
                },
                {
                    "auc": 0.80,
                    "lift": 1.60,
                    "mean_return": 0.20,
                },
            ],
        },
        {
            "window": {
                "name": "window_2",
            },
            "results": [
                {
                    "auc": 0.40,
                    "lift": 0.80,
                    "mean_return": -0.10,
                },
            ],
        },
    ]

    aggregated = aggregate_walk_forward(
        results
    )

    baseline = aggregated[
        "overall"
    ]["baseline_consistency"]

    assert (
        baseline["auc"]["above"] == 1
    )

    assert (
        baseline["auc"]["below"] == 1
    )

    assert (
        baseline["lift"]["above"] == 1
    )

    assert (
        baseline["lift"]["below"] == 1
    )

    assert (
        baseline["mean_return"]["above"] == 1
    )

    assert (
        baseline["mean_return"]["below"] == 1
    )
