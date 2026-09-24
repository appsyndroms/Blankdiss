from __future__ import annotations

from ml.research.migration_comparator import (
    compare_incremental_si_analysis,
    compare_interaction_analysis,
    compare_mapping,
    compare_population,
    compare_semantics,
)


def test_identical_mapping_passes() -> None:
    result = {
        "metadata": {
            "target": "down_10pct_5d",
            "analysis_rows": 1000,
        },
        "metrics": {
            "event_rate": 0.123,
            "mean_return": -0.045,
        },
    }

    comparisons = compare_mapping(result, result)

    assert comparisons
    assert all(comparison.equal for comparison in comparisons)


def test_numeric_tolerance_passes() -> None:
    old = {
        "value": 0.123456789,
    }

    new = {
        "value": 0.1234567891,
    }

    comparisons = compare_mapping(old, new)

    assert len(comparisons) == 1
    assert comparisons[0].equal


def test_population_difference_fails() -> None:
    old = {
        "analysis_rows": 1000,
    }

    new = {
        "analysis_rows": 999,
    }

    comparisons = compare_population(old, new)

    assert comparisons
    assert not all(comparison.equal for comparison in comparisons)


def test_target_difference_fails() -> None:
    old = {
        "target": "down_10pct_5d",
    }

    new = {
        "target": "down_10pct_20d",
    }

    comparisons = compare_semantics(old, new)

    assert comparisons
    assert not all(comparison.equal for comparison in comparisons)


def test_interaction_identical_results_pass() -> None:
    result = {
        "metadata": {
            "target": "down_10pct_5d",
            "discovery_end": "2025-12-19",
            "tail_fractions": [
                0.01,
                0.025,
                0.05,
                0.10,
            ],
            "bootstrap_iterations": 2000,
            "bootstrap_seed": 20260920,
            "feature_rows": 10000,
            "eligible_rows": 9000,
            "analysis_rows": 9000,
            "event_count": 800,
        },
        "cells": {
            "11": {
                "n": 100,
                "event_rate": 0.30,
                "mean_return": -0.10,
            },
            "10": {
                "n": 200,
                "event_rate": 0.20,
                "mean_return": -0.06,
            },
            "01": {
                "n": 300,
                "event_rate": 0.15,
                "mean_return": -0.04,
            },
            "00": {
                "n": 400,
                "event_rate": 0.10,
                "mean_return": -0.02,
            },
        },
        "metrics": {
            "additive_interaction": 0.15,
            "relative_risk_interaction": 2.5,
        },
        "bootstrap": {
            "iterations": 2000,
            "seed": 20260920,
        },
    }

    comparisons = compare_interaction_analysis(
        result,
        result,
    )

    assert comparisons
    assert all(comparison.equal for comparison in comparisons)


def test_interaction_population_difference_fails() -> None:
    old = {
        "metadata": {
            "analysis_rows": 1000,
        }
    }

    new = {
        "metadata": {
            "analysis_rows": 999,
        }
    }

    comparisons = compare_interaction_analysis(
        old,
        new,
    )

    assert comparisons
    assert not all(comparison.equal for comparison in comparisons)


def test_interaction_tail_difference_fails() -> None:
    old = {
        "metadata": {
            "tail_fractions": [
                0.01,
                0.025,
                0.05,
                0.10,
            ]
        }
    }

    new = {
        "metadata": {
            "tail_fractions": [
                0.01,
                0.025,
                0.05,
                0.10,
                0.20,
            ]
        }
    }

    comparisons = compare_interaction_analysis(
        old,
        new,
    )

    assert comparisons
    assert not all(comparison.equal for comparison in comparisons)


def test_incremental_si_identical_results_pass() -> None:
    result = {
        "metadata": {
            "target": "down_10pct_5d",
            "discovery_end": "2025-12-19",
            "feature_rows": 10000,
            "eligible_rows": 9000,
            "analysis_rows": 9000,
        },
        "decile_analysis": {
            "volatility_group_1": {
                "decile_1": {
                    "n": 100,
                    "event_rate": 0.10,
                }
            }
        },
        "conditional_rank_analysis": {
            "spearman": 0.42,
        },
        "walk_forward_models": {
            "volatility_only": {
                "auc": 0.61,
                "log_loss": 0.62,
                "brier": 0.21,
            },
            "volatility_plus_si": {
                "auc": 0.64,
                "log_loss": 0.60,
                "brier": 0.20,
            },
            "volatility_plus_si_interaction": {
                "auc": 0.65,
                "log_loss": 0.59,
                "brier": 0.19,
            },
        },
    }

    comparisons = compare_incremental_si_analysis(
        result,
        result,
    )

    assert comparisons
    assert all(comparison.equal for comparison in comparisons)


def test_incremental_si_model_difference_fails() -> None:
    old = {
        "walk_forward_models": {
            "volatility_plus_si_interaction": {
                "auc": 0.65,
            }
        }
    }

    new = {
        "walk_forward_models": {
            "volatility_plus_si_interaction": {
                "auc": 0.60,
            }
        }
    }

    comparisons = compare_incremental_si_analysis(
        old,
        new,
    )

    assert comparisons
    assert not all(comparison.equal for comparison in comparisons)
