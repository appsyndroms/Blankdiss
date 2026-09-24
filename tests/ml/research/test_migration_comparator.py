from __future__ import annotations
import sys
from pathlib import Path
import pytest
# Make the repository root importable when pytest is executed
# from GitHub Actions or another environment where the root is
# not automatically placed on sys.path.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ml.research.migration_comparator import (  # noqa: E402
    _build_migration_spec,
    _compare,
    _experiment_id,
    _fraction_name,
    _same_value,
)
def test_fraction_name() -> None:
    assert _fraction_name(0.20) == "20pct"
    assert _fraction_name(0.10) == "10pct"
    assert _fraction_name(0.05) == "5pct"
    assert _fraction_name(0.025) == "2_5pct"
    assert _fraction_name(0.01) == "1pct"
def test_experiment_id() -> None:
    assert (
        _experiment_id(
            "short_interest_level",
            "upper",
            0.10,
            "up_5pct_5d",
        )
        == (
            "short_interest_level"
            "__upper"
            "__10pct"
            "__up_5pct_5d"
        )
    )
@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (1, 1, True),
        (1.0, 1.0, True),
        (1.0, 1.0 + 1e-13, True),
        (1.0, 1.0 + 1e-8, False),
        (None, None, True),
        (None, 1.0, False),
        ("x", "x", True),
        ("x", "y", False),
    ],
)
def test_same_value(
    old,
    new,
    expected: bool,
) -> None:
    assert _same_value(
        old,
        new,
    ) is expected
def test_build_migration_spec() -> None:
    spec = _build_migration_spec()
    assert spec.id == (
        "generic_migration_comparison"
    )
    assert spec.mode == "scan"
    assert spec.analysis.type == "tail"
    assert spec.analysis.bootstrap is False
    assert spec.windows == (
        "window_1",
        "window_2",
    )
    assert spec.splits == ("test",)
    assert len(spec.signals) == 14
def test_compare_matching_results() -> None:
    old_rows = [
        {
            "signal": "signal_a",
            "direction": "upper",
            "fraction": 0.10,
            "target": "target_a",
            "window": "window_1",
            "split": "test",
            "n": 100,
            "events": 20,
            "event_rate": 0.20,
            "lift": 2.0,
            "mean_return": 0.05,
        }
    ]
    new_rows = [
        {
            "signal": "signal_a",
            "direction": "upper",
            "fraction": 0.10,
            "target": "target_a",
            "window": "window_1",
            "split": "test",
            "n": 100,
            "events": 20,
            "event_rate": 0.20,
            "lift": 2.0,
            "mean_return": 0.05,
        }
    ]
    comparison = _compare(
        old_rows,
        new_rows,
    )
    assert len(comparison) == 1
    assert comparison.iloc[0]["status"] == "PASS"
    assert comparison.iloc[0]["differences"] == ""
def test_compare_detects_difference() -> None:
    old_rows = [
        {
            "signal": "signal_a",
            "direction": "upper",
            "fraction": 0.10,
            "target": "target_a",
            "window": "window_1",
            "split": "test",
            "n": 100,
            "events": 20,
            "event_rate": 0.20,
            "lift": 2.0,
            "mean_return": 0.05,
        }
    ]
    new_rows = [
        {
            "signal": "signal_a",
            "direction": "upper",
            "fraction": 0.10,
            "target": "target_a",
            "window": "window_1",
            "split": "test",
            "n": 100,
            "events": 21,
            "event_rate": 0.21,
            "lift": 2.1,
            "mean_return": 0.06,
        }
    ]
    comparison = _compare(
        old_rows,
        new_rows,
    )
    assert len(comparison) == 1
    assert comparison.iloc[0]["status"] == "FAIL"
    assert (
        comparison.iloc[0]["differences"]
        == "events,event_rate,lift,mean_return"
    )
