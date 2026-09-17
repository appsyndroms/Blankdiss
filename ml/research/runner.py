"""Kör Blankdiss deterministiska research-matris."""
from __future__ import annotations
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from ml.config import (
    RANDOM_STATE,
    TARGETS,
    TEST_MIN_ROWS,
    WALK_FORWARD_WINDOWS,
)
from ml.dataset import (
    build_target,
    load_features,
)
from .experiments import (
    Experiment,
    build_experiment_matrix,
)
from .signals import (
    build_signal,
    tail_mask,
)
ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = (
    ROOT
    / "data"
    / "processed"
    / "research"
)
RUNS_DIR = RESEARCH_DIR / "runs"
LATEST_DIR = RESEARCH_DIR / "latest"
BOOTSTRAP_ITERATIONS = 2000
MIN_BOOTSTRAP_ROWS = 20
TARGET_BY_NAME = {
    target.name: target
    for target in TARGETS
}
def _safe_float(
    value: object,
) -> float | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)
def _seed_for(
    experiment_id: str,
    window_id: str,
) -> int:
    raw = (
        f"{RANDOM_STATE}:"
        f"{experiment_id}:"
        f"{window_id}"
    )
    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()
    return int(
        digest[:8],
        16,
    )
def bootstrap_mean_difference(
    selected_returns: np.ndarray,
    other_returns: np.ndarray,
    seed: int,
) -> dict[str, float | None]:
    """
    Bootstrap CI för:
        mean(selected) - mean(other)
    """
    selected_returns = np.asarray(
        selected_returns,
        dtype=float,
    )
    other_returns = np.asarray(
        other_returns,
        dtype=float,
    )
    selected_returns = (
        selected_returns[
            np.isfinite(selected_returns)
        ]
    )
    other_returns = (
        other_returns[
            np.isfinite(other_returns)
        ]
    )
    if (
        len(selected_returns) < MIN_BOOTSTRAP_ROWS
        or len(other_returns) < MIN_BOOTSTRAP_ROWS
    ):
        return {
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
            "bootstrap_p_positive": None,
        }
    rng = np.random.default_rng(seed)
    selected_indices = rng.integers(
        0,
        len(selected_returns),
        size=(
            BOOTSTRAP_ITERATIONS,
            len(selected_returns),
        ),
    )
    other_indices = rng.integers(
        0,
        len(other_returns),
        size=(
            BOOTSTRAP_ITERATIONS,
            len(other_returns),
        ),
    )
    selected_means = (
        selected_returns[selected_indices]
        .mean(axis=1)
    )
    other_means = (
        other_returns[other_indices]
        .mean(axis=1)
    )
    differences = (
        selected_means
        - other_means
    )
    return {
        "bootstrap_ci_low": float(
            np.quantile(
                differences,
                0.025,
            )
        ),
        "bootstrap_ci_high": float(
            np.quantile(
                differences,
                0.975,
            )
        ),
        "bootstrap_p_positive": float(
            np.mean(
                differences > 0
            )
        ),
    }
def _status(
    result: dict,
) -> str:
    if result["selected_n"] < MIN_BOOTSTRAP_ROWS:
        return "INSUFFICIENT DATA"
    ci_low = result.get(
        "bootstrap_ci_low"
    )
    ci_high = result.get(
        "bootstrap_ci_high"
    )
    if ci_low is None or ci_high is None:
        return "INSUFFICIENT DATA"
    if ci_low > 0 or ci_high < 0:
        return "INTERESTING"
    if (
        result["selected_n"] >= 100
        and result["lift"] is not None
        and abs(result["lift"]) >= 1.25
    ):
        return "UNSTABLE"
    return "NO SIGNAL"
def _evaluate(
    frame: pd.DataFrame,
    experiment: Experiment,
    window_id: str,
    test_start: str,
    test_end: str,
) -> dict:
    target = TARGET_BY_NAME[
        experiment.target_name
    ]
    test_start_ts = pd.Timestamp(
        test_start
    )
    test_end_ts = pd.Timestamp(
        test_end
    )
    mask = (
        (frame["snapshot_date"] > test_start_ts)
        & (frame["snapshot_date"] <= test_end_ts)
    )
    data = frame.loc[
        mask
    ].copy()
    if len(data) < TEST_MIN_ROWS:
        return {
            "experiment_id": experiment.experiment_id,
            "window_id": window_id,
            "signal_name": experiment.signal_name,
            "target_name": experiment.target_name,
            "tail_fraction": experiment.tail_fraction,
            "tail_direction": experiment.tail_direction,
            "test_start": test_start,
            "test_end": test_end,
            "status": "INSUFFICIENT DATA",
            "reason": "test_rows_below_minimum",
            "rows": int(len(data)),
        }
    target_values = build_target(
        data,
        target,
    )
    signal = build_signal(
        data,
        experiment.signal_name,
    )
    return_column = target.return_column
    returns = pd.to_numeric(
        data[return_column],
        errors="coerce",
    )
    valid = (
        target_values.notna()
        & signal.notna()
        & returns.notna()
    )
    data = data.loc[valid].copy()
    target_values = target_values.loc[
        valid
    ].astype(int)
    signal = signal.loc[
        valid
    ]
    returns = returns.loc[
        valid
    ].astype(float)
    if len(data) < TEST_MIN_ROWS:
        return {
            "experiment_id": experiment.experiment_id,
            "window_id": window_id,
            "signal_name": experiment.signal_name,
            "target_name": experiment.target_name,
            "tail_fraction": experiment.tail_fraction,
            "tail_direction": experiment.tail_direction,
            "test_start": test_start,
            "test_end": test_end,
            "status": "INSUFFICIENT DATA",
            "reason": "valid_rows_below_minimum",
            "rows": int(len(data)),
        }
    selected = tail_mask(
        data,
        signal,
        experiment.tail_fraction,
        experiment.tail_direction,
    )
    selected = selected.fillna(False)
    selected_returns = returns.loc[
        selected
    ]
    other_returns = returns.loc[
        ~selected
    ]
    selected_targets = target_values.loc[
        selected
    ]
    baseline_rate = float(
        target_values.mean()
    )
    selected_rate = (
        float(selected_targets.mean())
        if len(selected_targets)
        else None
    )
    lift = (
        selected_rate / baseline_rate
        if (
            selected_rate is not None
            and baseline_rate > 0
        )
        else None
    )
    return_difference = (
        float(selected_returns.mean())
        - float(other_returns.mean())
        if len(selected_returns)
        and len(other_returns)
        else None
    )
    auc = None
    if (
        target_values.nunique() == 2
        and signal.nunique() > 1
    ):
        auc = float(
            roc_auc_score(
                target_values,
                signal,
            )
        )
    bootstrap = bootstrap_mean_difference(
        selected_returns.to_numpy(),
        other_returns.to_numpy(),
        _seed_for(
            experiment.experiment_id,
            window_id,
        ),
    )
    result = {
        "experiment_id": experiment.experiment_id,
        "window_id": window_id,
        "signal_name": experiment.signal_name,
        "target_name": experiment.target_name,
        "tail_fraction": experiment.tail_fraction,
        "tail_direction": experiment.tail_direction,
        "test_start": test_start,
        "test_end": test_end,
        "rows": int(len(data)),
        "selected_n": int(selected.sum()),
        "other_n": int((~selected).sum()),
        "baseline_rate": baseline_rate,
        "selected_rate": selected_rate,
        "lift": _safe_float(lift),
        "auc": _safe_float(auc),
        "selected_mean_return": _safe_float(
            selected_returns.mean()
        ),
        "selected_median_return": _safe_float(
            selected_returns.median()
        ),
        "other_mean_return": _safe_float(
            other_returns.mean()
        ),
        "other_median_return": _safe_float(
            other_returns.median()
        ),
        "return_difference": _safe_float(
            return_difference
        ),
        **bootstrap,
    }
    result["status"] = _status(
        result
    )
    return result
def _pool_results(
    results: list[dict],
) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for result in results:
        if "selected_n" not in result:
            continue
        grouped.setdefault(
            result["experiment_id"],
            [],
        ).append(result)
    pooled_results: list[dict] = []
    for experiment_id, rows in grouped.items():
        selected_n = sum(
            row["selected_n"]
            for row in rows
        )
        other_n = sum(
            row["other_n"]
            for row in rows
        )
        total_rows = sum(
            row["rows"]
            for row in rows
        )
        weighted_selected_rate = (
            sum(
                row["selected_rate"]
                * row["selected_n"]
                for row in rows
                if row["selected_rate"] is not None
            )
            / selected_n
            if selected_n
            else None
        )
        weighted_baseline = (
            sum(
                row["baseline_rate"]
                * row["rows"]
                for row in rows
            )
            / total_rows
            if total_rows
            else None
        )
        selected_means = [
            (
                row["selected_mean_return"],
                row["selected_n"],
            )
            for row in rows
            if row["selected_mean_return"] is not None
        ]
        other_means = [
            (
                row["other_mean_return"],
                row["other_n"],
            )
            for row in rows
            if row["other_mean_return"] is not None
        ]
        pooled_selected_mean = (
            sum(
                value * count
                for value, count in selected_means
            )
            / sum(
                count
                for _, count in selected_means
            )
            if selected_means
            else None
        )
        pooled_other_mean = (
            sum(
                value * count
                for value, count in other_means
            )
            / sum(
                count
                for _, count in other_means
            )
            if other_means
            else None
        )
        positive_windows = sum(
            1
            for row in rows
            if (
                row.get("return_difference") is not None
                and row["return_difference"] > 0
            )
        )
        pooled_result = {
            "experiment_id": experiment_id,
            "window_id": "pooled_oos",
            "signal_name": rows[0]["signal_name"],
            "target_name": rows[0]["target_name"],
            "tail_fraction": rows[0]["tail_fraction"],
            "tail_direction": rows[0]["tail_direction"],
            "rows": total_rows,
            "selected_n": selected_n,
            "other_n": other_n,
            "baseline_rate": weighted_baseline,
            "selected_rate": weighted_selected_rate,
            "lift": (
                weighted_selected_rate
                / weighted_baseline
                if (
                    weighted_selected_rate is not None
                    and weighted_baseline is not None
                    and weighted_baseline > 0
                )
                else None
            ),
            "window_count": len(rows),
            "window_statuses": [
                row["status"]
                for row in rows
            ],
            "selected_mean_return": (
                pooled_selected_mean
            ),
            "other_mean_return": (
                pooled_other_mean
            ),
            "return_difference": (
                pooled_selected_mean
                - pooled_other_mean
                if (
                    pooled_selected_mean is not None
                    and pooled_other_mean is not None
                )
                else None
            ),
            "positive_return_windows": (
                positive_windows
            ),
            "stable_direction": (
                positive_windows == 0
                or positive_windows == len(rows)
            ),
            "window_ci_excludes_zero": [
                (
                    row.get("bootstrap_ci_low") is not None
                    and row.get("bootstrap_ci_high") is not None
                    and (
                        row["bootstrap_ci_low"] > 0
                        or row["bootstrap_ci_high"] < 0
                    )
                )
                for row in rows
            ],
        }
        if (
            pooled_result["stable_direction"]
            and any(
                pooled_result[
                    "window_ci_excludes_zero"
                ]
            )
        ):
            pooled_result["status"] = (
                "STRONG RESEARCH CANDIDATE"
            )
        elif (
            len(
                set(
                    pooled_result[
                        "window_statuses"
                    ]
                )
            ) > 1
        ):
            pooled_result["status"] = "UNSTABLE"
        else:
            pooled_result["status"] = "NO SIGNAL"
        pooled_results.append(
            pooled_result
        )
    return pooled_results
def _write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
def _build_summary(
    window_results: list[dict],
    pooled_results: list[dict],
) -> dict:
    status_counts: dict[str, int] = {}
    for result in pooled_results:
        status = result["status"]
        status_counts[status] = (
            status_counts.get(status, 0)
            + 1
        )
    return {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "result_count": len(
            pooled_results
        ),
        "window_result_count": len(
            window_results
        ),
        "experiment_count": len(
            {
                result["experiment_id"]
                for result in pooled_results
            }
        ),
        "status_counts": status_counts,
        "multiple_testing_note": (
            "Research sweepen testar många signaler, "
            "tails och targets. Enstaka starka resultat "
            "ska därför betraktas som hypotesgenererande "
            "tills de replikerats i separat data."
        ),
    }
def _write_report(
    path: Path,
    summary: dict,
    pooled_results: list[dict],
) -> None:
    lines = [
        "# Blankdiss Research Report",
        "",
        f"Resultat: **{summary['result_count']}**",
        "",
        "## Status",
        "",
    ]
    for status, count in sorted(
        summary["status_counts"].items()
    ):
        lines.append(
            f"- {status}: {count}"
        )
    lines.extend(
        [
            "",
            "## Metod",
            "",
            "- Deterministisk experimentmatris.",
            "- Cross-sectional tails per snapshot-datum.",
            "- Endast OOS walk-forward-perioder.",
            "- Bootstrap-CI med fast seed.",
            "- Pooled OOS-resultat.",
            "",
            "## Viktig begränsning",
            "",
            summary[
                "multiple_testing_note"
            ],
            "",
            "Detta är research-resultat och inte "
            "investeringsrekommendationer.",
            "",
        ]
    )
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
def main() -> None:
    run_id = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    frame = load_features()
    experiments = (
        build_experiment_matrix()
    )
    print(
        f"Research experiments: "
        f"{len(experiments):,}"
    )
    window_results: list[dict] = []
    for window_index, window in enumerate(
        WALK_FORWARD_WINDOWS,
        start=1,
    ):
        window_id = (
            f"window_{window_index}"
        )
        print(
            f"\nRunning {window_id}: "
            f"test <= {window.test_end}"
        )
        for index, experiment in enumerate(
            experiments,
            start=1,
        ):
            result = _evaluate(
                frame,
                experiment,
                window_id,
                window.validation_end,
                window.test_end,
            )
            window_results.append(
                result
            )
            if index % 100 == 0:
                print(
                    f"  {index:,}/"
                    f"{len(experiments):,}"
                )
    pooled_results = _pool_results(
        window_results
    )
    summary = _build_summary(
        window_results,
        pooled_results,
    )
    run_window_path = (
        run_dir
        / "experiment_results.jsonl"
    )
    run_pooled_path = (
        run_dir
        / "pooled_results.jsonl"
    )
    run_summary_path = (
        run_dir
        / "research_summary.json"
    )
    run_report_path = (
        run_dir
        / "research_report.md"
    )
    _write_jsonl(
        run_window_path,
        window_results,
    )
    _write_jsonl(
        run_pooled_path,
        pooled_results,
    )
    run_summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(
        run_report_path,
        summary,
        pooled_results,
    )
    if LATEST_DIR.exists():
        shutil.rmtree(
            LATEST_DIR
        )
    LATEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.copy2(
        run_window_path,
        LATEST_DIR
        / "experiment_results.jsonl",
    )
    shutil.copy2(
        run_pooled_path,
        LATEST_DIR
        / "pooled_results.jsonl",
    )
    shutil.copy2(
        run_summary_path,
        LATEST_DIR
        / "research_summary.json",
    )
    shutil.copy2(
        run_report_path,
        LATEST_DIR
        / "research_report.md",
    )
    print(
        "\nResearch complete."
    )
    print(
        json.dumps(
            summary[
                "status_counts"
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
if __name__ == "__main__":
    main()
