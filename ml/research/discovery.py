from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import TARGETS, WALK_FORWARD_WINDOWS
from ml.dataset import build_target, load_features
from ml.research.bootstrap import bootstrap_mean_ci
from ml.research.signals import build_signal, tail_mask


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "ml"
    / "research"
    / "discovery"
)


SI_SIGNALS = (
    "short_interest_change",
    "short_interest_level",
)

STRESS_SIGNALS = (
    "price_volatility_20d",
    "price_momentum_5d",
    "price_momentum_20d",
    "price_momentum_60d",
    "distance_from_20d_high",
    "distance_from_60d_high",
)

TARGETS_TO_TEST = (
    "down_5pct_5d",
    "down_7pct_5d",
    "down_10pct_5d",
)

TAIL_FRACTIONS = (
    0.20,
    0.10,
    0.05,
    0.025,
    0.01,
)

STRESS_DIRECTIONS = {
    "price_volatility_20d": "upper",
    "price_momentum_5d": "lower",
    "price_momentum_20d": "lower",
    "price_momentum_60d": "lower",
    "distance_from_20d_high": "lower",
    "distance_from_60d_high": "lower",
}

MIN_SELECTED_ROWS = 50
RANDOM_STATE = 42


@dataclass(frozen=True)
class DiscoveryCandidate:
    target_name: str
    components: tuple[str, str]
    fractions: tuple[float, float]
    directions: tuple[str, str]

    @property
    def candidate_id(self) -> str:
        parts = [self.target_name]

        for signal, fraction, direction in zip(
            self.components,
            self.fractions,
            self.directions,
        ):
            fraction_name = str(fraction).replace(
                ".",
                "_",
            )

            parts.append(
                f"{signal}__{direction}__{fraction_name}"
            )

        return "___".join(parts)


def _window_name(index: int) -> str:
    return f"window_{index + 1}"


def _safe_float(value) -> float | None:
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def _stable_seed(
    candidate_id: str,
    window_name: str,
) -> int:
    payload = (
        f"{candidate_id}|"
        f"{window_name}|"
        f"{RANDOM_STATE}"
    )

    value = 0

    for character in payload:
        value = (
            value * 131
            + ord(character)
        ) % (2**32 - 1)

    return value


def _split_mask(
    frame: pd.DataFrame,
    window,
) -> dict[str, pd.Series]:
    dates = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    train_end = pd.Timestamp(
        window.train_end
    )

    validation_end = pd.Timestamp(
        window.validation_end
    )

    test_end = pd.Timestamp(
        window.test_end
    )

    return {
        "train": dates <= train_end,
        "validation": (
            (dates > train_end)
            & (dates <= validation_end)
        ),
        "test": (
            (dates > validation_end)
            & (dates <= test_end)
        ),
    }


def _build_component_masks(
    frame: pd.DataFrame,
) -> dict[
    tuple[str, float, str],
    pd.Series,
]:
    masks = {}

    signal_names = (
        SI_SIGNALS
        + STRESS_SIGNALS
    )

    for signal_name in signal_names:
        signal = build_signal(
            frame,
            signal_name,
        )

        if signal_name in SI_SIGNALS:
            directions = ("upper",)
        else:
            directions = (
                STRESS_DIRECTIONS[
                    signal_name
                ],
            )

        for direction in directions:
            for fraction in TAIL_FRACTIONS:
                key = (
                    signal_name,
                    fraction,
                    direction,
                )

                masks[key] = tail_mask(
                    frame,
                    signal,
                    fraction,
                    direction,
                )

    return masks


def _empty_metrics(
    baseline_n: int = 0,
) -> dict:
    return {
        "n": 0,
        "baseline_n": baseline_n,
        "event_rate": None,
        "baseline_event_rate": None,
        "event_rate_difference": None,
        "lift": None,
        "mean_return": None,
        "median_return": None,
        "return_ci_low": None,
        "return_ci_high": None,
    }


def _evaluate_selection(
    frame: pd.DataFrame,
    target: pd.Series,
    selected: pd.Series,
    split_mask: pd.Series,
) -> dict:
    valid = (
        split_mask
        & selected
        & target.notna()
    )

    baseline_valid = (
        split_mask
        & target.notna()
    )

    baseline_values = pd.to_numeric(
        target.loc[baseline_valid],
        errors="coerce",
    ).dropna()

    selected_values = pd.to_numeric(
        target.loc[valid],
        errors="coerce",
    ).dropna()

    baseline_n = int(
        len(baseline_values)
    )

    n = int(
        len(selected_values)
    )

    if baseline_n == 0:
        return _empty_metrics()

    baseline_rate = float(
        baseline_values.mean()
    )

    if n == 0:
        result = _empty_metrics(
            baseline_n
        )

        result["baseline_event_rate"] = (
            baseline_rate
        )

        return result

    event_rate = float(
        selected_values.mean()
    )

    difference = (
        event_rate
        - baseline_rate
    )

    lift = (
        event_rate / baseline_rate
        if baseline_rate > 0
        else None
    )

    return {
        "n": n,
        "baseline_n": baseline_n,
        "event_rate": event_rate,
        "baseline_event_rate": baseline_rate,
        "event_rate_difference": difference,
        "lift": lift,
        "mean_return": None,
        "median_return": None,
        "return_ci_low": None,
        "return_ci_high": None,
    }


def _add_return_metrics(
    result: dict,
    frame: pd.DataFrame,
    selected: pd.Series,
    split_mask: pd.Series,
    return_column: str,
    seed: int,
) -> None:
    if return_column not in frame.columns:
        return

    mask = (
        split_mask
        & selected
    )

    values = pd.to_numeric(
        frame.loc[
            mask,
            return_column,
        ],
        errors="coerce",
    ).dropna()

    if values.empty:
        return

    result["mean_return"] = float(
        values.mean()
    )

    result["median_return"] = float(
        values.median()
    )

    ci_low, ci_high = (
        bootstrap_mean_ci(
            values.to_numpy(
                dtype=float
            ),
            seed=seed,
        )
    )

    result["return_ci_low"] = (
        _safe_float(ci_low)
    )

    result["return_ci_high"] = (
        _safe_float(ci_high)
    )


def _generate_candidates() -> list[
    DiscoveryCandidate
]:
    candidates = []

    for target_name in TARGETS_TO_TEST:
        for si_signal in SI_SIGNALS:
            for stress_signal in STRESS_SIGNALS:
                stress_direction = (
                    STRESS_DIRECTIONS[
                        stress_signal
                    ]
                )

                for si_fraction in TAIL_FRACTIONS:
                    for stress_fraction in (
                        TAIL_FRACTIONS
                    ):
                        candidates.append(
                            DiscoveryCandidate(
                                target_name=target_name,
                                components=(
                                    si_signal,
                                    stress_signal,
                                ),
                                fractions=(
                                    si_fraction,
                                    stress_fraction,
                                ),
                                directions=(
                                    "upper",
                                    stress_direction,
                                ),
                            )
                        )

    return candidates


def _run_candidate(
    frame: pd.DataFrame,
    target: pd.Series,
    return_column: str,
    candidate: DiscoveryCandidate,
    component_masks: dict,
) -> list[dict]:
    first_key = (
        candidate.components[0],
        candidate.fractions[0],
        candidate.directions[0],
    )

    second_key = (
        candidate.components[1],
        candidate.fractions[1],
        candidate.directions[1],
    )

    selected = (
        component_masks[first_key]
        & component_masks[second_key]
    )

    results = []

    for window_index, window in enumerate(
        WALK_FORWARD_WINDOWS
    ):
        window_name = _window_name(
            window_index
        )

        split_masks = _split_mask(
            frame,
            window,
        )

        test_mask = split_masks["test"]

        seed = _stable_seed(
            candidate.candidate_id,
            window_name,
        )

        metrics = _evaluate_selection(
            frame,
            target,
            selected,
            test_mask,
        )

        _add_return_metrics(
            metrics,
            frame,
            selected,
            test_mask,
            return_column,
            seed,
        )

        metrics.update(
            {
                "candidate_id":
                    candidate.candidate_id,
                "target_name":
                    candidate.target_name,
                "components":
                    list(candidate.components),
                "fractions":
                    list(candidate.fractions),
                "directions":
                    list(candidate.directions),
                "window":
                    window_name,
            }
        )

        results.append(metrics)

    return results


def _candidate_status(
    rows: list[dict],
    effects: list[float],
) -> str:
    if not effects:
        return "NO_SIGNAL"

    if len(rows) < 2:
        return "SINGLE_WINDOW"

    if all(
        effect > 0
        for effect in effects
    ):
        if min(
            row["n"]
            for row in rows
        ) >= 100:
            return "INTERESTING"

        return "INTERESTING_SMALL_N"

    if all(
        effect < 0
        for effect in effects
    ):
        return "OPPOSITE_DIRECTION"

    return "UNSTABLE"


def _pool_candidates(
    results: list[dict],
) -> list[dict]:
    grouped = {}

    for result in results:
        grouped.setdefault(
            result["candidate_id"],
            [],
        ).append(result)

    pooled = []

    for candidate_id, rows in grouped.items():
        valid = [
            row
            for row in rows
            if row["n"] >= MIN_SELECTED_ROWS
            and row["event_rate"] is not None
        ]

        if not valid:
            continue

        effects = [
            row["event_rate_difference"]
            for row in valid
            if row["event_rate_difference"]
            is not None
        ]

        lifts = [
            row["lift"]
            for row in valid
            if row["lift"] is not None
        ]

        returns = [
            row["mean_return"]
            for row in valid
            if row["mean_return"] is not None
        ]

        positive_windows = sum(
            effect > 0
            for effect in effects
        )

        negative_windows = sum(
            effect < 0
            for effect in effects
        )

        pooled.append(
            {
                "candidate_id":
                    candidate_id,
                "target_name":
                    valid[0]["target_name"],
                "components":
                    valid[0]["components"],
                "fractions":
                    valid[0]["fractions"],
                "directions":
                    valid[0]["directions"],
                "windows":
                    len(valid),
                "positive_windows":
                    positive_windows,
                "negative_windows":
                    negative_windows,
                "mean_event_rate_difference":
                    float(
                        np.mean(effects)
                    )
                    if effects
                    else None,
                "min_event_rate_difference":
                    float(
                        np.min(effects)
                    )
                    if effects
                    else None,
                "mean_lift":
                    float(
                        np.mean(lifts)
                    )
                    if lifts
                    else None,
                "mean_return":
                    float(
                        np.mean(returns)
                    )
                    if returns
                    else None,
                "total_n":
                    int(
                        sum(
                            row["n"]
                            for row in valid
                        )
                    ),
                "status":
                    _candidate_status(
                        valid,
                        effects,
                    ),
            }
        )

    return sorted(
        pooled,
        key=_discovery_sort_key,
        reverse=True,
    )


def _discovery_sort_key(
    result: dict,
) -> tuple:
    effect = (
        result[
            "mean_event_rate_difference"
        ]
        or 0.0
    )

    consistency = (
        result["positive_windows"]
        / max(
            result["windows"],
            1,
        )
    )

    n = result["total_n"]

    discovery_score = (
        max(effect, 0.0)
        * consistency
        * math.sqrt(
            max(n, 1)
        )
    )

    return (
        discovery_score,
        effect,
        n,
    )


def _build_findings(
    pooled: list[dict],
) -> dict:
    interesting = [
        row
        for row in pooled
        if row["status"]
        in {
            "INTERESTING",
            "INTERESTING_SMALL_N",
        }
    ]

    unstable = [
        row
        for row in pooled
        if row["status"] == "UNSTABLE"
    ]

    opposite = [
        row
        for row in pooled
        if row["status"]
        == "OPPOSITE_DIRECTION"
    ]

    return {
        "interesting": interesting[:50],
        "unstable": unstable[:25],
        "opposite_direction": opposite[:25],
        "next_tests": _suggest_next_tests(
            interesting
        ),
    }


def _suggest_next_tests(
    interesting: list[dict],
) -> list[dict]:
    suggestions = []

    for row in interesting[:10]:
        suggestions.append(
            {
                "candidate_id":
                    row["candidate_id"],
                "reason":
                    (
                        "Stabilt mönster över "
                        "walk-forward-fönster. "
                        "Verifiera med separat "
                        "diagnostik innan hypotesen "
                        "betraktas som robust."
                    ),
                "next_tests": [
                    "sector_relative",
                    "prior_return_control",
                    "event_risk_control",
                    "bootstrap_interaction",
                ],
            }
        )

    return suggestions


def _write_json(
    path: Path,
    payload,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def _fmt(
    value,
) -> str:
    if value is None:
        return ""

    return f"{value:.3f}"


def _fmt_pct(
    value,
) -> str:
    if value is None:
        return ""

    return f"{value * 100:.2f}%"


def _write_report(
    path: Path,
    metadata: dict,
    pooled: list[dict],
) -> None:
    lines = [
        "# Blankdiss Discovery",
        "",
        f"Generated: {metadata['created_at_utc']}",
        "",
        (
            f"Feature rows: "
            f"{metadata['feature_rows']:,}"
        ),
        (
            f"Candidates: "
            f"{metadata['candidates']:,}"
        ),
        "",
        "## Top candidates",
        "",
        (
            "| Target | Components | Fractions | "
            "Windows | Positive | Effect | Lift | "
            "Mean return | N | Status |"
        ),
        (
            "|---|---|---|---:|---:|---:|---:|"
            "---:|---:|---|"
        ),
    ]

    for row in pooled[:50]:
        components = " × ".join(
            row["components"]
        )

        fractions = " × ".join(
            f"{fraction:g}"
            for fraction in row["fractions"]
        )

        effect = row[
            "mean_event_rate_difference"
        ]

        lift = row["mean_lift"]

        mean_return = row["mean_return"]

        lines.append(
            "| "
            f"{row['target_name']} | "
            f"{components} | "
            f"{fractions} | "
            f"{row['windows']} | "
            f"{row['positive_windows']} | "
            f"{_fmt_pct(effect)} | "
            f"{_fmt(lift)} | "
            f"{_fmt_pct(mean_return)} | "
            f"{row['total_n']} | "
            f"{row['status']} |"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )


def run() -> None:
    started = datetime.now(
        timezone.utc
    )

    print(
        "Loading features...",
        flush=True,
    )

    frame = load_features()

    print(
        f"Loaded {len(frame):,} feature rows",
        flush=True,
    )

    candidates = _generate_candidates()

    print(
        f"Discovery candidates: "
        f"{len(candidates):,}",
        flush=True,
    )

    expected_candidates = (
        len(TARGETS_TO_TEST)
        * len(SI_SIGNALS)
        * len(STRESS_SIGNALS)
        * len(TAIL_FRACTIONS)
        * len(TAIL_FRACTIONS)
    )

    if len(candidates) != expected_candidates:
        raise RuntimeError(
            "Discovery candidate count mismatch: "
            f"{len(candidates)} != "
            f"{expected_candidates}"
        )

    print(
        "Building component masks...",
        flush=True,
    )

    component_masks = (
        _build_component_masks(frame)
    )

    target_configs = {
        target.name: target
        for target in TARGETS
        if target.name
        in TARGETS_TO_TEST
    }

    missing_targets = sorted(
        set(TARGETS_TO_TEST)
        - set(target_configs)
    )

    if missing_targets:
        raise ValueError(
            "Discovery saknar targets: "
            + ", ".join(
                missing_targets
            )
        )

    all_results = []

    for target_name in TARGETS_TO_TEST:
        target_config = target_configs[
            target_name
        ]

        target = build_target(
            frame,
            target_config,
        )

        target_candidates = [
            candidate
            for candidate in candidates
            if candidate.target_name
            == target_name
        ]

        print(
            f"Testing target: "
            f"{target_name}",
            flush=True,
        )

        for index, candidate in enumerate(
            target_candidates,
            start=1,
        ):
            rows = _run_candidate(
                frame,
                target,
                target_config.return_column,
                candidate,
                component_masks,
            )

            all_results.extend(rows)

            if (
                index % 100 == 0
                or index
                == len(target_candidates)
            ):
                print(
                    f"  {index:,}/"
                    f"{len(target_candidates):,}",
                    flush=True,
                )

    pooled = _pool_candidates(
        all_results
    )

    findings = _build_findings(
        pooled
    )

    finished = datetime.now(
        timezone.utc
    )

    timestamp = finished.strftime(
        "%Y%m%dT%H%M%SZ"
    )

    run_dir = (
        OUTPUT_DIR
        / timestamp
    )

    latest_dir = (
        OUTPUT_DIR
        / "latest"
    )

    metadata = {
        "created_at_utc":
            finished.isoformat(),
        "started_at_utc":
            started.isoformat(),
        "feature_rows":
            int(len(frame)),
        "candidates":
            int(len(candidates)),
        "raw_results":
            int(len(all_results)),
        "pooled_candidates":
            int(len(pooled)),
        "walk_forward_windows":
            len(WALK_FORWARD_WINDOWS),
        "targets":
            list(TARGETS_TO_TEST),
        "si_signals":
            list(SI_SIGNALS),
        "stress_signals":
            list(STRESS_SIGNALS),
        "tail_fractions":
            list(TAIL_FRACTIONS),
    }

    for directory in (
        run_dir,
        latest_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        _write_json(
            directory / "metadata.json",
            metadata,
        )

        _write_jsonl(
            directory / "results.jsonl",
            all_results,
        )

        _write_json(
            directory / "pooled.json",
            pooled,
        )

        _write_json(
            directory / "findings.json",
            findings,
        )

        _write_report(
            directory / "report.md",
            metadata,
            pooled,
        )

    print()
    print(
        f"Discovery complete: "
        f"{run_dir}",
        flush=True,
    )

    print(
        f"Interesting candidates: "
        f"{len(findings['interesting']):,}",
        flush=True,
    )


if __name__ == "__main__":
    run()
