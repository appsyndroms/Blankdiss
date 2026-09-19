"""YAML-driven discovery engine for Blankdiss research."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ml.config import TARGETS, WALK_FORWARD_WINDOWS
from ml.dataset import build_target, load_features
from ml.research.bootstrap import bootstrap_mean_difference
from ml.research.signals import build_signal, tail_mask


ROOT = Path(__file__).resolve().parents[2]

WORKFLOW_PATH = (
    ROOT
    / ".github"
    / "workflows"
    / "ml-research.yml"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "ml"
    / "research"
    / "discovery"
)


DEFAULT_MIN_ROWS_PER_WINDOW = 20
DEFAULT_MIN_POSITIVE_WINDOWS = 2
DEFAULT_MAX_FINDINGS = 25


def _load_config() -> dict[str, Any]:
    if not WORKFLOW_PATH.exists():
        raise FileNotFoundError(
            f"Saknar workflow-konfiguration: {WORKFLOW_PATH}"
        )

    with WORKFLOW_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        workflow = yaml.safe_load(handle)

    if not isinstance(workflow, dict):
        raise ValueError(
            "ml-research.yml kunde inte läsas som ett YAML-objekt."
        )

    config = workflow.get("discovery")

    if not isinstance(config, dict):
        raise ValueError(
            "Saknar 'discovery:' i ml-research.yml."
        )

    if config.get("enabled", True) is not True:
        raise ValueError(
            "Discovery är inte aktiverad i ml-research.yml."
        )

    return config


def _require_list(
    config: dict[str, Any],
    key: str,
) -> list[Any]:
    value = config.get(key)

    if not isinstance(value, list) or not value:
        raise ValueError(
            f"Discovery-konfigurationen kräver en icke-tom lista: {key}"
        )

    return value


def _fraction_name(
    fraction: float,
) -> str:
    mapping = {
        0.20: "20pct",
        0.10: "10pct",
        0.05: "5pct",
        0.025: "2_5pct",
        0.01: "1pct",
    }

    if fraction in mapping:
        return mapping[fraction]

    return str(fraction).replace(".", "_")


def _stable_seed(
    *parts: object,
) -> int:
    payload = "|".join(
        str(part)
        for part in parts
    ).encode("utf-8")

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    ) % (2**32 - 1)


def _window_name(
    index: int,
) -> str:
    return f"window_{index + 1}"


def _target_configs(
    target_names: list[str],
) -> dict[str, Any]:
    configs = {
        target.name: target
        for target in TARGETS
    }

    missing = sorted(
        set(target_names)
        - set(configs)
    )

    if missing:
        raise ValueError(
            "Okända discovery targets: "
            + ", ".join(missing)
        )

    return {
        name: configs[name]
        for name in target_names
    }


def _validate_signal_names(
    frame: pd.DataFrame,
    signal_names: list[str],
) -> None:
    for name in signal_names:
        build_signal(
            frame,
            name,
        )


def _build_window_masks(
    frame: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    dates = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    ).to_numpy(
        dtype="datetime64[ns]"
    )

    masks: dict[
        str,
        dict[str, np.ndarray],
    ] = {}

    for index, window in enumerate(
        WALK_FORWARD_WINDOWS
    ):
        name = _window_name(index)

        train_end = np.datetime64(
            window.train_end
        )

        validation_end = np.datetime64(
            window.validation_end
        )

        test_end = np.datetime64(
            window.test_end
        )

        masks[name] = {
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

    return masks


def _build_tail_masks(
    frame: pd.DataFrame,
    signal_names: list[str],
    fractions: list[float],
    directions: dict[str, str],
) -> dict[str, np.ndarray]:
    cache: dict[
        str,
        np.ndarray,
    ] = {}

    for signal_name in signal_names:
        signal = build_signal(
            frame,
            signal_name,
        )

        for fraction in fractions:
            direction = directions.get(
                signal_name,
                "upper",
            )

            key = (
                f"{signal_name}|"
                f"{direction}|"
                f"{fraction}"
            )

            cache[key] = tail_mask(
                frame,
                signal,
                fraction,
                direction,
            ).to_numpy(
                dtype=bool
            )

    return cache


def _tail_key(
    signal_name: str,
    direction: str,
    fraction: float,
) -> str:
    return (
        f"{signal_name}|"
        f"{direction}|"
        f"{fraction}"
    )


def _target_arrays(
    frame: pd.DataFrame,
    target_names: list[str],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
]:
    configs = _target_configs(
        target_names
    )

    targets: dict[
        str,
        np.ndarray,
    ] = {}

    for name, config in configs.items():
        values = build_target(
            frame,
            config,
        )

        targets[name] = pd.to_numeric(
            values,
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

    return targets, configs


def _return_arrays(
    frame: pd.DataFrame,
    target_configs: dict[str, Any],
) -> dict[str, np.ndarray]:
    columns = sorted(
        {
            target.return_column
            for target in target_configs.values()
            if getattr(
                target,
                "return_column",
                None,
            )
            and target.return_column
            in frame.columns
        }
    )

    return {
        column: pd.to_numeric(
            frame[column],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )
        for column in columns
    }


def _safe_auc(
    target: np.ndarray,
    score: np.ndarray,
) -> float | None:
    valid = (
        np.isfinite(target)
        & np.isfinite(score)
    )

    if not valid.any():
        return None

    y = target[valid]
    s = score[valid]

    if np.unique(y).size < 2:
        return None

    from sklearn.metrics import roc_auc_score

    try:
        return float(
            roc_auc_score(
                y,
                s,
            )
        )
    except ValueError:
        return None


def _combined_score(
    signal_rank: np.ndarray,
    stress_rank: np.ndarray,
) -> np.ndarray:
    valid = (
        np.isfinite(signal_rank)
        & np.isfinite(stress_rank)
    )

    score = np.full(
        signal_rank.shape,
        np.nan,
        dtype=float,
    )

    score[valid] = (
        signal_rank[valid]
        + stress_rank[valid]
    ) / 2.0

    return score


def _rank_signal(
    frame: pd.DataFrame,
    values: pd.Series,
) -> np.ndarray:
    working = pd.DataFrame(
        {
            "snapshot_date": frame[
                "snapshot_date"
            ],
            "value": values,
        }
    )

    ranks = (
        working.groupby(
            "snapshot_date"
        )["value"]
        .rank(
            pct=True,
            method="average",
        )
    )

    return ranks.to_numpy(
        dtype=float
    )


def _event_metrics(
    target: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    valid = np.isfinite(target)

    if not valid.any():
        return {
            "n": 0,
            "events": 0,
            "event_rate": None,
            "baseline_event_rate": None,
            "lift": None,
        }

    y = target[valid]
    selection = selected[valid]

    events = y > 0

    n = int(
        selection.sum()
    )

    selected_events = int(
        events[selection].sum()
    )

    baseline_rate = float(
        events.mean()
    )

    if n == 0:
        return {
            "n": 0,
            "events": 0,
            "event_rate": None,
            "baseline_event_rate": baseline_rate,
            "lift": None,
        }

    event_rate = (
        selected_events / n
    )

    lift = (
        event_rate / baseline_rate
        if baseline_rate > 0
        else None
    )

    return {
        "n": n,
        "events": selected_events,
        "event_rate": float(event_rate),
        "baseline_event_rate": baseline_rate,
        "lift": (
            float(lift)
            if lift is not None
            else None
        ),
    }


def _return_metrics(
    returns: np.ndarray,
    selected: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    selected_valid = (
        np.isfinite(returns)
        & selected
    )

    rest_valid = (
        np.isfinite(returns)
        & ~selected
    )

    selected_values = returns[
        selected_valid
    ]

    rest_values = returns[
        rest_valid
    ]

    if selected_values.size == 0:
        return {
            "return_n": 0,
            "mean_return": None,
            "median_return": None,
            "rest_mean_return": None,
            "mean_difference": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
        }

    mean_return = float(
        np.mean(selected_values)
    )

    median_return = float(
        np.median(selected_values)
    )

    rest_mean = (
        float(np.mean(rest_values))
        if rest_values.size
        else None
    )

    ci_low, ci_high = (
        bootstrap_mean_difference(
            selected_values,
            rest_values,
            seed=seed,
        )
        if rest_values.size
        else (None, None)
    )

    mean_difference = (
        mean_return - rest_mean
        if rest_mean is not None
        else None
    )

    return {
        "return_n": int(
            selected_values.size
        ),
        "mean_return": mean_return,
        "median_return": median_return,
        "rest_mean_return": rest_mean,
        "mean_difference": (
            float(mean_difference)
            if mean_difference is not None
            else None
        ),
        "bootstrap_ci_low": (
            float(ci_low)
            if ci_low is not None
            else None
        ),
        "bootstrap_ci_high": (
            float(ci_high)
            if ci_high is not None
            else None
        ),
    }


def _evaluate_candidate(
    *,
    frame: pd.DataFrame,
    target: np.ndarray,
    returns: np.ndarray,
    signal: np.ndarray,
    stress: np.ndarray,
    signal_tail: np.ndarray,
    stress_tail: np.ndarray,
    window_mask: np.ndarray,
    candidate: dict[str, Any],
    window_name: str,
) -> dict[str, Any]:
    selected = (
        signal_tail
        & stress_tail
        & window_mask
        & np.isfinite(signal)
        & np.isfinite(stress)
        & np.isfinite(target)
    )

    target_window = (
        target[window_mask]
    )

    selected_window = (
        signal_tail
        & stress_tail
    )[window_mask]

    event = _event_metrics(
        target_window,
        selected_window,
    )

    return_metrics = _return_metrics(
        returns[window_mask],
        selected_window,
        seed=_stable_seed(
            candidate["candidate_id"],
            window_name,
        ),
    )

    signal_window = signal[
        window_mask
    ]

    stress_window = stress[
        window_mask
    ]

    valid_auc = (
        window_mask
        & np.isfinite(target)
        & np.isfinite(signal)
        & np.isfinite(stress)
    )

    signal_score = _rank_signal(
        frame.loc[
            window_mask
        ].reset_index(
            drop=True
        ),
        pd.Series(
            signal_window
        ),
    )

    stress_score = _rank_signal(
        frame.loc[
            window_mask
        ].reset_index(
            drop=True
        ),
        pd.Series(
            stress_window
        ),
    )

    combined_score = _combined_score(
        signal_score,
        stress_score,
    )

    auc = _safe_auc(
        target[window_mask],
        combined_score,
    )

    n_valid = int(
        selected.sum()
    )

    return {
        "candidate_id": candidate[
            "candidate_id"
        ],
        "target_name": candidate[
            "target_name"
        ],
        "signal_name": candidate[
            "signal_name"
        ],
        "stress_feature": candidate[
            "stress_feature"
        ],
        "signal_tail_fraction": candidate[
            "signal_tail_fraction"
        ],
        "stress_tail_fraction": candidate[
            "stress_tail_fraction"
        ],
        "signal_tail_direction": candidate[
            "signal_tail_direction"
        ],
        "stress_tail_direction": candidate[
            "stress_tail_direction"
        ],
        "window": window_name,
        "split": "test",
        "n": n_valid,
        "events": event["events"],
        "event_rate": event["event_rate"],
        "baseline_event_rate": event[
            "baseline_event_rate"
        ],
        "lift": event["lift"],
        "auc": auc,
        "return_n": return_metrics[
            "return_n"
        ],
        "mean_return": return_metrics[
            "mean_return"
        ],
        "median_return": return_metrics[
            "median_return"
        ],
        "rest_mean_return": return_metrics[
            "rest_mean_return"
        ],
        "mean_difference": return_metrics[
            "mean_difference"
        ],
        "bootstrap_ci_low": return_metrics[
            "bootstrap_ci_low"
        ],
        "bootstrap_ci_high": return_metrics[
            "bootstrap_ci_high"
        ],
        "n_valid_auc": int(
            valid_auc.sum()
        ),
    }


def _candidate_list(
    *,
    target_names: list[str],
    signal_names: list[str],
    stress_features: list[str],
    tails: list[float],
    stress_directions: dict[str, str],
) -> list[dict[str, Any]]:
    candidates: list[
        dict[str, Any]
    ] = []

    for target_name in target_names:
        for signal_name in signal_names:
            for stress_feature in stress_features:
                stress_direction = (
                    stress_directions.get(
                        stress_feature
                    )
                )

                if stress_direction not in {
                    "upper",
                    "lower",
                }:
                    raise ValueError(
                        "Saknar giltig stress-riktning "
                        f"för {stress_feature}."
                    )

                for signal_fraction in tails:
                    for stress_fraction in tails:
                        candidate_id = (
                            f"{signal_name}"
                            f"__{stress_feature}"
                            f"__{target_name}"
                            f"__signal_"
                            f"{_fraction_name(signal_fraction)}"
                            f"__stress_"
                            f"{_fraction_name(stress_fraction)}"
                        )

                        candidates.append(
                            {
                                "candidate_id": candidate_id,
                                "target_name": target_name,
                                "signal_name": signal_name,
                                "stress_feature": stress_feature,
                                "signal_tail_fraction": signal_fraction,
                                "stress_tail_fraction": stress_fraction,
                                "signal_tail_direction": "upper",
                                "stress_tail_direction": stress_direction,
                            }
                        )

    return candidates


def _pool_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for result in results:
        grouped.setdefault(
            result["candidate_id"],
            [],
        ).append(result)

    pooled: list[dict[str, Any]] = []

    for candidate_id, rows in grouped.items():
        first = rows[0]

        def values(
            key: str,
        ) -> list[float]:
            return [
                float(row[key])
                for row in rows
                if row.get(key) is not None
            ]

        def mean(
            key: str,
        ) -> float | None:
            current = values(key)

            if not current:
                return None

            return float(
                np.mean(current)
            )

        positive_windows = sum(
            1
            for row in rows
            if (
                row.get("lift") is not None
                and row["lift"] > 1.0
            )
            or (
                row.get(
                    "mean_difference"
                ) is not None
                and row[
                    "mean_difference"
                ] < 0
            )
        )

        pooled.append(
            {
                "candidate_id": candidate_id,
                "target_name": first[
                    "target_name"
                ],
                "signal_name": first[
                    "signal_name"
                ],
                "stress_feature": first[
                    "stress_feature"
                ],
                "signal_tail_fraction": first[
                    "signal_tail_fraction"
                ],
                "stress_tail_fraction": first[
                    "stress_tail_fraction"
                ],
                "signal_tail_direction": first[
                    "signal_tail_direction"
                ],
                "stress_tail_direction": first[
                    "stress_tail_direction"
                ],
                "windows": len(rows),
                "positive_windows": positive_windows,
                "n": sum(
                    int(row.get("n", 0))
                    for row in rows
                ),
                "return_n": sum(
                    int(
                        row.get(
                            "return_n",
                            0,
                        )
                    )
                    for row in rows
                ),
                "auc": mean("auc"),
                "event_rate": mean(
                    "event_rate"
                ),
                "baseline_event_rate": mean(
                    "baseline_event_rate"
                ),
                "lift": mean("lift"),
                "mean_return": mean(
                    "mean_return"
                ),
                "rest_mean_return": mean(
                    "rest_mean_return"
                ),
                "mean_difference": mean(
                    "mean_difference"
                ),
                "bootstrap_ci_low": mean(
                    "bootstrap_ci_low"
                ),
                "bootstrap_ci_high": mean(
                    "bootstrap_ci_high"
                ),
            }
        )

    return pooled


def _findings(
    pooled: list[dict[str, Any]],
    *,
    min_rows: int,
    min_positive_windows: int,
    max_findings: int,
) -> list[dict[str, Any]]:
    candidates = []

    for row in pooled:
        if row["positive_windows"] < (
            min_positive_windows
        ):
            continue

        if row["n"] < (
            min_rows
            * max(
                1,
                row["windows"],
            )
        ):
            continue

        lift = row.get("lift")
        difference = row.get(
            "mean_difference"
        )
        ci_low = row.get(
            "bootstrap_ci_low"
        )
        ci_high = row.get(
            "bootstrap_ci_high"
        )

        event_signal = (
            lift is not None
            and lift > 1.0
        )

        return_signal = (
            difference is not None
            and difference < 0
        )

        confidence_signal = (
            ci_high is not None
            and ci_high < 0
        )

        if not (
            event_signal
            or return_signal
        ):
            continue

        strength = 0.0

        if lift is not None:
            strength += max(
                0.0,
                lift - 1.0,
            )

        if difference is not None:
            strength += max(
                0.0,
                -difference,
            ) * 10.0

        if confidence_signal:
            strength += 1.0

        if ci_low is not None and ci_high is not None:
            strength += max(
                0.0,
                min(
                    1.0,
                    -ci_high,
                )
                * 10.0,
            )

        row = dict(row)

        row["discovery_score"] = float(
            strength
        )

        row["finding_reason"] = (
            "stable event lift"
            if event_signal
            else "negative return difference"
        )

        candidates.append(row)

    candidates.sort(
        key=lambda row: (
            row["discovery_score"],
            row["positive_windows"],
            row.get("lift") or 0.0,
        ),
        reverse=True,
    )

    return candidates[
        :max_findings
    ]


def _write_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: list[dict[str, Any]],
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


def _write_report(
    path: Path,
    metadata: dict[str, Any],
    findings: list[dict[str, Any]],
) -> None:
    lines = [
        "# Blankdiss Discovery Report",
        "",
        f"Generated: {metadata['created_at_utc']}",
        "",
        f"- Feature rows: {metadata['feature_rows']:,}",
        f"- Candidates: {metadata['candidate_count']:,}",
        f"- OOS results: {metadata['oos_result_count']:,}",
        f"- Findings: {len(findings):,}",
        "",
        "## Findings",
        "",
    ]

    if not findings:
        lines.append(
            "No candidates passed the discovery criteria."
        )
        lines.append("")
    else:
        lines.extend(
            [
                "| # | Target | Signal | Stress | Signal tail | Stress tail | Windows | Lift | Mean diff | CI high | Score |",
                "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for index, finding in enumerate(
            findings,
            start=1,
        ):
            def fmt(
                value: Any,
            ) -> str:
                if value is None:
                    return ""

                if isinstance(
                    value,
                    float,
                ):
                    return f"{value:.4f}"

                return str(value)

            lines.append(
                "| "
                f"{index} | "
                f"{finding['target_name']} | "
                f"{finding['signal_name']} | "
                f"{finding['stress_feature']} | "
                f"{fmt(finding['signal_tail_fraction'])} | "
                f"{fmt(finding['stress_tail_fraction'])} | "
                f"{finding['positive_windows']}/"
                f"{finding['windows']} | "
                f"{fmt(finding.get('lift'))} | "
                f"{fmt(finding.get('mean_difference'))} | "
                f"{fmt(finding.get('bootstrap_ci_high'))} | "
                f"{fmt(finding.get('discovery_score'))} |"
            )

    lines.extend(
        [
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(
                metadata["configuration"],
                indent=2,
                ensure_ascii=False,
            ),
            "```",
            "",
        ]
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run() -> None:
    config = _load_config()

    target_names = [
        str(value)
        for value in _require_list(
            config,
            "targets",
        )
    ]

    signal_names = [
        str(value)
        for value in _require_list(
            config,
            "signals",
        )
    ]

    stress_features = [
        str(value)
        for value in _require_list(
            config,
            "stress_features",
        )
    ]

    tails = [
        float(value)
        for value in _require_list(
            config,
            "tails",
        )
    ]

    stress_directions = config.get(
        "stress_directions",
        {},
    )

    if not isinstance(
        stress_directions,
        dict,
    ):
        raise ValueError(
            "'stress_directions' måste vara ett objekt."
        )

    validation = config.get(
        "validation",
        {},
    )

    if not isinstance(
        validation,
        dict,
    ):
        raise ValueError(
            "'validation' måste vara ett objekt."
        )

    min_rows = int(
        validation.get(
            "min_rows_per_window",
            DEFAULT_MIN_ROWS_PER_WINDOW,
        )
    )

    min_positive_windows = int(
        validation.get(
            "min_positive_windows",
            DEFAULT_MIN_POSITIVE_WINDOWS,
        )
    )

    max_findings = int(
        validation.get(
            "max_findings",
            DEFAULT_MAX_FINDINGS,
        )
    )

    if min_rows < 1:
        raise ValueError(
            "min_rows_per_window måste vara > 0."
        )

    if min_positive_windows < 1:
        raise ValueError(
            "min_positive_windows måste vara > 0."
        )

    if max_findings < 1:
        raise ValueError(
            "max_findings måste vara > 0."
        )

    for fraction in tails:
        if not 0 < fraction <= 1:
            raise ValueError(
                f"Ogiltig tail-fraktion: {fraction}"
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

    _validate_signal_names(
        frame,
        signal_names,
    )

    for feature in stress_features:
        build_signal(
            frame,
            feature,
        )

    targets, target_configs = _target_arrays(
        frame,
        target_names,
    )

    returns = _return_arrays(
        frame,
        target_configs,
    )

    candidates = _candidate_list(
        target_names=target_names,
        signal_names=signal_names,
        stress_features=stress_features,
        tails=tails,
        stress_directions={
            str(key): str(value)
            for key, value
            in stress_directions.items()
        },
    )

    expected_candidates = (
        len(target_names)
        * len(signal_names)
        * len(stress_features)
        * len(tails)
        * len(tails)
    )

    if len(candidates) != expected_candidates:
        raise RuntimeError(
            "Discovery candidate count mismatch: "
            f"{len(candidates)} != "
            f"{expected_candidates}"
        )

    print(
        f"Discovery candidates: {len(candidates):,}",
        flush=True,
    )

    window_masks = _build_window_masks(
        frame
    )

    all_signal_names = sorted(
        set(signal_names)
        | set(stress_features)
    )

    signal_values = {
        name: build_signal(
            frame,
            name,
        ).to_numpy(
            dtype=float
        )
        for name in all_signal_names
    }

    tail_masks = _build_tail_masks(
        frame,
        all_signal_names,
        tails,
        {
            **{
                name: "upper"
                for name in signal_names
            },
            **{
                str(key): str(value)
                for key, value
                in stress_directions.items()
            },
        },
    )

    results: list[
        dict[str, Any]
    ] = []

    test_windows = list(
        window_masks.items()
    )

    for window_name, masks in test_windows:
        test_mask = masks["test"]

        print(
            f"{window_name}: "
            f"evaluating {len(candidates):,} "
            "candidates on test/OOS",
            flush=True,
        )

        for index, candidate in enumerate(
            candidates,
            start=1,
        ):
            signal_name = candidate[
                "signal_name"
            ]

            stress_name = candidate[
                "stress_feature"
            ]

            signal_tail = tail_masks[
                _tail_key(
                    signal_name,
                    "upper",
                    candidate[
                        "signal_tail_fraction"
                    ],
                )
            ]

            stress_tail = tail_masks[
                _tail_key(
                    stress_name,
                    candidate[
                        "stress_tail_direction"
                    ],
                    candidate[
                        "stress_tail_fraction"
                    ],
                )
            ]

            target = targets[
                candidate["target_name"]
            ]

            return_column = (
                target_configs[
                    candidate["target_name"]
                ].return_column
            )

            if return_column not in returns:
                raise ValueError(
                    "Saknar return-array för target "
                    f"{candidate['target_name']}: "
                    f"{return_column}"
                )

            result = _evaluate_candidate(
                frame=frame,
                target=target,
                returns=returns[
                    return_column
                ],
                signal=signal_values[
                    signal_name
                ],
                stress=signal_values[
                    stress_name
                ],
                signal_tail=signal_tail,
                stress_tail=stress_tail,
                window_mask=test_mask,
                candidate=candidate,
                window_name=window_name,
            )

            results.append(result)

            if (
                index % 100 == 0
                or index == len(candidates)
            ):
                print(
                    f"  {index:,}/{len(candidates):,}",
                    flush=True,
                )

    pooled = _pool_results(
        results
    )

    findings = _findings(
        pooled,
        min_rows=min_rows,
        min_positive_windows=min_positive_windows,
        max_findings=max_findings,
    )

    run_timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    run_dir = (
        OUTPUT_DIR
        / run_timestamp
    )

    latest_dir = (
        OUTPUT_DIR
        / "latest"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "created_at_utc": run_timestamp,
        "feature_rows": int(
            len(frame)
        ),
        "candidate_count": int(
            len(candidates)
        ),
        "oos_result_count": int(
            len(results)
        ),
        "pooled_candidate_count": int(
            len(pooled)
        ),
        "finding_count": int(
            len(findings)
        ),
        "targets": target_names,
        "signals": signal_names,
        "stress_features": stress_features,
        "tails": tails,
        "walk_forward_windows": [
            {
                "name": _window_name(index),
                "train_end": str(
                    window.train_end
                ),
                "validation_end": str(
                    window.validation_end
                ),
                "test_end": str(
                    window.test_end
                ),
            }
            for index, window
            in enumerate(
                WALK_FORWARD_WINDOWS
            )
        ],
        "configuration": config,
    }

    for directory in (
        run_dir,
        latest_dir,
    ):
        _write_jsonl(
            directory / "results.jsonl",
            results,
        )

        _write_json(
            directory / "pooled.json",
            pooled,
        )

        _write_json(
            directory / "findings.json",
            findings,
        )

        _write_json(
            directory / "metadata.json",
            metadata,
        )

        _write_report(
            directory / "report.md",
            metadata,
            findings,
        )

    print(
        "",
        flush=True,
    )

    print(
        f"Discovery candidates: "
        f"{len(candidates):,}",
        flush=True,
    )

    print(
        f"Discovery OOS results: "
        f"{len(results):,}",
        flush=True,
    )

    print(
        f"Discovery findings: "
        f"{len(findings):,}",
        flush=True,
    )

    print(
        f"Discovery complete: {run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    run()
