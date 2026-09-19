from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FrozenCandidateConfig:
    candidate_id: str
    target_name: str
    signal_name: str
    signal_tail: float
    stress_feature: str
    stress_tail: float
    stress_direction: str


@dataclass(frozen=True)
class FrozenEvaluationConfig:
    start_date: date
    end_date: date


@dataclass(frozen=True)
class FrozenNullConfig:
    metric: str
    permutations: int
    seed: int


@dataclass(frozen=True)
class FrozenHypothesisConfig:
    version: int
    hypothesis_id: str
    question: str
    discovery_run: str
    discovery_end_date: date
    candidate: FrozenCandidateConfig
    evaluation: FrozenEvaluationConfig
    null_test: FrozenNullConfig


def _require_string(
    value: Any,
    name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Frozen hypothesis '{name}' måste vara en icke-tom sträng."
        )

    return value.strip()


def _parse_date(
    value: Any,
    name: str,
) -> date:
    text = _require_string(
        value,
        name,
    )

    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"Ogiltigt datum för '{name}': {text}"
        ) from exc


def _parse_candidate(
    raw: dict[str, Any],
) -> FrozenCandidateConfig:
    signal_tail = float(
        raw["signal_tail"]
    )

    stress_tail = float(
        raw["stress_tail"]
    )

    if not 0 < signal_tail <= 1:
        raise ValueError(
            "signal_tail måste vara > 0 och <= 1."
        )

    if not 0 < stress_tail <= 1:
        raise ValueError(
            "stress_tail måste vara > 0 och <= 1."
        )

    stress_direction = _require_string(
        raw["stress_direction"],
        "stress_direction",
    )

    if stress_direction not in {
        "upper",
        "lower",
    }:
        raise ValueError(
            "stress_direction måste vara 'upper' eller 'lower'."
        )

    return FrozenCandidateConfig(
        candidate_id=_require_string(
            raw["candidate_id"],
            "candidate_id",
        ),
        target_name=_require_string(
            raw["target_name"],
            "target_name",
        ),
        signal_name=_require_string(
            raw["signal_name"],
            "signal_name",
        ),
        signal_tail=signal_tail,
        stress_feature=_require_string(
            raw["stress_feature"],
            "stress_feature",
        ),
        stress_tail=stress_tail,
        stress_direction=stress_direction,
    )


def load_config() -> FrozenHypothesisConfig:
    config_path = Path(
        os.environ.get(
            "FROZEN_HYPOTHESIS_CONFIG",
            "ml/research/frozen/hypothesis.yml",
        )
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(
            "Frozen hypothesis config måste vara ett YAML-objekt."
        )

    if int(raw.get("version", 0)) != 1:
        raise ValueError(
            "Okänd frozen hypothesis config-version."
        )

    hypothesis = raw.get("hypothesis")

    if not isinstance(hypothesis, dict):
        raise ValueError(
            "'hypothesis' måste vara ett YAML-objekt."
        )

    if hypothesis.get("status") != "frozen":
        raise ValueError(
            "Hypotesen måste ha status='frozen'."
        )

    source = hypothesis.get("source")

    if not isinstance(source, dict):
        raise ValueError(
            "'source' måste vara ett YAML-objekt."
        )

    candidate_raw = hypothesis.get("candidate")

    if not isinstance(candidate_raw, dict):
        raise ValueError(
            "'candidate' måste vara ett YAML-objekt."
        )

    evaluation_raw = hypothesis.get("evaluation")

    if not isinstance(evaluation_raw, dict):
        raise ValueError(
            "'evaluation' måste vara ett YAML-objekt."
        )

    null_raw = hypothesis.get("null_test")

    if not isinstance(null_raw, dict):
        raise ValueError(
            "'null_test' måste vara ett YAML-objekt."
        )

    metric = _require_string(
        null_raw.get("metric", "lift"),
        "null_test.metric",
    )

    if metric not in {
        "lift",
        "return_difference",
    }:
        raise ValueError(
            "null_test.metric måste vara 'lift' "
            "eller 'return_difference'."
        )

    result = FrozenHypothesisConfig(
        version=1,
        hypothesis_id=_require_string(
            hypothesis["id"],
            "id",
        ),
        question=_require_string(
            hypothesis["question"],
            "question",
        ),
        discovery_run=_require_string(
            source["discovery_run"],
            "discovery_run",
        ),
        discovery_end_date=_parse_date(
            source["discovery_end_date"],
            "discovery_end_date",
        ),
        candidate=_parse_candidate(
            candidate_raw
        ),
        evaluation=FrozenEvaluationConfig(
            start_date=_parse_date(
                evaluation_raw["start_date"],
                "evaluation.start_date",
            ),
            end_date=_parse_date(
                evaluation_raw["end_date"],
                "evaluation.end_date",
            ),
        ),
        null_test=FrozenNullConfig(
            metric=metric,
            permutations=int(
                null_raw.get(
                    "permutations",
                    1000,
                )
            ),
            seed=int(
                null_raw.get(
                    "seed",
                    42,
                )
            ),
        ),
    )

    if (
        result.evaluation.start_date
        <= result.discovery_end_date
    ):
        raise ValueError(
            "OOS-perioden måste börja efter discovery_end_date."
        )

    if (
        result.evaluation.end_date
        < result.evaluation.start_date
    ):
        raise ValueError(
            "evaluation.end_date måste vara >= evaluation.start_date."
        )

    if result.null_test.permutations < 1:
        raise ValueError(
            "Antalet null-permutationer måste vara >= 1."
        )

    return result
