from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hashlib
import yaml

from ml.research.engine import run_spec
from ml.research.reporting import write_json
from ml.research.spec import load_spec

from config import (
    ROOT,
    RUNS_DIR,
    SPEC_DIR,
)

from adaptive_config import (
    ADAPTIVE_PREFIX,
    SOURCE_SPEC_ID,
)

from state import read_json


EXECUTION_FINGERPRINT_VERSION = "1"


def execution_fingerprint(
    spec,
) -> str:
    """Return a fingerprint for the exact spec and research execution code."""
    digest = hashlib.sha256()

    spec_path = (
        SPEC_DIR
        / f"{spec.id}.yaml"
    )

    digest.update(
        spec_path.read_bytes()
    )

    for relative_path in (
        "ml/research/engine.py",
        "ml/research/conditional.py",
        "ml/research/spec.py",
        "ml/research/signals.py",
    ):
        path = ROOT / relative_path

        digest.update(
            relative_path.encode(
                "utf-8"
            )
        )

        digest.update(
            path.read_bytes()
        )

    digest.update(
        EXECUTION_FINGERPRINT_VERSION.encode(
            "utf-8"
        )
    )

    return digest.hexdigest()


def safe_id(
    value: str,
) -> str:
    import re

    return re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        value,
    ).strip("_")


def build_adaptive_spec(
    source: dict[str, Any],
    baseline_fraction: float,
    incremental_fraction: float,
    target: str,
) -> dict[str, Any]:
    source_signals = source.get(
        "signals",
        [],
    )

    if len(source_signals) != 2:
        raise ValueError(
            "Adaptive source must define exactly two signals."
        )

    baseline_signal = source_signals[0]
    incremental_signal = source_signals[1]

    spec_id = (
        f"{ADAPTIVE_PREFIX}"
        f"b{baseline_fraction:.3f}_"
        f"si{incremental_fraction:.3f}_"
        f"{safe_id(target)}"
    ).replace(
        ".",
        "p",
    )

    return {
        "id": spec_id,
        "question": (
            "Kontrollerad förfining av den "
            "fördefinierade momentum/SI-regimen. "
            "Vilket utfall observeras för denna "
            "parameterpunkt i validation-data?"
        ),
        "mode": "deep",
        "signals": [
            {
                "name": baseline_signal[
                    "name"
                ],
                "direction": baseline_signal.get(
                    "direction",
                    "lower",
                ),
                "bins": [
                    baseline_fraction
                ],
            },
            {
                "name": incremental_signal[
                    "name"
                ],
                "direction": incremental_signal.get(
                    "direction",
                    "upper",
                ),
                "bins": [
                    incremental_fraction
                ],
            },
        ],
        "targets": [
            target
        ],
        "analysis": {
            "type": "regime_comparison",
            "bootstrap": True,
            "bootstrap_iterations": 2000,
        },
        "windows": list(
            source.get(
                "windows",
                [
                    "window_1",
                    "window_2",
                ],
            )
        ),
        "splits": [
            "validation"
        ],
        "metadata": {
            "stage": "adaptive_refinement",
            "purpose": (
                "controlled_parameter_space_exploration"
            ),
            "source_spec": SOURCE_SPEC_ID,
            "selection_policy": (
                "fixed_predeclared_grid_order"
            ),
            "candidate": {
                "baseline_fraction": (
                    baseline_fraction
                ),
                "incremental_fraction": (
                    incremental_fraction
                ),
                "target": target,
            },
            "rules": [
                "validation_only_for_adaptation",
                "test_data_never_selects_parameters",
                "no_result_based_candidate_ranking",
                "no_locked_spec_modification",
                "no_locked_confirmation_execution",
            ],
        },
    }


def write_and_read_spec(
    payload: dict[str, Any],
):
    path = (
        SPEC_DIR
        / f"{payload['id']}.yaml"
    )

    if path.exists():
        existing = yaml.safe_load(
            path.read_text(
                encoding="utf-8"
            )
        )

        if existing != payload:
            raise ValueError(
                "Refusing to overwrite existing spec: "
                f"{path}"
            )

    else:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            yaml.safe_dump(
                payload,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

    parsed = load_spec(
        path
    )

    if parsed.id != payload[
        "id"
    ]:
        raise ValueError(
            "Spec read-back changed the experiment id."
        )

    return (
        path,
        parsed,
    )


def run_research_spec(
    session,
    spec,
) -> Path:
    run_timestamp = (
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
    )

    run_dir = (
        RUNS_DIR
        / run_timestamp
    )

    result = run_spec(
        session.cache,
        spec,
    )

    result_path = (
        run_dir
        / f"{spec.id}.json"
    )

    write_json(
        result_path,
        result,
    )

    manifest = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "feature_rows": int(
            len(session.frame)
        ),
        "specs": [
            {
                "id": spec.id,
                "mode": spec.mode,
                "question": spec.question,
                "result": str(
                    result_path.relative_to(
                        ROOT
                    )
                ),
                "rows": len(
                    result.get(
                        "results",
                        [],
                    )
                ),
                "execution_fingerprint": (
                    execution_fingerprint(
                        spec
                    )
                ),
            }
        ],
    }

    write_json(
        run_dir
        / "manifest.json",
        manifest,
    )

    persisted = read_json(
        result_path
    )

    if persisted is None:
        raise ValueError(
            "Research result could not be read back: "
            f"{result_path}"
        )

    if persisted.get(
        "spec_id"
    ) != spec.id:
        raise ValueError(
            "Research result read-back has unexpected spec_id."
        )

    return result_path
