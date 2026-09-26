"""Autonomous, controlled research loop for Blankdiss AI Lab.

The workflow only starts this module and later commits generated artifacts.
This module owns the research loop:

    OBSERVE -> ANALYZE -> ADAPT -> EXTEND -> OBSERVE ...

Important boundaries:
- validation data is used for adaptive decisions;
- test data is never used to choose parameters or hypotheses;
- locked prospective confirmation is never modified or executed;
- every generated spec is written to disk and then read back before execution;
- every experiment result is written to disk and then read back before analysis;
- outcome classification is descriptive, not a search for positive effects;
- parameter candidates are traversed in a fixed order, independent of result size;
- when the finite parameter space is exhausted, the engine creates a new
  controlled experiment family rather than silently stopping.

The extension mechanism uses a fixed, reviewed code template. It does not
execute arbitrary generated shell commands or arbitrary AI-generated Python.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# When this file is executed directly as:
#
#     python -u ml/ai-lab/adaptive_research.py
#
# Python puts ml/ai-lab on sys.path, not the repository root. Add the
# repository root explicitly so imports such as `ml.research.engine`
# resolve independently of the current working directory.
ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from ml.research.engine import run_spec
from ml.research.reporting import write_json
from ml.research.session import build_session
from ml.research.spec import load_spec


SPEC_DIR = (
    ROOT
    / "ml"
    / "research"
    / "specs"
)

RUNS_DIR = (
    ROOT
    / "data"
    / "processed"
    / "ml"
    / "research"
    / "spec_runs"
)

DISCOVERY_DIR = (
    ROOT
    / "ml"
    / "research"
    / "discovery"
)

STATE_DIR = (
    ROOT
    / "data"
    / "ai_lab"
    / "adaptive_research"
)

STATE_PATH = (
    STATE_DIR
    / "state.json"
)


SOURCE_SPEC_ID = (
    "momentum_si_regime_walk_forward"
)

LOCKED_CONFIRMATION_ID = (
    "momentum_si_prospective_confirmation"
)

ADAPTIVE_PREFIX = (
    "adaptive_momentum_si_"
)

# The original walk-forward experiment already evaluated:
#
#   0.20, 0.10, 0.05, 0.025
#
# The adaptive engine adds a new, explicitly declared outer point:
#
#   0.30
#
# The complete adaptive grid is traversed deterministically.
#
# The order is fixed BEFORE any result is observed.
ADAPTIVE_FRACTIONS = (
    0.30,
    0.20,
    0.10,
    0.05,
    0.025,
)

MIN_VALID_N = 100

SIGN_EPSILON = 1e-12

# Once the finite parameter space is exhausted, the research process
# changes experiment family instead of returning "no proposal".
EXTENSION_FAMILIES = (
    "target_profile",
)


def utc_now() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_id(
    value: str,
) -> str:
    """Make a repository-safe identifier."""
    return re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        value,
    ).strip("_")


def _read_json(
    path: Path,
) -> dict[str, Any] | None:
    """Read a JSON object from disk."""
    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    if not isinstance(
        payload,
        dict,
    ):
        return None

    return payload


def _write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Persist a JSON object."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        path,
        payload,
    )


def _load_yaml(
    path: Path,
) -> dict[str, Any]:
    """Read a research YAML specification."""
    payload = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"Research spec must be an object: {path}"
        )

    return payload


def _completed_spec_ids() -> set[str]:
    """Return specification IDs represented by persisted research runs."""
    completed: set[str] = set()

    if not RUNS_DIR.exists():
        return completed

    for manifest_path in RUNS_DIR.glob(
        "*/manifest.json"
    ):
        payload = _read_json(
            manifest_path
        )

        if not payload:
            continue

        for item in payload.get(
            "specs",
            [],
        ):
            if isinstance(
                item,
                str,
            ):
                completed.add(
                    item
                )
                continue

            if not isinstance(
                item,
                dict,
            ):
                continue

            value = (
                item.get("id")
                or item.get("spec_id")
            )

            if value is not None:
                completed.add(
                    str(value)
                )

    return completed


def _source_spec() -> dict[str, Any]:
    """Load and validate the adaptive source specification."""
    path = (
        SPEC_DIR
        / f"{SOURCE_SPEC_ID}.yaml"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Source spec does not exist: {path}"
        )

    payload = _load_yaml(
        path
    )

    metadata = payload.get(
        "metadata",
        {},
    )

    if not isinstance(
        metadata,
        dict,
    ):
        metadata = {}

    if metadata.get(
        "locked"
    ) is True:
        raise ValueError(
            "Source adaptive spec is unexpectedly locked."
        )

    if (
        str(
            metadata.get(
                "stage",
                "",
            )
        ).lower()
        == "migration"
    ):
        raise ValueError(
            "Migration spec cannot be an adaptive source."
        )

    return payload


def _existing_adaptive_specs() -> dict[
    str,
    dict[str, Any],
]:
    """Load already-created adaptive specifications."""
    result: dict[
        str,
        dict[str, Any],
    ] = {}

    for path in SPEC_DIR.glob(
        f"{ADAPTIVE_PREFIX}*.yaml"
    ):
        try:
            payload = _load_yaml(
                path
            )
        except Exception:
            continue

        spec_id = payload.get(
            "id"
        )

        if spec_id:
            result[
                str(spec_id)
            ] = payload

    return result


def _candidate_grid(
    source: dict[str, Any],
) -> list[
    tuple[
        float,
        float,
        str,
    ]
]:
    """Return the complete, deterministic adaptive candidate grid."""
    targets = source.get(
        "targets",
        [],
    )

    if not targets:
        raise ValueError(
            "Source spec has no targets."
        )

    return [
        (
            float(
                baseline
            ),
            float(
                incremental
            ),
            str(
                target
            ),
        )
        for baseline, incremental, target in itertools.product(
            ADAPTIVE_FRACTIONS,
            ADAPTIVE_FRACTIONS,
            [
                str(target)
                for target in targets
            ],
        )
    ]


def _candidate_key(
    baseline: float,
    incremental: float,
    target: str,
) -> str:
    """Return a canonical parameter-space key."""
    return (
        f"{baseline:.4f}|"
        f"{incremental:.4f}|"
        f"{target}"
    )


def _spec_candidate(
    spec: dict[str, Any],
) -> tuple[
    float,
    float,
    str,
] | None:
    """Extract the single candidate represented by an adaptive spec."""
    try:
        signals = spec[
            "signals"
        ]

        baseline = float(
            signals[0][
                "bins"
            ][0]
        )

        incremental = float(
            signals[1][
                "bins"
            ][0]
        )

        target = str(
            spec[
                "targets"
            ][0]
        )

        return (
            baseline,
            incremental,
            target,
        )

    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
    ):
        return None


def _generated_candidate_ids() -> dict[
    str,
    str,
]:
    """Map adaptive parameter points to generated specification IDs."""
    mapping: dict[
        str,
        str,
    ] = {}

    for (
        spec_id,
        spec,
    ) in _existing_adaptive_specs().items():
        candidate = _spec_candidate(
            spec
        )

        if candidate is None:
            continue

        mapping[
            _candidate_key(
                *candidate
            )
        ] = spec_id

    return mapping


def _source_candidate_keys(
    source: dict[str, Any],
) -> set[str]:
    """Return all parameter points already tested by the source spec."""
    signals = source.get(
        "signals",
        [],
    )

    if len(signals) != 2:
        return set()

    baseline_bins = [
        float(value)
        for value in signals[0].get(
            "bins",
            [],
        )
    ]

    incremental_bins = [
        float(value)
        for value in signals[1].get(
            "bins",
            [],
        )
    ]

    targets = [
        str(value)
        for value in source.get(
            "targets",
            [],
        )
    ]

    return {
        _candidate_key(
            baseline,
            incremental,
            target,
        )
        for (
            baseline,
            incremental,
            target,
        ) in itertools.product(
            baseline_bins,
            incremental_bins,
            targets,
        )
    }


def _build_adaptive_spec(
    source: dict[str, Any],
    baseline_fraction: float,
    incremental_fraction: float,
    target: str,
) -> dict[str, Any]:
    """Build one controlled adaptive research specification."""
    source_signals = source.get(
        "signals",
        [],
    )

    if len(source_signals) != 2:
        raise ValueError(
            "Adaptive source must define exactly two signals."
        )

    baseline_signal = (
        source_signals[0]
    )

    incremental_signal = (
        source_signals[1]
    )

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


def _write_and_read_spec(
    payload: dict[str, Any],
) -> tuple[
    Path,
    Any,
]:
    """Write an adaptive spec and reconstruct it from disk."""
    path = (
        SPEC_DIR
        / f"{payload['id']}.yaml"
    )

    if path.exists():
        existing = _load_yaml(
            path
        )

        if existing != payload:
            raise ValueError(
                f"Refusing to overwrite existing spec: {path}"
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

    # Important architectural boundary:
    #
    # The executable specification comes from the persisted YAML,
    # not from the in-memory proposal.
    read_back = load_spec(
        path
    )

    if read_back.id != payload[
        "id"
    ]:
        raise ValueError(
            "Spec read-back changed the experiment id."
        )

    return (
        path,
        read_back,
    )


def _run_research_spec(
    spec: Any,
) -> Path:
    """Execute one research specification and persist its result."""
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

    session = build_session(
        [spec]
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
        "created_at_utc": utc_now(),
        "feature_rows": int(
            len(
                session.frame
            )
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
            }
        ],
    }

    write_json(
        run_dir
        / "manifest.json",
        manifest,
    )

    # Important architectural boundary:
    #
    # Analysis consumes the persisted result, not the object returned
    # directly from run_spec().
    persisted = _read_json(
        result_path
    )

    if (
        not isinstance(
            persisted,
            dict,
        )
        or not isinstance(
            persisted.get(
                "results"
            ),
            list,
        )
    ):
        raise ValueError(
            "Persisted result could not be read back: "
            f"{result_path}"
        )

    return result_path


def _load_result(
    path: Path,
) -> dict[str, Any]:
    """Load a persisted research result."""
    payload = _read_json(
        path
    )

    if (
        not payload
        or not isinstance(
            payload.get(
                "results"
            ),
            list,
        )
    ):
        raise ValueError(
            f"Invalid persisted research result: {path}"
        )

    return payload


def classify_outcome(
    rows: list[dict[str, Any]],
) -> str:
    """Classify observed validation effects neutrally.

    Classification is performed after the experiment. It is never used
    to decide which candidate to test next.

    POSITIVE:
        all sufficiently populated validation observations are positive.

    NEGATIVE:
        all sufficiently populated validation observations are negative.

    MIXED:
        both positive and negative observations occur.

    INCONCLUSIVE:
        too little usable evidence or no directional effect.
    """
    validation = [
        row
        for row in rows
        if (
            isinstance(
                row,
                dict,
            )
            and str(
                row.get(
                    "split",
                    "",
                )
            ).lower()
            == "validation"
        )
    ]

    differences: list[
        float
    ] = []

    for row in validation:
        try:
            n = int(
                row.get(
                    "combined_n",
                    row.get(
                        "n",
                        0,
                    ),
                )
            )

            difference = float(
                row[
                    "absolute_event_rate_difference"
                ]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        if (
            n >= MIN_VALID_N
            and math.isfinite(
                difference
            )
        ):
            differences.append(
                difference
            )

    if len(
        differences
    ) < 2:
        return (
            "OUTCOME_INCONCLUSIVE"
        )

    positive = any(
        value > SIGN_EPSILON
        for value in differences
    )

    negative = any(
        value < -SIGN_EPSILON
        for value in differences
    )

    if positive and negative:
        return "OUTCOME_MIXED"

    if positive:
        return "OUTCOME_POSITIVE"

    if negative:
        return "OUTCOME_NEGATIVE"

    return "OUTCOME_INCONCLUSIVE"


def _summarize_result(
    path: Path,
) -> dict[str, Any]:
    """Create a descriptive summary from a persisted result."""
    payload = _load_result(
        path
    )

    rows = payload[
        "results"
    ]

    outcome = classify_outcome(
        rows
    )

    differences: list[
        float
    ] = []

    for row in rows:
        if (
            not isinstance(
                row,
                dict,
            )
            or str(
                row.get(
                    "split",
                    "",
                )
            ).lower()
            != "validation"
        ):
            continue

        value = row.get(
            "absolute_event_rate_difference"
        )

        if (
            isinstance(
                value,
                (
                    int,
                    float,
                ),
            )
            and math.isfinite(
                float(value)
            )
        ):
            differences.append(
                float(value)
            )

    return {
        "path": str(
            path.relative_to(
                ROOT
            )
        ),
        "experiment_id": payload.get(
            "id"
        ),
        "outcome": outcome,
        "validation_difference_count": len(
            differences
        ),
        "validation_min_difference": (
            min(differences)
            if differences
            else None
        ),
        "validation_max_difference": (
            max(differences)
            if differences
            else None
        ),
    }


def _all_adaptive_summaries() -> list[
    dict[str, Any]
]:
    """Read all persisted adaptive experiment results."""
    summaries: list[
        dict[str, Any]
    ] = []

    if not RUNS_DIR.exists():
        return summaries

    for manifest_path in sorted(
        RUNS_DIR.glob(
            "*/manifest.json"
        )
    ):
        manifest = _read_json(
            manifest_path
        )

        if not manifest:
            continue

        for item in manifest.get(
            "specs",
            [],
        ):
            if not isinstance(
                item,
                dict,
            ):
                continue

            spec_id = str(
                item.get(
                    "id",
                    "",
                )
            )

            if not spec_id.startswith(
                ADAPTIVE_PREFIX
            ):
                continue

            result_path = (
                ROOT
                / str(
                    item.get(
                        "result",
                        "",
                    )
                )
            )

            if result_path.is_file():
                summaries.append(
                    _summarize_result(
                        result_path
                    )
                )

    return summaries


def _choose_next_candidate(
    source: dict[str, Any],
) -> tuple[
    float,
    float,
    str,
] | None:
    """Choose the next point by fixed grid order only."""
    completed = _completed_spec_ids()

    generated = (
        _generated_candidate_ids()
    )

    source_keys = (
        _source_candidate_keys(
            source
        )
    )

    for (
        baseline,
        incremental,
        target,
    ) in _candidate_grid(
        source
    ):
        key = _candidate_key(
            baseline,
            incremental,
            target,
        )

        # The original walk-forward grid already tested these exact
        # points. They are observed evidence, not new adaptive candidates.
        if key in source_keys:
            continue

        spec_id = generated.get(
            key
        )

        if (
            spec_id is not None
            and spec_id in completed
        ):
            continue

        return (
            baseline,
            incremental,
            target,
        )

    return None


def _state_payload(
    phase: str,
    history: list[
        dict[str, Any]
    ],
    next_candidate: tuple[
        float,
        float,
        str,
    ] | None,
    extension_family: str | None = None,
) -> dict[str, Any]:
    """Build the persisted adaptive-engine state."""
    return {
        "state_version": 1,
        "updated_at_utc": utc_now(),
        "phase": phase,
        "history": history,
        "next_candidate": (
            {
                "baseline_fraction": (
                    next_candidate[0]
                ),
                "incremental_fraction": (
                    next_candidate[1]
                ),
                "target": (
                    next_candidate[2]
                ),
            }
            if next_candidate
            else None
        ),
        "extension_family": (
            extension_family
        ),
        "locked_boundary": {
            "confirmation_spec": (
                LOCKED_CONFIRMATION_ID
            ),
            "confirmation_may_not_be_executed_or_modified": True,
        },
    }


def _write_state(
    payload: dict[str, Any],
) -> None:
    """Persist the adaptive state."""
    _write_json(
        STATE_PATH,
        payload,
    )


def _target_profile_code(
    module_name: str,
) -> str:
    """Return the fixed template for the first extension family."""
    return f'''"""Controlled adaptive target-profile experiment generated by AI Lab."""

from __future__ import annotations

from typing import Any


def run(
    cache,
    signal_name: str,
    direction: str,
    fraction: float,
    targets: list[str],
) -> dict[str, Any]:
    """Report target event rates for one fixed regime.

    No target is selected by this experiment. Every supplied target
    is reported in the same deterministic order.
    """
    mask = cache.tail_masks[
        f"{{signal_name}}|{{direction}}|{{fraction}}"
    ]

    window = cache.window_masks[
        "window_2"
    ][
        "validation"
    ]

    rows = []

    for target_name in targets:
        target = cache.targets[
            target_name
        ]

        selected = (
            window
            & mask
        )

        valid = selected

        n = int(
            valid.sum()
        )

        events = int(
            (
                target[valid] > 0
            ).sum()
        ) if n else 0

        rows.append(
            {{
                "target": target_name,
                "n": n,
                "events": events,
                "event_rate": (
                    events / n
                    if n
                    else None
                ),
            }}
        )

    return {{
        "experiment_family": "{module_name}",
        "rows": rows,
    }}
'''


def _extend(
    history: list[
        dict[str, Any]
    ],
    family: str,
) -> dict[str, Any]:
    """Create a new controlled experiment-family module."""
    DISCOVERY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    module_name = (
        f"adaptive_{family}_experiment"
    )

    path = (
        DISCOVERY_DIR
        / f"{module_name}.py"
    )

    if family == "target_profile":
        content = _target_profile_code(
            module_name
        )
    else:
        raise ValueError(
            f"Unknown extension family: {family}"
        )

    if path.exists():
        existing = path.read_text(
            encoding="utf-8"
        )

        if existing != content:
            raise ValueError(
                "Refusing to overwrite extension code: "
                f"{path}"
            )

    else:
        path.write_text(
            content,
            encoding="utf-8",
        )

    # Compile before execution.
    compile(
        path.read_text(
            encoding="utf-8"
        ),
        str(path),
        "exec",
    )

    return {
        "phase": "EXTEND",
        "family": family,
        "code_path": str(
            path.relative_to(
                ROOT
            )
        ),
        "reason": (
            "finite_parameter_space_exhausted"
        ),
        "history_count": len(
            history
        ),
    }


def _execute_target_profile_extension(
    source: dict[str, Any],
) -> dict[str, Any]:
    """Execute the newly created controlled extension experiment."""
    module_path = (
        DISCOVERY_DIR
        / "adaptive_target_profile_experiment.py"
    )

    module_name = (
        "blankdiss_adaptive_target_profile"
    )

    spec = (
        importlib.util.spec_from_file_location(
            module_name,
            module_path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise ValueError(
            "Could not load target-profile extension."
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    signals = source[
        "signals"
    ]

    signal_name = str(
        signals[0][
            "name"
        ]
    )

    direction = str(
        signals[0].get(
            "direction",
            "lower",
        )
    )

    fraction = float(
        signals[0].get(
            "bins",
            [0.20],
        )[0]
    )

    targets = [
        str(value)
        for value in source[
            "targets"
        ]
    ]

    source_spec = load_spec(
        SPEC_DIR
        / f"{SOURCE_SPEC_ID}.yaml"
    )

    session = build_session(
        [source_spec]
    )

    result = module.run(
        session.cache,
        signal_name,
        direction,
        fraction,
        targets,
    )

    path = (
        RUNS_DIR
        / datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        / "adaptive_target_profile.json"
    )

    _write_json(
        path,
        result,
    )

    persisted = _read_json(
        path
    )

    if persisted is None:
        raise ValueError(
            "Extension result could not be read back."
        )

    return {
        "path": str(
            path.relative_to(
                ROOT
            )
        ),
        "result": persisted,
    }


def run() -> dict[str, Any]:
    """Run the complete autonomous research cycle."""
    history: list[
        dict[str, Any]
    ] = []

    source = _source_spec()

    while True:
        # ----------------------------------------------------------
        # OBSERVE
        # ----------------------------------------------------------
        #
        # State and results are read from disk on every cycle.
        # The persisted files are the source of truth.
        _ = _read_json(
            STATE_PATH
        )

        next_candidate = (
            _choose_next_candidate(
                source
            )
        )

        # ----------------------------------------------------------
        # ADAPT
        # ----------------------------------------------------------
        if next_candidate is not None:
            (
                baseline,
                incremental,
                target,
            ) = next_candidate

            _write_state(
                _state_payload(
                    "ADAPT",
                    history,
                    next_candidate,
                )
            )

            proposal = _build_adaptive_spec(
                source,
                baseline,
                incremental,
                target,
            )

            (
                spec_path,
                parsed_spec,
            ) = _write_and_read_spec(
                proposal
            )

            # Never allow adaptive code to cross the locked boundary.
            metadata = _load_yaml(
                spec_path
            ).get(
                "metadata",
                {},
            )

            if (
                metadata.get(
                    "locked"
                )
                is True
                or str(
                    metadata.get(
                        "stage",
                        "",
                    )
                ).lower()
                == "migration"
            ):
                raise ValueError(
                    "Adaptive engine refused protected spec: "
                    f"{spec_path}"
                )

            # ------------------------------------------------------
            # EXPERIMENT
            # ------------------------------------------------------
            result_path = (
                _run_research_spec(
                    parsed_spec
                )
            )

            # ------------------------------------------------------
            # ANALYZE
            # ------------------------------------------------------
            summary = (
                _summarize_result(
                    result_path
                )
            )

            history.append(
                {
                    "phase": "ANALYZE",
                    "spec_id": (
                        parsed_spec.id
                    ),
                    "candidate": {
                        "baseline_fraction": (
                            baseline
                        ),
                        "incremental_fraction": (
                            incremental
                        ),
                        "target": target,
                    },
                    "outcome": (
                        summary[
                            "outcome"
                        ]
                    ),
                    "result_path": (
                        summary[
                            "path"
                        ]
                    ),
                }
            )

            _write_state(
                _state_payload(
                    "OBSERVE",
                    history,
                    _choose_next_candidate(
                        source
                    ),
                )
            )

            continue

        # ----------------------------------------------------------
        # EXTEND
        # ----------------------------------------------------------
        #
        # The finite parameter space is now exhausted.
        # This is deliberately not a "no proposal" failure.
        summaries = (
            _all_adaptive_summaries()
        )

        if not summaries:
            raise RuntimeError(
                "Parameter space is exhausted without "
                "persisted adaptive results."
            )

        completed_extensions = {
            item.get(
                "extension_family"
            )
            for item in history
            if item.get(
                "phase"
            )
            == "EXTEND"
        }

        family = next(
            (
                candidate
                for candidate
                in EXTENSION_FAMILIES
                if candidate
                not in completed_extensions
            ),
            None,
        )

        if family is None:
            _write_state(
                _state_payload(
                    "COMPLETE",
                    history,
                    None,
                )
            )

            return {
                "status": (
                    "research_cycle_complete"
                ),
                "history": history,
                "outcomes": summaries,
                "reason": (
                    "parameter_space_and_declared_"
                    "extension_families_exhausted"
                ),
            }

        extension = _extend(
            history,
            family,
        )

        history.append(
            extension
        )

        _write_state(
            _state_payload(
                "EXTEND",
                history,
                None,
                family,
            )
        )

        # Execute the new experiment family.
        executed = (
            _execute_target_profile_extension(
                source
            )
        )

        history.append(
            {
                "phase": "OBSERVE",
                "extension_family": family,
                "result_path": (
                    executed[
                        "path"
                    ]
                ),
                "observation": (
                    executed[
                        "result"
                    ]
                ),
            }
        )

        _write_state(
            _state_payload(
                "COMPLETE",
                history,
                None,
                family,
            )
        )

        return {
            "status": (
                "research_cycle_extended"
            ),
            "history": history,
            "outcomes": summaries,
            "reason": (
                "new_controlled_experiment_"
                "family_created_and_executed"
            ),
        }


def main() -> int:
    """CLI entry point."""
    result = run()

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
