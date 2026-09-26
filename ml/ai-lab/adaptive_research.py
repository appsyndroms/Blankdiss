"""Autonomous, controlled research loop for Blankdiss AI Lab.

The workflow only starts this module and later commits generated artifacts.

Research loop:

    OBSERVE
        ↓
    ADAPT
        ↓
    EXPERIMENT
        ↓
    ANALYZE
        ↓
    OBSERVE

When the finite parameter space is exhausted:

    EXTEND
        ↓
    new controlled experiment family
        ↓
    OBSERVE
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------------
# Import setup
# ----------------------------------------------------------------------

ROOT = Path(
    __file__
).resolve().parents[2]

AI_LAB_DIR = (
    ROOT
    / "ml"
    / "ai-lab"
)

# AI Lab contains local modules such as analysis.py, candidates.py,
# config.py, experiments.py, extension.py, session.py and state.py.
#
# AI_LAB_DIR must have priority over the repository root.
# Otherwise:
#
#     from analysis import ...
#
# may resolve to the repository-level analysis package instead of
# ml/ai-lab/analysis.py.
if str(AI_LAB_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(AI_LAB_DIR),
    )

# The repository root is still required for imports such as:
#
#     from ml.research...
#
# Keep it available after the AI Lab directory.
if str(ROOT) not in sys.path:
    sys.path.append(
        str(ROOT),
    )


from analysis import (
    all_adaptive_summaries,
    summarize_result,
)
from adaptive_config import (
    LOCKED_CONFIRMATION_ID,
    SOURCE_SPEC_ID,
)
from candidates import (
    choose_next_candidate,
)
from config import (
    EXTENSION_FAMILIES,
    SPEC_DIR,
)
from experiments import (
    build_adaptive_spec,
    run_research_spec,
    write_and_read_spec,
)
from extension import (
    execute_target_profile_extension,
    extend,
)
from session import (
    build_shared_session,
)
from state import (
    build_state,
    read_state,
    write_state,
)


# ----------------------------------------------------------------------
# Source specification
# ----------------------------------------------------------------------


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    import yaml

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


def source_spec() -> dict[str, Any]:
    path = (
        SPEC_DIR
        / f"{SOURCE_SPEC_ID}.yaml"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Source spec does not exist: {path}"
        )

    payload = load_yaml(
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

    if str(
        metadata.get(
            "stage",
            "",
        )
    ).lower() == "migration":
        raise ValueError(
            "Migration spec cannot be an adaptive source."
        )

    return payload


# ----------------------------------------------------------------------
# Main research cycle
# ----------------------------------------------------------------------


def run() -> dict[str, Any]:
    # --------------------------------------------------------------
    # OBSERVE
    # --------------------------------------------------------------
    #
    # The AI Lab state is persistent and disk-backed.
    #
    # Previous history must be restored before continuing. Otherwise
    # a new Python process would forget which extension families have
    # already been materialized/executed and could try to create them
    # again.
    # --------------------------------------------------------------

    previous_state = read_state()

    if previous_state:
        print(
            "Existing AI Lab state found.",
            flush=True,
        )
    else:
        print(
            "No existing AI Lab state found.",
            flush=True,
        )

    previous_history = []

    if isinstance(
        previous_state,
        dict,
    ):
        stored_history = previous_state.get(
            "history",
            [],
        )

        if isinstance(
            stored_history,
            list,
        ):
            previous_history = [
                item
                for item in stored_history
                if isinstance(
                    item,
                    dict,
                )
            ]

    history: list[
        dict[str, Any]
    ] = list(
        previous_history
    )

    source = source_spec()

    print(
        "=== START AI LAB RESEARCH ===",
        flush=True,
    )

    # --------------------------------------------------------------
    # SESSION
    # --------------------------------------------------------------
    #
    # This is deliberately outside the adaptive loop.
    #
    # load_features()
    # build_research_cache()
    #
    # therefore happen exactly once for this process.
    shared_session = build_shared_session(
        source
    )

    while True:
        # ----------------------------------------------------------
        # OBSERVE
        # ----------------------------------------------------------

        # Re-read state from disk.
        #
        # The persistent history loaded above remains the working
        # history for this process. The read here keeps the explicit
        # OBSERVE step in the research loop.
        _ = read_state()

        # ----------------------------------------------------------
        # ADAPT
        # ----------------------------------------------------------

        next_candidate = (
            choose_next_candidate(
                source
            )
        )

        if next_candidate is not None:
            (
                baseline,
                incremental,
                target,
            ) = next_candidate

            write_state(
                build_state(
                    "ADAPT",
                    history,
                    next_candidate,
                )
            )

            proposal = build_adaptive_spec(
                source,
                baseline,
                incremental,
                target,
            )

            (
                spec_path,
                parsed_spec,
            ) = write_and_read_spec(
                proposal
            )

            # Never cross the locked research boundary.
            metadata = load_yaml(
                spec_path
            ).get(
                "metadata",
                {},
            )

            if (
                metadata.get(
                    "locked"
                ) is True
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

            if (
                parsed_spec.id
                == LOCKED_CONFIRMATION_ID
            ):
                raise ValueError(
                    "Adaptive engine attempted to execute "
                    "the locked confirmation spec."
                )

            # ------------------------------------------------------
            # EXPERIMENT
            # ------------------------------------------------------

            result_path = (
                run_research_spec(
                    shared_session,
                    parsed_spec,
                )
            )

            # ------------------------------------------------------
            # ANALYZE
            # ------------------------------------------------------

            summary = summarize_result(
                result_path
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

            write_state(
                build_state(
                    "OBSERVE",
                    history,
                    choose_next_candidate(
                        source
                    ),
                )
            )

            continue

        # ----------------------------------------------------------
        # EXTEND
        # ----------------------------------------------------------
        #
        # There are no more candidates in the predeclared parameter
        # space.
        #
        # This is not treated as a failed proposal.
        #
        # Previously materialized extension families are detected from
        # the restored persistent history and are not materialized
        # again.
        # ----------------------------------------------------------

        summaries = (
            all_adaptive_summaries()
        )

        if not summaries:
            raise RuntimeError(
                "Parameter space is exhausted without "
                "persisted adaptive results."
            )

        completed_extensions = {
            item.get(
                "extension_family",
                item.get(
                    "family"
                ),
            )
            for item in history
            if (
                item.get(
                    "phase"
                ) == "EXTEND"
            )
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
            write_state(
                build_state(
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

        extension = extend(
            history,
            family,
        )

        history.append(
            extension
        )

        write_state(
            build_state(
                "EXTEND",
                history,
                None,
                family,
            )
        )

        # ----------------------------------------------------------
        # EXTENDED EXPERIMENT
        # ----------------------------------------------------------

        if family == "target_profile":
            executed = (
                execute_target_profile_extension(
                    source,
                    shared_session,
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

        else:
            raise ValueError(
                f"Extension family has no executor: {family}"
            )

        write_state(
            build_state(
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
