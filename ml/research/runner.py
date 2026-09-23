from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from ml.research.engine import run_spec
from ml.research.reporting import (
    write_json,
)
from ml.research.session import build_session
from ml.research.spec import load_spec


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SPEC_DIR = (
    ROOT
    / "ml"
    / "research"
    / "specs"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "ml"
    / "research"
    / "spec_runs"
)


def _find_specs(
    spec_dir: Path,
) -> list[Path]:
    return sorted(
        spec_dir.glob("*.yaml")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Blankdiss declarative research runner."
        )
    )

    parser.add_argument(
        "specs",
        nargs="*",
        help=(
            "Spec-filer. Om inga anges körs "
            "alla YAML-filer i specs/."
        ),
    )

    args = parser.parse_args()

    if args.specs:
        spec_paths = [
            Path(path)
            for path in args.specs
        ]
    else:
        spec_paths = _find_specs(
            DEFAULT_SPEC_DIR
        )

    if not spec_paths:
        raise SystemExit(
            "Hittade inga research specs."
        )

    specs = [
        load_spec(path)
        for path in spec_paths
    ]

    print(
        f"Research specs: {len(specs):,}",
        flush=True,
    )

    session = build_session(
        specs
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

    manifest = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "feature_rows": int(
            len(session.frame)
        ),
        "specs": [],
    }

    for spec in specs:
        print(
            f"Running: {spec.id}",
            flush=True,
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

        manifest["specs"].append(
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
                    result["results"]
                ),
            }
        )

        print(
            f"Completed: {spec.id} "
            f"({len(result['results']):,} cells)",
            flush=True,
        )

    write_json(
        run_dir / "manifest.json",
        manifest,
    )

    print()
    print(
        f"Research complete: {run_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
