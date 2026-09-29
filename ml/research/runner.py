from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from ml.research.engine import run_spec
from ml.research.reporting import write_json
from ml.research.session import build_session
from ml.research.spec import ResearchSpec, load_spec

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


def _find_specs(spec_dir: Path) -> list[Path]:
    return sorted(spec_dir.glob("*.yaml"))


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _validate_unique_spec_ids(
    specs: list[ResearchSpec],
    spec_paths: list[Path],
) -> None:
    seen: dict[str, Path] = {}

    for spec, path in zip(specs, spec_paths):
        previous = seen.get(spec.id)
        if previous is not None:
            raise ValueError(
                "Duplicate research spec id "
                f"'{spec.id}'.\n"
                f"First: {previous}\n"
                f"Duplicate: {path}"
            )

        seen[spec.id] = path


def _validate_mode(
    specs: list[ResearchSpec],
    spec_paths: list[Path],
    mode: str | None,
) -> None:
    if mode is None:
        return

    invalid = [
        (spec, path)
        for spec, path in zip(specs, spec_paths)
        if spec.mode != mode
    ]

    if not invalid:
        return

    details = "\n".join(
        (
            f"- {path}: "
            f"spec mode='{spec.mode}', "
            f"requested='{mode}'"
        )
        for spec, path in invalid
    )

    raise ValueError(
        "Research spec har fel mode för "
        f"'{mode}':\n{details}"
    )


def _load_specs(
    spec_paths: list[Path],
    *,
    mode: str | None,
    filter_mode: bool,
) -> tuple[list[ResearchSpec], list[Path]]:
    if not spec_paths:
        raise ValueError("Hittade inga research specs.")

    loaded_specs = [
        load_spec(path)
        for path in spec_paths
    ]

    _validate_unique_spec_ids(
        loaded_specs,
        spec_paths,
    )

    if filter_mode and mode is not None:
        selected = [
            (spec, path)
            for spec, path in zip(
                loaded_specs,
                spec_paths,
            )
            if spec.mode == mode
        ]

        if not selected:
            raise ValueError(
                f"Hittade inga research specs med mode='{mode}'."
            )

        specs = [spec for spec, _ in selected]
        selected_paths = [path for _, path in selected]

        return specs, selected_paths

    _validate_mode(
        loaded_specs,
        spec_paths,
        mode,
    )

    return loaded_specs, spec_paths


def run_research(
    spec_paths: list[str | Path] | None = None,
    *,
    mode: str | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    if mode is not None and mode not in {"scan", "deep"}:
        raise ValueError(
            f"Ogiltigt research mode: {mode}"
        )

    explicit_specs = bool(spec_paths)

    if explicit_specs:
        resolved_spec_paths = [
            Path(path)
            for path in spec_paths
        ]
    else:
        resolved_spec_paths = _find_specs(
            DEFAULT_SPEC_DIR
        )

    specs, selected_spec_paths = _load_specs(
        resolved_spec_paths,
        mode=mode,
        filter_mode=not explicit_specs,
    )

    print(
        f"Research specs: {len(specs):,}",
        flush=True,
    )

    if mode is not None:
        print(
            f"Research mode: {mode}",
            flush=True,
        )

    session = build_session(specs)

    run_timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    base_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else OUTPUT_DIR
    )

    run_dir = (
        base_output_dir
        / run_timestamp
    )

    manifest = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "mode": mode,
        "feature_rows": int(
            len(session.frame)
        ),
        "specs": [],
    }

    for spec, spec_path in zip(
        specs,
        selected_spec_paths,
    ):
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
                "spec": _relative_to_root(
                    spec_path
                ),
                "result": _relative_to_root(
                    result_path
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

    return run_dir


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

    run_research(
        args.specs,
    )


if __name__ == "__main__":
    main()
