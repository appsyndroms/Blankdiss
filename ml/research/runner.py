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
def _find_specs(
    spec_dir: Path,
) -> list[Path]:
    return sorted(
        spec_dir.glob("*.yaml")
    )
def _validate_unique_spec_ids(
    specs: list[ResearchSpec],
    spec_paths: list[Path],
) -> None:
    seen: dict[str, Path] = {}
    for spec, path in zip(
        specs,
        spec_paths,
    ):
        previous = seen.get(
            spec.id
        )
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
        for spec, path in zip(
            specs,
            spec_paths,
        )
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
) -> list[ResearchSpec]:
    if not spec_paths:
        raise ValueError(
            "Hittade inga research specs."
        )
    specs = [
        load_spec(path)
        for path in spec_paths
    ]
    _validate_unique_spec_ids(
        specs,
        spec_paths,
    )
    _validate_mode(
        specs,
        spec_paths,
        mode,
    )
    return specs
def run_research(
    spec_paths: list[str | Path] | None = None,
    *,
    mode: str | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Kör en grupp deklarativa research specs.
    Args:
        spec_paths:
            Spec-filer som ska köras. Om None eller tom lista
            används alla YAML-filer i DEFAULT_SPEC_DIR.
        mode:
            Valfritt mode-filter/verifiering.
            Tillåtna värden är "scan" och "deep".
            När mode anges måste varje vald spec ha samma mode.
        output_dir:
            Basdirectory för körresultat. Om None används
            projektets standarddirectory.
    Returns:
        Path till katalogen för den aktuella research-körningen.
    """
    if mode is not None and mode not in {
        "scan",
        "deep",
    }:
        raise ValueError(
            f"Ogiltigt research mode: {mode}"
        )
    if spec_paths:
        resolved_spec_paths = [
            Path(path)
            for path in spec_paths
        ]
    else:
        resolved_spec_paths = _find_specs(
            DEFAULT_SPEC_DIR
        )
    specs = _load_specs(
        resolved_spec_paths,
        mode=mode,
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
    session = build_session(
        specs
    )
    run_timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )
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
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "mode": mode,
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
        args.specs
    )
if __name__ == "__main__":
    main()
