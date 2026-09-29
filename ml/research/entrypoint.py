from __future__ import annotations
import argparse
import sys
from ml.research.candidates.freeze import (
    freeze_candidate,
)
from ml.research.evaluation.runner import (
    run_evaluation,
)
from ml.research.runner import (
    run_research,
)
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blankdiss-research",
        description=(
            "Blankdiss research lifecycle entrypoint."
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    freeze_parser = subparsers.add_parser(
        "freeze",
        help="Freeze a candidate.",
    )
    freeze_parser.add_argument(
        "candidate",
        help="Path till draft candidate YAML.",
    )
    freeze_parser.add_argument(
        "--output",
        help="Path till frozen candidate YAML.",
    )
    freeze_parser.add_argument(
        "--freeze-at",
        help="Override freeze_at.",
    )
    scan_parser = subparsers.add_parser(
        "scan",
        help="Run declarative research in scan mode.",
    )
    scan_parser.add_argument(
        "specs",
        nargs="*",
        help=(
            "Research-specs. Om inga anges körs "
            "alla specs med mode=scan."
        ),
    )
    scan_parser.add_argument(
        "--output-dir",
        help="Override research output directory.",
    )
    deep_parser = subparsers.add_parser(
        "deep",
        help="Run declarative research in deep mode.",
    )
    deep_parser.add_argument(
        "specs",
        nargs="*",
        help=(
            "Research-specs. Om inga anges körs "
            "alla specs med mode=deep."
        ),
    )
    deep_parser.add_argument(
        "--output-dir",
        help="Override research output directory.",
    )
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run prospective evaluation.",
    )
    evaluate_parser.add_argument(
        "candidate",
        help="Path till frozen candidate YAML.",
    )
    evaluate_parser.add_argument(
        "evaluation",
        help="Path till evaluation YAML.",
    )
    evaluate_parser.add_argument(
        "--output-dir",
        help="Override evaluation output directory.",
    )
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help=(
            "Run the complete Blankdiss research "
            "pipeline."
        ),
    )
    pipeline_parser.add_argument(
        "--scan",
        action="store_true",
        help="Run discovery/research scan first.",
    )
    pipeline_parser.add_argument(
        "--deep",
        action="store_true",
        help="Run deep research before evaluation.",
    )
    pipeline_parser.add_argument(
        "--scan-spec",
        action="append",
        dest="scan_specs",
        default=[],
        help=(
            "Research spec for scan. May be "
            "specified multiple times."
        ),
    )
    pipeline_parser.add_argument(
        "--deep-spec",
        action="append",
        dest="deep_specs",
        default=[],
        help=(
            "Research spec for deep analysis. May be "
            "specified multiple times."
        ),
    )
    pipeline_parser.add_argument(
        "--candidate",
        help="Frozen candidate YAML.",
    )
    pipeline_parser.add_argument(
        "--evaluation",
        help="Evaluation YAML.",
    )
    pipeline_parser.add_argument(
        "--output-dir",
        help="Override evaluation output directory.",
    )
    return parser
def _run_research_command(
    specs: list[str],
    *,
    mode: str,
    output_dir: str | None,
):
    return run_research(
        specs,
        mode=mode,
        output_dir=output_dir,
    )
def _run_pipeline(
    args: argparse.Namespace,
) -> int:
    if args.scan:
        _run_research_command(
            args.scan_specs,
            mode="scan",
            output_dir=None,
        )
    if args.deep:
        _run_research_command(
            args.deep_specs,
            mode="deep",
            output_dir=None,
        )
    if args.candidate is None:
        raise ValueError(
            "pipeline kräver --candidate."
        )
    if args.evaluation is None:
        raise ValueError(
            "pipeline kräver --evaluation."
        )
    output = run_evaluation(
        args.candidate,
        args.evaluation,
        output_dir=args.output_dir,
    )
    print(
        f"Pipeline evaluation complete: {output}",
        flush=True,
    )
    return 0
def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            output = freeze_candidate(
                args.candidate,
                args.output,
                freeze_at=args.freeze_at,
            )
            print(
                f"Candidate frozen: {output}",
                flush=True,
            )
            return 0
        if args.command == "scan":
            output = _run_research_command(
                args.specs,
                mode="scan",
                output_dir=args.output_dir,
            )
            print(
                f"Research scan complete: {output}",
                flush=True,
            )
            return 0
        if args.command == "deep":
            output = _run_research_command(
                args.specs,
                mode="deep",
                output_dir=args.output_dir,
            )
            print(
                f"Research deep complete: {output}",
                flush=True,
            )
            return 0
        if args.command == "evaluate":
            output = run_evaluation(
                args.candidate,
                args.evaluation,
                output_dir=args.output_dir,
            )
            print(
                f"Evaluation complete: {output}",
                flush=True,
            )
            return 0
        if args.command == "pipeline":
            return _run_pipeline(
                args
            )
        parser.error(
            f"Unknown command: {args.command}"
        )
    except Exception as exc:
        print(
            f"Blankdiss research failed: {exc}",
            file=sys.stderr,
        )
        return 1
    return 1
if __name__ == "__main__":
    raise SystemExit(
        main()
    )
