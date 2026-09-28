from __future__ import annotations

import argparse
import sys

from ml.research.candidates.freeze import (
    freeze_candidate,
)
from ml.research.evaluation.runner import (
    run_evaluation,
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

    return parser


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
