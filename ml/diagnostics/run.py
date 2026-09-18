from __future__ import annotations

import argparse

from .framework.runner import (
    list_experiments,
    run_many,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kör diagnostics-experiment."
    )

    parser.add_argument(
        "experiments",
        nargs="*",
        help=(
            "Experimentnamn, 'all' eller 'list'."
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Kör utan standardiserad rapportering.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    experiments = list_experiments()

    if not args.experiments or args.experiments == ["list"]:
        print("Tillgängliga experiments:")
        print()

        for experiment in experiments:
            suffix = " [legacy]" if experiment.legacy else ""
            print(
                f"  {experiment.name}{suffix}"
                f" - {experiment.description}"
            )

        return

    if "all" in args.experiments:
        names = [
            experiment.name
            for experiment in experiments
        ]
    else:
        names = args.experiments

    run_many(
        names,
        print_output=not args.quiet,
    )


if __name__ == "__main__":
    main()
