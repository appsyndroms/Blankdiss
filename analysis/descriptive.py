"""
Statistisk analys av blankningshändelser.

Ingen köp/sälj-signal skapas här.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EVENT_DIR = (
    ROOT
    / "data"
    / "events"
)

ANALYSIS_DIR = (
    ROOT
    / "data"
    / "analysis"
)


BINS = [
    (
        -float("inf"),
        0.0,
        "Minskning",
    ),
    (
        0.0,
        0.25,
        "0–0,25 pp",
    ),
    (
        0.25,
        0.50,
        "0,25–0,50 pp",
    ),
    (
        0.50,
        1.00,
        "0,50–1,00 pp",
    ),
    (
        1.00,
        float("inf"),
        ">1,00 pp",
    ),
]


def read_events() -> pd.DataFrame:

    files = sorted(
        EVENT_DIR.glob(
            "short_events_*.jsonl"
        )
    )

    if not files:
        return pd.DataFrame()

    records = []

    for path in files:

        with path.open(
            encoding="utf-8"
        ) as handle:

            for line in handle:

                if line.strip():

                    records.append(
                        json.loads(line)
                    )

    return pd.DataFrame(
        records
    )


def bucket(
    value: float,
) -> str:

    for lower, upper, label in BINS:

        if lower <= value < upper:
            return label

    return "Okänd"


def analyze(
    events: pd.DataFrame,
) -> list[dict]:

    if events.empty:
        return []

    events = events.copy()

    events["bucket"] = events[
        "change_pp"
    ].apply(bucket)

    results = []

    for label, group in events.groupby(
        "bucket",
        sort=False,
    ):

        row = {
            "bucket": label,
            "events": len(group),
        }

        for days in (
            1,
            5,
            20,
            60,
        ):

            column = (
                f"return_{days}d"
            )

            if column not in group:

                row[
                    f"median_{days}d"
                ] = None

                row[
                    f"mean_{days}d"
                ] = None

                row[
                    f"positive_{days}d"
                ] = None

                continue

            values = group[
                column
            ].dropna()

            if values.empty:

                row[
                    f"median_{days}d"
                ] = None

                row[
                    f"mean_{days}d"
                ] = None

                row[
                    f"positive_{days}d"
                ] = None

                continue

            row[
                f"median_{days}d"
            ] = float(
                values.median()
            )

            row[
                f"mean_{days}d"
            ] = float(
                values.mean()
            )

            row[
                f"positive_{days}d"
            ] = float(
                (values > 0).mean()
            )

        results.append(
            row
        )

    return results


def write_analysis(
    results: list[dict],
) -> Path:

    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        ANALYSIS_DIR
        / (
            "analysis_"
            f"{date.today().isoformat()}.json"
        )
    )

    path.write_text(
        json.dumps(
            {
                "generated_at": (
                    date.today().isoformat()
                ),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def main() -> None:

    events = read_events()

    results = analyze(
        events
    )

    path = write_analysis(
        results
    )

    print(
        f"Analys: {len(results)} "
        f"grupper → {path}"
    )


if __name__ == "__main__":
    main()
