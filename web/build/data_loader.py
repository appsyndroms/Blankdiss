"""
Inläsning av data som används av Blankdiss webbplats.
"""

from __future__ import annotations

import json

from .config import (
    ANALYSIS_DIR,
    EVALUATION_DIR,
    EVENT_DIR,
    ML_DIR,
    ROOT,
)


def read_latest_analysis() -> dict:
    files = sorted(
        ANALYSIS_DIR.glob(
            "analysis_*.json"
        )
    )

    if not files:
        return {
            "generated_at": None,
            "results": [],
        }

    return json.loads(
        files[-1].read_text(
            encoding="utf-8"
        )
    )


def read_events() -> list[dict]:
    records: list[dict] = []

    files = sorted(
        EVENT_DIR.glob(
            "short_events_*.jsonl"
        )
    )

    for path in files:
        with path.open(
            encoding="utf-8"
        ) as handle:
            for line in handle:
                line = line.strip()

                if line:
                    records.append(
                        json.loads(line)
                    )

    # Samma händelse kan förekomma i flera
    # dagliga snapshot-filer. Behåll endast
    # den senaste observationen för varje
    # kombination av datum och aktie.
    unique: dict[
        tuple[str, str],
        dict
    ] = {}

    for event in records:
        event_date = str(
            event.get("event_date")
            or ""
        )

        identity = (
            event.get("isin")
            or event.get("lei")
            or event.get("yahoo_symbol")
            or event.get("issuer")
            or ""
        )

        unique[
            (
                event_date,
                str(identity),
            )
        ] = event

    return list(
        unique.values()
    )


def read_economic_results() -> dict:
    path = (
        ML_DIR
        / "economic_results.json"
    )

    if not path.exists():
        return {
            "created_at": None,
            "experiments": [],
            "experiment_count": 0,
        }

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def read_evaluation_history() -> list[dict]:
    """
    Läser alla sparade prospektiva evaluation-körningar.

    Varje körning ligger i en egen katalog:
        data/processed/ml/research/evaluation/<run-id>/

    och innehåller evaluation.json.
    """

    if not EVALUATION_DIR.exists():
        return []

    evaluations: list[dict] = []

    for path in sorted(
        EVALUATION_DIR.glob(
            "*/evaluation.json"
        )
    ):
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
            continue

        payload["_source_path"] = str(
            path.relative_to(ROOT)
        )

        evaluations.append(
            payload
        )

    evaluations.sort(
        key=lambda item: (
            item.get(
                "created_at_utc",
                "",
            )
        ),
        reverse=True,
    )

    return evaluations
