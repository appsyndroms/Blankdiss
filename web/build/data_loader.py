"""
Inläsning av data som används av Blankdiss webbplats.
"""
from __future__ import annotations

import json

from analysis.feature_utils import security_key

from .config import (
    ANALYSIS_DIR,
    EVALUATION_DIR,
    EVENT_DIR,
    FEATURE_DIR,
    ML_DIR,
    ROOT,
)


RETURN_FIELDS = (
    "return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
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


def _event_security_key(
    event: dict,
) -> str:
    """
    Returnerar den kanoniska identiteten för ett web-event.

    security_key är samma identitet som används av
    feature-pipelinen:

        ISIN:<normaliserat ISIN>

    eller, om ISIN saknas:

        ISSUER:<normaliserad issuer>

    Äldre event-filer behöver inte innehålla security_key;
    den byggs då från eventets befintliga identitetsfält.
    """
    existing_key = event.get(
        "security_key"
    )

    if existing_key:
        return str(
            existing_key
        ).strip()

    return security_key(
        event.get("isin"),
        event.get("issuer"),
    )


def _feature_security_key(
    feature: dict,
) -> str:
    """
    Returnerar den kanoniska identiteten för en feature-rad.

    Canonical feature-data ska innehålla security_key.
    Fallback finns för äldre feature-filer.
    """
    existing_key = feature.get(
        "security_key"
    )

    if existing_key:
        return str(
            existing_key
        ).strip()

    return security_key(
        feature.get("isin"),
        feature.get("issuer"),
    )


def _event_date(
    event: dict,
) -> str:
    return str(
        event.get(
            "event_date"
        )
        or ""
    )[:10]


def _feature_date(
    feature: dict,
) -> str:
    """
    snapshot_date i canonical feature-datasetet är
    den faktiska FI position_date.

    Raw FI:
        position_date

    Canonical feature:
        snapshot_date

    Web event:
        event_date
    """
    return str(
        feature.get(
            "snapshot_date"
        )
        or ""
    )[:10]


def _read_feature_returns(
    events: list[dict],
) -> dict[
    tuple[str, str],
    dict,
]:
    """
    Läser det kanoniska feature-datasetet och bygger
    ett index för de event som faktiskt finns på webben.

    Indexnyckeln är:

        (feature_date, security_key)

    security_key är den gemensamma identiteten mellan
    FI-event och canonical feature-data.

    Forward returns kommer direkt från det kanoniska
    feature-datasetet. Webblagret räknar alltså inte
    själv ut framtida avkastning.
    """
    required_events: set[
        tuple[str, str]
    ] = set()

    for event in events:
        event_date = _event_date(
            event
        )

        if not event_date:
            continue

        event_key = _event_security_key(
            event
        )

        if not event_key:
            continue

        required_events.add(
            (
                event_date,
                event_key,
            )
        )

    if not required_events:
        return {}

    feature_files = sorted(
        FEATURE_DIR.glob(
            "features_*.jsonl"
        )
    )

    if not feature_files:
        return {}

    index: dict[
        tuple[str, str],
        dict,
    ] = {}

    for path in feature_files:
        try:
            handle = path.open(
                encoding="utf-8"
            )
        except OSError:
            continue

        with handle:
            for line in handle:
                line = line.strip()

                if not line:
                    continue

                try:
                    feature = json.loads(
                        line
                    )
                except json.JSONDecodeError:
                    continue

                feature_date = _feature_date(
                    feature
                )

                if not feature_date:
                    continue

                feature_key = _feature_security_key(
                    feature
                )

                if not feature_key:
                    continue

                key = (
                    feature_date,
                    feature_key,
                )

                if key not in required_events:
                    continue

                index[key] = {
                    "forward_return_1d": (
                        feature.get(
                            "forward_return_1d"
                        )
                    ),
                    "forward_return_5d": (
                        feature.get(
                            "forward_return_5d"
                        )
                    ),
                    "forward_return_20d": (
                        feature.get(
                            "forward_return_20d"
                        )
                    ),
                    "forward_return_60d": (
                        feature.get(
                            "forward_return_60d"
                        )
                    ),
                }

    return index


def _enrich_events_with_feature_returns(
    events: list[dict],
) -> list[dict]:
    """
    Berikar events med forward returns från det
    kanoniska feature-datasetet.

    Ett event utan feature-match får None för samtliga
    returnfält.

    Event-filerna är alltså inte en alternativ källa
    för returns.
    """
    feature_index = _read_feature_returns(
        events
    )

    enriched: list[dict] = []

    for event in events:
        result = dict(
            event
        )

        # Canonical feature-datasetet är den enda
        # källan till forward returns i webbskiktet.
        for field in RETURN_FIELDS:
            result[field] = None

        event_date = _event_date(
            event
        )

        event_key = _event_security_key(
            event
        )

        if event_date and event_key:
            matched_feature = feature_index.get(
                (
                    event_date,
                    event_key,
                )
            )
        else:
            matched_feature = None

        if matched_feature is not None:
            for field in RETURN_FIELDS:
                result[field] = (
                    matched_feature.get(
                        f"forward_{field}"
                    )
                )

        enriched.append(
            result
        )

    return enriched


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
    # dagliga snapshot-filer.
    #
    # Använd samma canonical security_key som
    # feature-datasetet även vid deduplicering.
    unique: dict[
        tuple[str, str],
        dict,
    ] = {}

    for event in records:
        event_date = _event_date(
            event
        )

        event_key = _event_security_key(
            event
        )

        unique[
            (
                event_date,
                event_key,
            )
        ] = event

    events = list(
        unique.values()
    )

    return _enrich_events_with_feature_returns(
        events
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
