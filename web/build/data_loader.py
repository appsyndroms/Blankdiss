"""
Inläsning av data som används av Blankdiss webbplats.
"""
from __future__ import annotations
import json
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
def _event_identity_keys(
    event: dict,
) -> list[tuple[str, str]]:
    """
    Returnerar möjliga identiteter för ett event.
    Identiteten används tillsammans med eventets datum
    för att slå upp motsvarande rad i det kanoniska
    feature-datasetet.
    Ordningen går från stabilast till svagast:
        ISIN
        LEI
        Yahoo-symbol
        issuer
    """
    keys: list[tuple[str, str]] = []
    for field in (
        "isin",
        "lei",
        "yahoo_symbol",
        "issuer",
    ):
        value = event.get(
            field
        )
        if value is None:
            continue
        value = str(
            value
        ).strip()
        if not value:
            continue
        key = (
            field,
            value,
        )
        if key not in keys:
            keys.append(key)
    return keys
def _feature_identity_keys(
    feature: dict,
) -> list[tuple[str, str]]:
    """
    Returnerar möjliga identiteter för en feature-rad.
    Feature-datasetet kan ha olika identifierare beroende
    på vilken källa som lyckades matcha FI-raden.
    """
    keys: list[tuple[str, str]] = []
    for field in (
        "isin",
        "lei",
        "yahoo_symbol",
        "issuer",
    ):
        value = feature.get(
            field
        )
        if value is None:
            continue
        value = str(
            value
        ).strip()
        if not value:
            continue
        key = (
            field,
            value,
        )
        if key not in keys:
            keys.append(key)
    return keys
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
) -> dict[tuple[str, str, str], dict]:
    """
    Läser det kanoniska feature-datasetet och bygger
    ett index för de event som faktiskt finns på webben.
    Indexnyckeln är:
        (feature_date, identity_field, identity_value)
    Feature-datumet är canonical snapshot_date, vilket
    motsvarar FI position_date.
    Forward returns kommer direkt från det kanoniska
    feature-datasetet. Webblagret räknar alltså inte
    själv ut framtida avkastning.
    """
    required_events: set[
        tuple[str, str, str]
    ] = set()
    for event in events:
        event_date = _event_date(
            event
        )
        if not event_date:
            continue
        for field, value in _event_identity_keys(
            event
        ):
            required_events.add(
                (
                    event_date,
                    field,
                    value,
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
        tuple[str, str, str],
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
                identities = (
                    _feature_identity_keys(
                        feature
                    )
                )
                for field, value in identities:
                    key = (
                        feature_date,
                        field,
                        value,
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
    returnfält. Det är viktigt eftersom event-filerna inte
    längre ska vara en alternativ källa för returns.
    """
    feature_index = _read_feature_returns(
        events
    )
    enriched: list[dict] = []
    for event in events:
        result = dict(
            event
        )
        # Canonical feature-datasetet är den enda källan
        # till forward returns i webbskiktet.
        for field in RETURN_FIELDS:
            result[field] = None
        event_date = _event_date(
            event
        )
        matched_feature = None
        if event_date:
            for field, value in _event_identity_keys(
                event
            ):
                key = (
                    event_date,
                    field,
                    value,
                )
                matched_feature = feature_index.get(
                    key
                )
                if matched_feature is not None:
                    break
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
    # dagliga snapshot-filer. Behåll endast
    # den senaste observationen för varje
    # kombination av datum och aktie.
    unique: dict[
        tuple[str, str],
        dict,
    ] = {}
    for event in records:
        event_date = _event_date(
            event
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
