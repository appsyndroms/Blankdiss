from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from shared.instrument_identity import (
    InstrumentIdentity,
    normalize_identifier,
    normalize_isin,
    normalize_lei,
    normalize_text,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FI_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fi"
    / "aggregate"
    / "reconstructed.jsonl"
)
SAMPLE_LIMIT = 50
def _read_fi_records() -> list[dict[str, Any]]:
    if not FI_PATH.exists():
        raise FileNotFoundError(
            f"FI-filen finns inte: {FI_PATH}"
        )
    records: list[dict[str, Any]] = []
    with FI_PATH.open(
        encoding="utf-8"
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Ogiltig JSON på rad {line_number}: "
                    f"{FI_PATH}"
                ) from exc
            if isinstance(record, dict):
                records.append(record)
    return records
def _value(
    record: dict[str, Any],
    *keys: str,
) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None
def _identity_key(
    record: dict[str, Any],
) -> tuple[str, str, str]:
    return (
        normalize_isin(
            _value(record, "isin")
        ),
        normalize_lei(
            _value(record, "lei", "LEI")
        ),
        normalize_text(
            _value(record, "issuer")
        ),
    )
def _has_identity(
    key: tuple[str, str, str],
) -> bool:
    isin, lei, issuer = key
    return bool(
        isin
        or lei
        or issuer
    )
def _describe_key(
    key: tuple[str, str, str],
) -> str:
    isin, lei, issuer = key
    return (
        f"isin={isin or '-'} "
        f"lei={lei or '-'} "
        f"issuer={issuer or '-'}"
    )
def _entity_resolution(
    identity: InstrumentIdentity,
    key: tuple[str, str, str],
) -> tuple[
    str,
    str | None,
    list[str],
]:
    """
    Klassificerar en identity-observation utan att skriva.
    Returnerar:
        status
        entity_id
        matching_entity_ids
    Status:
        existing_entity
        safe_new_entity
        ambiguous
        insufficient_identity
    """
    isin, lei, issuer = key
    if not _has_identity(key):
        return (
            "insufficient_identity",
            None,
            [],
        )
    matches = identity._matching_records(
        isin=isin or None,
        lei=lei or None,
        issuer=issuer or None,
    )
    if not matches:
        return (
            "safe_new_entity",
            None,
            [],
        )
    priority = {
        "isin": 0,
        "lei": 1,
        "yahoo_symbol": 2,
        "ticker": 3,
        "issuer": 4,
    }
    matches.sort(
        key=lambda match: priority[
            match.resolution
        ]
    )
    best_priority = priority[
        matches[0].resolution
    ]
    best = [
        match
        for match in matches
        if priority[
            match.resolution
        ]
        == best_priority
    ]
    entity_ids = sorted(
        {
            match.entity_id
            for match in best
        }
    )
    if len(entity_ids) == 1:
        return (
            "existing_entity",
            entity_ids[0],
            entity_ids,
        )
    return (
        "ambiguous",
        None,
        entity_ids,
    )
def _source_conflicts(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, set[str]]]:
    """
    Letar efter konflikter i FI-källan.
    Vi letar särskilt efter:
        samma ISIN -> flera issuer
        samma LEI  -> flera issuer
        samma issuer -> flera LEI
    Detta är viktigt innan discovery får börja skriva till registret.
    """
    isin_to_issuer: dict[str, set[str]] = defaultdict(set)
    lei_to_issuer: dict[str, set[str]] = defaultdict(set)
    issuer_to_lei: dict[str, set[str]] = defaultdict(set)
    for record in records:
        isin = normalize_isin(
            _value(record, "isin")
        )
        lei = normalize_lei(
            _value(record, "lei", "LEI")
        )
        issuer = normalize_text(
            _value(record, "issuer")
        )
        if isin and issuer:
            isin_to_issuer[isin].add(
                issuer
            )
        if lei and issuer:
            lei_to_issuer[lei].add(
                issuer
            )
        if issuer and lei:
            issuer_to_lei[issuer].add(
                lei
            )
    return {
        "same_isin_multiple_issuers": {
            key: values
            for key, values in isin_to_issuer.items()
            if len(values) > 1
        },
        "same_lei_multiple_issuers": {
            key: values
            for key, values in lei_to_issuer.items()
            if len(values) > 1
        },
        "same_issuer_multiple_lei": {
            key: values
            for key, values in issuer_to_lei.items()
            if len(values) > 1
        },
    }
def _print_conflicts(
    conflicts: dict[str, dict[str, set[str]]],
) -> None:
    print()
    print("=" * 70)
    print("KONFLIKTER I FI-KÄLLAN")
    print("=" * 70)
    total = 0
    for name, values in conflicts.items():
        print()
        print(name)
        print("-" * 70)
        if not values:
            print("Inga konflikter.")
            continue
        total += len(values)
        for key, related in sorted(values.items()):
            print(
                f"{key}: "
                + ", ".join(sorted(related))
            )
    print()
    print(
        f"Totalt antal konfliktgrupper: {total}"
    )
def _print_samples(
    title: str,
    values: list[tuple[Any, ...]],
) -> None:
    print()
    print(title)
    print("-" * 70)
    if not values:
        print("Inga.")
        return
    for value in values[:SAMPLE_LIMIT]:
        print(" | ".join(str(part) for part in value))
    if len(values) > SAMPLE_LIMIT:
        print()
        print(
            f"... {len(values) - SAMPLE_LIMIT:,} "
            "ytterligare poster."
        )
def main() -> None:
    print("=" * 70)
    print("IDENTITY / DISCOVERY FELSÖKNING")
    print("=" * 70)
    print()
    print(f"FI: {FI_PATH}")
    identity = InstrumentIdentity()
    print()
    print("BEFINTLIGT IDENTITY-REGISTER")
    print("-" * 70)
    print(
        f"Observationer : {len(identity.records):,}"
    )
    print(
        f"Entity-id:n   : "
        f"{len(identity._entity_ids()):,}"
    )
    print(
        f"Instrument map: "
        f"{len(identity.instrument_map):,}"
    )
    records = _read_fi_records()
    print()
    print("FI-KÄLLA")
    print("-" * 70)
    print(
        f"Rader: {len(records):,}"
    )
    unique_keys = {
        _identity_key(record)
        for record in records
    }
    print(
        f"Unika identity-observationer: "
        f"{len(unique_keys):,}"
    )
    # ---------------------------------------------------------
    # 1. Klassificera varje unik identity-observation
    # ---------------------------------------------------------
    counts = defaultdict(int)
    existing: list[
        tuple[str, str, str, str]
    ] = []
    new_candidates: list[
        tuple[str, str, str]
    ] = []
    ambiguous: list[
        tuple[str, str, str, str]
    ] = []
    insufficient: list[
        tuple[str, str, str]
    ] = []
    for key in sorted(unique_keys):
        status, entity_id, matching_entities = (
            _entity_resolution(
                identity,
                key,
            )
        )
        counts[status] += 1
        isin, lei, issuer = key
        if status == "existing_entity":
            existing.append(
                (
                    entity_id or "-",
                    isin or "-",
                    lei or "-",
                    issuer or "-",
                )
            )
        elif status == "safe_new_entity":
            new_candidates.append(
                (
                    isin or "-",
                    lei or "-",
                    issuer or "-",
                )
            )
        elif status == "ambiguous":
            ambiguous.append(
                (
                    ",".join(matching_entities),
                    isin or "-",
                    lei or "-",
                    issuer or "-",
                )
            )
        else:
            insufficient.append(
                (
                    isin or "-",
                    lei or "-",
                    issuer or "-",
                )
            )
    # ---------------------------------------------------------
    # 2. Sammanfattning
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("DISCOVERY DRY-RUN")
    print("=" * 70)
    print()
    print(
        f"existing_entity     : "
        f"{counts['existing_entity']:,}"
    )
    print(
        f"safe_new_entity     : "
        f"{counts['safe_new_entity']:,}"
    )
    print(
        f"ambiguous           : "
        f"{counts['ambiguous']:,}"
    )
    print(
        f"insufficient_identity: "
        f"{counts['insufficient_identity']:,}"
    )
    print()
    print(
        "VIKTIGT: Detta är endast analys. "
        "Ingen discovery körs och inget skrivs."
    )
    # ---------------------------------------------------------
    # 3. Existerande entity-matchningar
    # ---------------------------------------------------------
    _print_samples(
        "BEFINTLIGA ENTITY-MATCHNINGAR",
        existing,
    )
    # ---------------------------------------------------------
    # 4. Kandidater för nya entitys
    # ---------------------------------------------------------
    _print_samples(
        "KANDIDATER FÖR NYA ENTITYS",
        new_candidates,
    )
    # ---------------------------------------------------------
    # 5. Ambiguous
    # ---------------------------------------------------------
    _print_samples(
        "AMBIGUA MATCHNINGAR",
        ambiguous,
    )
    # ---------------------------------------------------------
    # 6. Otillräcklig identitet
    # ---------------------------------------------------------
    _print_samples(
        "OTILLRÄCKLIG IDENTITET",
        insufficient,
    )
    # ---------------------------------------------------------
    # 7. Konflikter i själva FI-källan
    # ---------------------------------------------------------
    conflicts = _source_conflicts(
        records
    )
    _print_conflicts(
        conflicts
    )
    # ---------------------------------------------------------
    # 8. Slutsats
    # ---------------------------------------------------------
    print()
    print("=" * 70)
    print("SLUTSATS")
    print("=" * 70)
    if counts["ambiguous"]:
        print()
        print(
            "Det finns ambiguösa identiteter som måste "
            "hanteras innan discovery kan köras säkert."
        )
    if counts["insufficient_identity"]:
        print()
        print(
            "Det finns FI-poster utan tillräcklig identitet "
            "och de ska inte få en entity automatiskt."
        )
    if (
        not counts["ambiguous"]
        and not counts["insufficient_identity"]
    ):
        print()
        print(
            "Inga identity-matchningar blockerar en "
            "försiktig discovery-bootstrap."
        )
    print()
    print(
        "Nästa steg efter denna felsökning är att använda "
        "resultatet för den riktiga bootstrapen."
    )
    print()
    print(
        "instrument_aliases.jsonl har INTE ändrats."
    )
if __name__ == "__main__":
    main()
