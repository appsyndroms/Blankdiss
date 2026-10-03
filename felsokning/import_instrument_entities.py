from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from shared.entity_identity import (
    ENTITY_REGISTRY_PATH,
    EntityRegistry,
)
from shared.instrument_identity import (
    normalize_isin,
    normalize_lei,
    normalize_text,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DISCOVERY_PATH = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "instrument_entity_discovery.json"
)

ALIASES_PATH = (
    PROJECT_ROOT
    / "shared"
    / "instrument_aliases.jsonl"
)

SOURCE = "GLEIF_ANNA"
STATUS = "candidate"
RELATION = "observed"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Importerar GLEIF/ANNA entity-kandidater "
            "från discovery-rapporten."
        )
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Skriver nya entities och aliasposter. "
            "Utan --apply körs endast dry-run."
        ),
    )

    return parser.parse_args()


def _load_discovery() -> dict[str, Any]:
    if not DISCOVERY_PATH.exists():
        raise FileNotFoundError(
            "Discovery-rapport saknas: "
            f"{DISCOVERY_PATH}"
        )

    with DISCOVERY_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "Discovery-rapporten måste vara ett "
            "JSON-objekt."
        )

    return data


def _load_alias_records() -> list[dict[str, Any]]:
    if not ALIASES_PATH.exists():
        return []

    records: list[
        dict[str, Any]
    ] = []

    with ALIASES_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Ogiltig JSON i "
                    f"{ALIASES_PATH} "
                    f"rad {line_number}."
                ) from exc

            if not isinstance(
                record,
                dict,
            ):
                raise ValueError(
                    "Aliaspost på rad "
                    f"{line_number} är inte ett objekt."
                )

            records.append(
                record
            )

    return records


def _entity_ids_from_records(
    records: list[dict[str, Any]],
) -> set[str]:
    return {
        str(record["entity_id"])
        for record in records
        if record.get("entity_id")
    }


def _lei_to_entities(
    records: list[dict[str, Any]],
) -> dict[str, set[str]]:
    mapping: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for record in records:
        lei = normalize_lei(
            record.get("lei")
        )

        entity_id = record.get(
            "entity_id"
        )

        if not lei or not entity_id:
            continue

        mapping[lei].add(
            str(entity_id)
        )

    return dict(mapping)


def _isin_to_entities(
    records: list[dict[str, Any]],
) -> dict[str, set[str]]:
    mapping: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for record in records:
        isin = normalize_isin(
            record.get("isin")
        )

        entity_id = record.get(
            "entity_id"
        )

        if not isin or not entity_id:
            continue

        mapping[isin].add(
            str(entity_id)
        )

    return dict(mapping)


def _existing_alias_keys(
    records: list[dict[str, Any]],
) -> set[
    tuple[str, str, str]
]:
    """
    Importen är idempotent per:

        entity_id + ISIN + source
    """

    keys: set[
        tuple[str, str, str]
    ] = set()

    for record in records:
        entity_id = record.get(
            "entity_id"
        )

        isin = normalize_isin(
            record.get("isin")
        )

        source = normalize_text(
            record.get("source")
        )

        if not entity_id or not isin:
            continue

        keys.add(
            (
                str(entity_id),
                isin,
                source,
            )
        )

    return keys


def _candidate_records(
    discovery: dict[str, Any],
) -> dict[
    str,
    list[dict[str, Any]],
]:
    """
    Grupperar discovery-resultatet på LEI.

    Ett LEI representerar här discoveryns observerade
    entity-kandidat. ISIN används därefter som viktig
    identitetsbrygga mot redan kända entities.
    """

    groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for candidate in discovery.get(
        "candidate_entities",
        [],
    ):
        lei = normalize_lei(
            candidate.get("lei")
        )

        if not lei:
            continue

        for instrument in candidate.get(
            "instruments",
            [],
        ):
            isin = normalize_isin(
                instrument.get("isin")
            )

            if not isin:
                continue

            groups[lei].append(
                {
                    "isin": isin,
                    "issuer": normalize_text(
                        instrument.get("issuer")
                    ),
                    "yahoo_symbol": normalize_text(
                        instrument.get("yahoo_symbol")
                    ),
                    "instrument_lei": normalize_lei(
                        instrument.get("instrument_lei")
                    ),
                }
            )

    return dict(
        sorted(
            groups.items()
        )
    )


def _migrate_entities_from_aliases(
    registry: EntityRegistry,
    aliases: list[dict[str, Any]],
) -> None:
    """
    Säkerställer att befintliga entity-id:n i aliasregistret
    också finns i det centrala entity-registret.

    Detta behövs för den första migrationen från den tidigare
    modellen där entity-id:t endast fanns i aliasregistret.
    """

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in aliases:
        entity_id = record.get(
            "entity_id"
        )

        if entity_id:
            grouped[
                str(entity_id)
            ].append(
                record
            )

    changed = False

    for entity_id, records in grouped.items():
        if registry.contains(
            entity_id
        ):
            continue

        first = records[0]

        observed_dates = sorted(
            record.get(
                "observed_date"
            )
            for record in records
            if record.get(
                "observed_date"
            )
        )

        created_date = (
            observed_dates[0]
            if observed_dates
            else "unknown"
        )

        registry.add(
            entity_id=entity_id,
            legal_name=first.get(
                "issuer"
            ),
            observed_date=created_date,
            source=first.get(
                "source"
            ) or "migration",
            status="active",
        )

        changed = True

    if changed:
        registry.save()


def _build_import_plan(
    discovery: dict[str, Any],
    aliases: list[dict[str, Any]],
    registry: EntityRegistry,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    lei_entities = _lei_to_entities(
        aliases
    )

    isin_entities = _isin_to_entities(
        aliases
    )

    existing_keys = _existing_alias_keys(
        aliases
    )

    groups = _candidate_records(
        discovery
    )

    observed_date = (
        discovery.get(
            "source",
            {},
        ).get(
            "mapping_date"
        )
    )

    if not observed_date:
        raise ValueError(
            "Discovery-rapporten saknar "
            "source.mapping_date."
        )

    planned_entities: list[
        dict[str, Any]
    ] = []

    import_records: list[
        dict[str, Any]
    ] = []

    planned_entity_ids: set[str] = set(
        registry.ids()
    )

    skipped_existing = 0
    conflicts = 0

    for lei, instruments in groups.items():
        known_by_lei = lei_entities.get(
            lei,
            set(),
        )

        # --------------------------------------------------------------
        # 1. Samma LEI finns redan
        # --------------------------------------------------------------

        if len(known_by_lei) > 1:
            conflicts += 1

            planned_entities.append(
                {
                    "lei": lei,
                    "status": "conflict",
                    "entity_ids": sorted(
                        known_by_lei
                    ),
                    "instrument_count": len(
                        instruments
                    ),
                }
            )

            continue

        if len(known_by_lei) == 1:
            entity_id = next(
                iter(
                    known_by_lei
                )
            )

            entity_status = (
                "existing_entity"
            )

        else:
            # ----------------------------------------------------------
            # 2. Försök hitta identitetsbrygga via ISIN
            # ----------------------------------------------------------

            isin_entity_ids: set[str] = set()

            for instrument in instruments:
                isin_entity_ids.update(
                    isin_entities.get(
                        instrument["isin"],
                        set(),
                    )
                )

            if len(isin_entity_ids) > 1:
                conflicts += 1

                planned_entities.append(
                    {
                        "lei": lei,
                        "status": "conflict",
                        "entity_ids": sorted(
                            isin_entity_ids
                        ),
                        "instrument_count": len(
                            instruments
                        ),
                    }
                )

                continue

            if len(isin_entity_ids) == 1:
                entity_id = next(
                    iter(
                        isin_entity_ids
                    )
                )

                entity_status = (
                    "existing_entity_via_isin"
                )

            else:
                # ------------------------------------------------------
                # 3. Ny persistent entity
                # ------------------------------------------------------

                entity_id = (
                    registry.next_entity_id()
                )

                while entity_id in planned_entity_ids:
                    numeric = int(
                        entity_id.split(
                            "-",
                            1,
                        )[1]
                    )

                    entity_id = (
                        f"ENT-{numeric + 1:06d}"
                    )

                planned_entity_ids.add(
                    entity_id
                )

                entity_status = (
                    "new_entity_candidate"
                )

        legal_name = next(
            (
                instrument["issuer"]
                for instrument in instruments
                if instrument.get("issuer")
            ),
            None,
        )

        planned_entities.append(
            {
                "lei": lei,
                "entity_id": entity_id,
                "legal_name": legal_name,
                "status": entity_status,
                "instrument_count": len(
                    instruments
                ),
            }
        )

        for instrument in instruments:
            isin = instrument[
                "isin"
            ]

            key = (
                entity_id,
                isin,
                SOURCE,
            )

            if key in existing_keys:
                skipped_existing += 1
                continue

            record = {
                "entity_id": entity_id,
                "isin": isin,
                "issuer": instrument.get(
                    "issuer"
                ),
                "lei": lei,
                "valid_from": None,
                "valid_to": None,
                "relation": RELATION,
                "source": SOURCE,
                "status": STATUS,
                "observed_date": observed_date,
            }

            import_records.append(
                record
            )

            existing_keys.add(
                key
            )

    new_entity_count = sum(
        1
        for entity in planned_entities
        if entity.get(
            "status"
        ) == "new_entity_candidate"
    )

    reused_count = sum(
        1
        for entity in planned_entities
        if entity.get(
            "status"
        ) in {
            "existing_entity",
            "existing_entity_via_isin",
        }
    )

    summary = {
        "discovery_mapping_date": observed_date,
        "candidate_lei_groups": len(
            groups
        ),
        "new_entity_candidates": new_entity_count,
        "existing_entities_reused": reused_count,
        "conflicting_entities": conflicts,
        "new_alias_records": len(
            import_records
        ),
        "already_present": skipped_existing,
    }

    return (
        planned_entities,
        import_records,
        summary,
    )


def _apply_import(
    entities: list[dict[str, Any]],
    aliases: list[dict[str, Any]],
    registry: EntityRegistry,
) -> None:
    """
    Skriver entity-registret och aliasregistret.

    Entity-registret skrivs först eftersom aliasposter refererar
    till entity_id.
    """

    for entity in entities:
        if entity.get(
            "status"
        ) not in {
            "new_entity_candidate",
        }:
            continue

        registry.add(
            entity_id=entity["entity_id"],
            legal_name=entity.get(
                "legal_name"
            ),
            observed_date=entity.get(
                "observed_date"
            ) or entity.get(
                "created_date"
            ) or "",
            source=SOURCE,
            status=STATUS,
        )

    registry.save()

    if not aliases:
        return

    ALIASES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with ALIASES_PATH.open(
        "a",
        encoding="utf-8",
    ) as handle:
        for record in aliases:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                )
            )
            handle.write("\n")


def _print_plan(
    entities: list[dict[str, Any]],
    aliases: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    print()
    print("=" * 70)
    print(
        "GLEIF / ENTITY IMPORT"
    )
    print("=" * 70)

    print()
    print(
        "Discoverydatum       : "
        f"{summary['discovery_mapping_date']}"
    )

    print(
        "LEI-grupper          : "
        f"{summary['candidate_lei_groups']:,}"
    )

    print(
        "Nya entity-kandidater: "
        f"{summary['new_entity_candidates']:,}"
    )

    print(
        "Befintliga entities  : "
        f"{summary['existing_entities_reused']:,}"
    )

    print(
        "Entity-konflikter    : "
        f"{summary['conflicting_entities']:,}"
    )

    print(
        "Nya aliasposter      : "
        f"{summary['new_alias_records']:,}"
    )

    print(
        "Redan importerade    : "
        f"{summary['already_present']:,}"
    )

    print()
    print(
        "EXEMPEL PÅ ENTITY-KANDIDATER"
    )
    print("-" * 70)

    count = 0

    for entity in entities:
        if entity.get(
            "status"
        ) != "new_entity_candidate":
            continue

        print(
            f"{entity['entity_id']} | "
            f"LEI={entity['lei']} | "
            f"{entity.get('legal_name')} | "
            f"{entity['instrument_count']} instrument"
        )

        count += 1

        if count >= 30:
            break

    print()
    print(
        "EXEMPEL PÅ NYA ALIASPOSTER"
    )
    print("-" * 70)

    for record in aliases[:30]:
        print(
            f"{record['entity_id']} | "
            f"{record['isin']} | "
            f"{record['lei']} | "
            f"{record['issuer']}"
        )


def main() -> None:
    args = _parse_args()

    print(
        "Läser discovery-rapport:"
    )
    print(
        DISCOVERY_PATH
    )

    discovery = _load_discovery()
    aliases = _load_alias_records()

    registry = EntityRegistry()

    # Första körningen migrerar befintliga ENT-ID:n från
    # aliasregistret till det nya centrala registret.
    _migrate_entities_from_aliases(
        registry,
        aliases,
    )

    (
        entities,
        import_records,
        summary,
    ) = _build_import_plan(
        discovery,
        aliases,
        registry,
    )

    _print_plan(
        entities,
        import_records,
        summary,
    )

    if not args.apply:
        print()
        print("=" * 70)
        print(
            "DRY-RUN"
        )
        print("=" * 70)
        print()
        print(
            "Inga filer ändrades."
        )
        print()
        print(
            "Entity-register:"
        )
        print(
            ENTITY_REGISTRY_PATH
        )
        print()
        print(
            "Alias-register:"
        )
        print(
            ALIASES_PATH
        )

        return

    _apply_import(
        entities,
        import_records,
        registry,
    )

    print()
    print("=" * 70)
    print(
        "IMPORT KLAR"
    )
    print("=" * 70)

    print()
    print(
        f"Nya entity-poster: "
        f"{summary['new_entity_candidates']:,}"
    )

    print(
        f"Nya aliasposter: "
        f"{len(import_records):,}"
    )

    print()
    print(
        "Entity-register:"
    )
    print(
        ENTITY_REGISTRY_PATH
    )

    print()
    print(
        "Alias-register:"
    )
    print(
        ALIASES_PATH
    )

    print()
    print(
        "instrument_map.json har inte ändrats."
    )


if __name__ == "__main__":
    main()
