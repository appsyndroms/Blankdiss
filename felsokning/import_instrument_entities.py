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
    INSTRUMENT_ALIASES_PATH,
    IdentityContractError,
    InstrumentIdentity,
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

ALIASES_PATH = INSTRUMENT_ALIASES_PATH

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

    records: list[dict[str, Any]] = []

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


def _existing_observation_keys(
    records: list[dict[str, Any]],
) -> set[
    tuple[str, str, str, str, str, str, str]
]:
    """
    Importen är idempotent per observerad identitet:

        entity_id
        + ISIN
        + LEI
        + Yahoo-symbol
        + issuer
        + source
        + observed_date

    Två observationer med samma entity + ISIN men
    olika issuer-namn eller olika observed_date
    är inte dubbletter.

    De representerar historiska observationer och
    ska därför kunna samexistera.

    En exakt upprepad observation importeras däremot
    inte två gånger.
    """

    keys: set[
        tuple[str, str, str, str, str, str, str]
    ] = set()

    for record in records:
        entity_id = record.get(
            "entity_id"
        )

        isin = normalize_isin(
            record.get("isin")
        )

        lei = normalize_lei(
            record.get("lei")
        )

        yahoo_symbol = normalize_text(
            record.get("yahoo_symbol")
        )

        issuer = normalize_text(
            record.get("issuer")
        )

        source = normalize_text(
            record.get("source")
        )

        observed_date = normalize_text(
            record.get("observed_date")
        )

        if not entity_id or not isin:
            continue

        keys.add(
            (
                str(entity_id),
                isin,
                lei,
                yahoo_symbol,
                issuer,
                source,
                observed_date,
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
    Grupperar discovery-resultatet på GLEIF-LEI.

    Discovery sparar de klassificerade instrumenten under
    top-level-fältet "instruments". "candidate_entities"
    är endast en diagnostisk delmängd för LEI:n som ännu
    saknar entity.

    Importen måste därför läsa "instruments" och inte
    "candidate_entities".

    Följande discovery-statusar kan importeras:

        entity_resolved_by_gleif
        gleif_lei_without_entity

    Statusar utan en entydig GLEIF-koppling importeras inte
    här och hanteras av discovery/identity-kontraktet.
    """

    groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    instruments = discovery.get(
        "instruments",
        [],
    )

    if not isinstance(
        instruments,
        list,
    ):
        raise ValueError(
            "Discovery-rapportens "
            "'instruments' måste vara en lista."
        )

    for instrument in instruments:
        if not isinstance(
            instrument,
            dict,
        ):
            continue

        status = instrument.get(
            "status"
        )

        if status not in {
            "entity_resolved_by_gleif",
            "gleif_lei_without_entity",
        }:
            continue

        isin = normalize_isin(
            instrument.get("isin")
        )

        if not isin:
            continue

        gleif_leis = [
            normalize_lei(lei)
            for lei in instrument.get(
                "gleif_leis",
                [],
            )
        ]

        gleif_leis = sorted(
            {
                lei
                for lei in gleif_leis
                if lei
            }
        )

        if len(gleif_leis) != 1:
            continue

        lei = gleif_leis[0]

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
                "discovery_status": status,
                "discovery_entity_ids": sorted(
                    {
                        str(entity_id)
                        for entity_id in instrument.get(
                            "entity_ids",
                            [],
                        )
                        if entity_id
                    }
                ),
            }
        )

    return {
        lei: sorted(
            instruments,
            key=lambda instrument: (
                instrument.get("isin") or "",
                instrument.get("issuer") or "",
            ),
        )
        for lei, instruments in sorted(
            groups.items()
        )
    }


def _migrate_entities_from_aliases(
    registry: EntityRegistry,
    aliases: list[dict[str, Any]],
) -> bool:
    """
    Säkerställer att befintliga entity-id:n i aliasregistret
    också finns i det centrala entity-registret.

    Returnerar True om registret behöver sparas.

    Funktionen skriver inte själv.

    Detta är viktigt eftersom dry-run ska vara helt
    utan sidoeffekter.
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

    return changed


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

    existing_keys = _existing_observation_keys(
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
        # 1. LEI är den starkaste externa entity-bryggan.
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
            # 2. ISIN används som identitetsbrygga.
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
                # 3. Ingen identitetsbrygga.
                #
                #    Skapa ny persistent entity.
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
                "observed_date": observed_date,
            }
        )

        for instrument in instruments:
            isin = instrument[
                "isin"
            ]

            yahoo_symbol = normalize_text(
                instrument.get(
                    "yahoo_symbol"
                )
            )

            issuer = normalize_text(
                instrument.get(
                    "issuer"
                )
            )

            key = (
                entity_id,
                isin,
                lei,
                yahoo_symbol,
                issuer,
                SOURCE,
                normalize_text(
                    observed_date
                ),
            )

            if key in existing_keys:
                skipped_existing += 1
                continue

            record = {
                "entity_id": entity_id,
                "isin": isin,
                "issuer": issuer,
                "yahoo_symbol": yahoo_symbol,
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


def _validate_import_plan(
    aliases: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    import_records: list[dict[str, Any]],
) -> None:
    """
    Validerar den kompletta importplanen innan något skrivs.

    Detta är sista spärren mellan discovery och persistent data.

    Reglerna kommer från det centrala identity-lagret:

        entity_id -> måste finnas
        ISIN      -> får bara peka på en entity
        LEI       -> får bara peka på en entity

    Skillnader i issuer, Yahoo-symbol eller observationstid
    är däremot tillåtna och representerar historiska
    observationer.

    En importplan som redan innehåller en upptäckt LEI- eller
    ISIN-konflikt stoppas direkt. Konflikten får inte filtreras
    bort bara för att själva konfliktgruppen saknar
    import_records.
    """

    planned_conflicts = [
        entity
        for entity in entities
        if entity.get(
            "status"
        ) == "conflict"
    ]

    if planned_conflicts:
        conflict_messages: list[str] = []

        for entity in planned_conflicts:
            lei = entity.get(
                "lei"
            )

            entity_ids = ", ".join(
                str(entity_id)
                for entity_id in entity.get(
                    "entity_ids",
                    [],
                )
            )

            if lei:
                conflict_messages.append(
                    "LEI "
                    f"{lei} tillhör flera entities: "
                    f"{entity_ids}"
                )
            else:
                conflict_messages.append(
                    "Identity-konflikt för "
                    f"entities: {entity_ids}"
                )

        raise IdentityContractError(
            "Importplanen innehåller identity-konflikter:\n"
            + "\n".join(
                f"  - {message}"
                for message in sorted(
                    set(
                        conflict_messages
                    )
                )
            )
        )

    identity = InstrumentIdentity(
        aliases_path=ALIASES_PATH,
        entity_registry_path=ENTITY_REGISTRY_PATH,
    )

    # Nya entities finns ännu inte i det persistenta registret.
    # Lägg därför till dem endast i minnet så att de kan
    # valideras mot samma identity contract.
    for entity in entities:
        if entity.get(
            "status"
        ) != "new_entity_candidate":
            continue

        entity_id = entity.get(
            "entity_id"
        )

        if not entity_id:
            continue

        if identity.entity_registry.contains(
            str(entity_id)
        ):
            continue

        identity.entity_registry.add(
            entity_id=str(
                entity_id
            ),
            legal_name=entity.get(
                "legal_name"
            ),
            observed_date=entity.get(
                "observed_date"
            ) or "",
            source=SOURCE,
            status=STATUS,
        )

    planned_records = (
        list(aliases)
        + list(import_records)
    )

    conflicts: list[str] = []

    for record in import_records:
        record_conflicts = (
            identity.validate_observation(
                record,
                existing_records=planned_records,
            )
        )

        for conflict in record_conflicts:
            conflicts.append(
                conflict.message
            )

    if conflicts:
        unique_conflicts = sorted(
            set(conflicts)
        )

        message = (
            "Importplanen bryter mot "
            "identity contract:\n"
            + "\n".join(
                f"  - {conflict}"
                for conflict in unique_conflicts
            )
        )

        raise IdentityContractError(
            message
        )

    # Slutlig kontroll av hela identity-datasetet.
    #
    # InstrumentIdentity.validate_identity_data()
    # arbetar mot befintliga records, så ersätt records
    # tillfälligt med den kompletta planen.
    identity.records = planned_records
    identity.validate_identity_data()


def _apply_import(
    entities: list[dict[str, Any]],
    aliases: list[dict[str, Any]],
    registry: EntityRegistry,
    migration_changed: bool,
) -> None:
    """
    Skriver entity-registret och aliasregistret.

    Alla valideringar måste redan ha passerat innan denna
    funktion anropas.
    """

    for entity in entities:
        if entity.get(
            "status"
        ) != "new_entity_candidate":
            continue

        registry.add(
            entity_id=entity["entity_id"],
            legal_name=entity.get(
                "legal_name"
            ),
            observed_date=entity.get(
                "observed_date"
            ) or "",
            source=SOURCE,
            status=STATUS,
        )

    if migration_changed or entities:
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
            f"{record.get('yahoo_symbol')} | "
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

    # Första körningen kan behöva migrera befintliga
    # entity-id:n från aliasregistret till det centrala
    # registret.
    #
    # Migrationen sker endast i minnet under dry-run.
    migration_changed = (
        _migrate_entities_from_aliases(
            registry,
            aliases,
        )
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

    # --------------------------------------------------------------
    # VIKTIGT:
    #
    # Discovery får aldrig direkt bli persistent data.
    #
    # Hela planen måste först passera identity contract.
    # --------------------------------------------------------------

    _validate_import_plan(
        aliases,
        entities,
        import_records,
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
            "Identity contract : OK"
        )

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
        migration_changed,
    )

    print()
    print("=" * 70)
    print(
        "IMPORT KLAR"
    )
    print("=" * 70)

    print()
    print(
        "Identity contract : OK"
    )

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
