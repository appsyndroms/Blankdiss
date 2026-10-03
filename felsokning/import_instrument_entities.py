from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    / "data"
    / "analysis"
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
            "Skriv kandidater till "
            "instrument_aliases.jsonl. "
            "Utan --apply körs endast dry-run."
        ),
    )

    return parser.parse_args()


def _load_discovery() -> dict[str, Any]:
    if not DISCOVERY_PATH.exists():
        raise FileNotFoundError(
            f"Discovery-rapport saknas: "
            f"{DISCOVERY_PATH}"
        )

    with DISCOVERY_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


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
                    f"Ogiltig JSON i "
                    f"{ALIASES_PATH} "
                    f"rad {line_number}."
                ) from exc

            if not isinstance(
                record,
                dict,
            ):
                raise ValueError(
                    f"Aliaspost på rad "
                    f"{line_number} är inte ett objekt."
                )

            records.append(
                record
            )

    return records


def _entity_ids_from_records(
    records: list[dict[str, Any]],
) -> set[str]:
    entity_ids: set[str] = set()

    for record in records:
        entity_id = record.get(
            "entity_id"
        )

        if entity_id:
            entity_ids.add(
                str(entity_id)
            )

    return entity_ids


def _next_entity_id(
    existing_ids: set[str],
) -> str:
    highest = 0

    for entity_id in existing_ids:
        if not entity_id.startswith(
            "ENT-"
        ):
            continue

        suffix = entity_id[
            4:
        ]

        if not suffix.isdigit():
            continue

        highest = max(
            highest,
            int(suffix),
        )

    return (
        f"ENT-{highest + 1:06d}"
    )


def _lei_to_existing_entities(
    records: list[dict[str, Any]],
) -> dict[str, set[str]]:
    mapping: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for record in records:
        lei = normalize_lei(
            record.get(
                "lei"
            )
        )

        entity_id = record.get(
            "entity_id"
        )

        if not lei or not entity_id:
            continue

        mapping[
            lei
        ].add(
            str(entity_id)
        )

    return dict(
        mapping
    )


def _existing_alias_keys(
    records: list[dict[str, Any]],
) -> set[
    tuple[str, str, str]
]:
    """
    Returnerar nycklar för redan importerade
    observationer.

    Kombinationen entity + ISIN + source gör
    importen idempotent.
    """

    keys: set[
        tuple[str, str, str]
    ] = set()

    for record in records:
        entity_id = record.get(
            "entity_id"
        )

        isin = normalize_isin(
            record.get(
                "isin"
            )
        )

        source = normalize_text(
            record.get(
                "source"
            )
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
) -> dict[str, list[dict[str, Any]]]:
    """
    Grupperar discovery-resultatet på LEI.

    Endast GLEIF-poster som ännu saknar
    entity inkluderas.
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
            candidate.get(
                "lei"
            )
        )

        if not lei:
            continue

        for instrument in candidate.get(
            "instruments",
            [],
        ):
            isin = normalize_isin(
                instrument.get(
                    "isin"
                )
            )

            if not isin:
                continue

            groups[
                lei
            ].append(
                {
                    "isin": isin,
                    "issuer": normalize_text(
                        instrument.get(
                            "issuer"
                        )
                    ),
                    "yahoo_symbol": normalize_text(
                        instrument.get(
                            "yahoo_symbol"
                        )
                    ),
                    "instrument_lei": normalize_lei(
                        instrument.get(
                            "instrument_lei"
                        )
                    ),
                }
            )

    return {
        lei: instruments
        for lei, instruments
        in sorted(
            groups.items()
        )
    }


def _build_import_plan(
    discovery: dict[str, Any],
    existing_records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    existing_entities = (
        _entity_ids_from_records(
            existing_records
        )
    )

    lei_entities = (
        _lei_to_existing_entities(
            existing_records
        )
    )

    existing_keys = (
        _existing_alias_keys(
            existing_records
        )
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

    import_records: list[
        dict[str, Any]
    ] = []

    entity_plan: list[
        dict[str, Any]
    ] = []

    skipped_existing = 0
    skipped_conflict = 0

    for lei, instruments in groups.items():
        known_entities = lei_entities.get(
            lei,
            set(),
        )

        if len(
            known_entities
        ) > 1:
            skipped_conflict += 1

            entity_plan.append(
                {
                    "lei": lei,
                    "status": "conflict",
                    "entity_ids": sorted(
                        known_entities
                    ),
                    "instrument_count": len(
                        instruments
                    ),
                }
            )

            continue

        if len(
            known_entities
        ) == 1:
            entity_id = next(
                iter(
                    known_entities
                )
            )

            entity_status = (
                "existing_entity"
            )

        else:
            entity_id = _next_entity_id(
                existing_entities
            )

            existing_entities.add(
                entity_id
            )

            lei_entities[
                lei
            ] = {
                entity_id
            }

            entity_status = (
                "new_entity_candidate"
            )

        entity_plan.append(
            {
                "lei": lei,
                "entity_id": entity_id,
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

    summary = {
        "discovery_mapping_date": observed_date,
        "candidate_lei_groups": len(
            groups
        ),
        "new_entity_candidates": sum(
            1
            for item in entity_plan
            if item.get(
                "status"
            )
            == "new_entity_candidate"
        ),
        "existing_entities_reused": sum(
            1
            for item in entity_plan
            if item.get(
                "status"
            )
            == "existing_entity"
        ),
        "conflicting_entities": skipped_conflict,
        "new_alias_records": len(
            import_records
        ),
        "already_present": skipped_existing,
    }

    return (
        import_records,
        {
            "summary": summary,
            "entity_plan": entity_plan,
        },
    )


def _append_records(
    records: list[dict[str, Any]],
) -> None:
    if not records:
        return

    ALIASES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with ALIASES_PATH.open(
        "a",
        encoding="utf-8",
    ) as handle:
        for record in records:
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

            handle.write(
                "\n"
            )


def _print_plan(
    plan: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    summary = plan[
        "summary"
    ]

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
        "EXEMPEL PÅ NYA ENTITY-KANDIDATER"
    )
    print("-" * 70)

    count = 0

    for item in plan[
        "entity_plan"
    ]:
        if item.get(
            "status"
        ) != "new_entity_candidate":
            continue

        print(
            f"{item['entity_id']} | "
            f"LEI={item['lei']} | "
            f"{item['instrument_count']} instrument"
        )

        count += 1

        if count >= 30:
            break

    print()
    print(
        "EXEMPEL PÅ NYA ALIASPOSTER"
    )
    print("-" * 70)

    for record in records[
        :30
    ]:
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
    existing_records = (
        _load_alias_records()
    )

    import_records, plan = (
        _build_import_plan(
            discovery,
            existing_records,
        )
    )

    _print_plan(
        plan,
        import_records,
    )

    if not args.apply:
        print()
        print(
            "=" * 70
        )
        print(
            "DRY-RUN"
        )
        print(
            "=" * 70
        )
        print()
        print(
            "Inga filer ändrades."
        )
        print()
        print(
            "Kör med --apply för att "
            "skriva kandidaterna till:"
        )
        print(
            ALIASES_PATH
        )

        return

    _append_records(
        import_records
    )

    print()
    print(
        "=" * 70
    )
    print(
        "IMPORT KLAR"
    )
    print(
        "=" * 70
    )

    print()
    print(
        f"Tillagda aliasposter: "
        f"{len(import_records):,}"
    )

    print(
        f"Fil: {ALIASES_PATH}"
    )

    print()
    print(
        "Alla importerade poster har:"
    )
    print(
        f"  source = {SOURCE}"
    )
    print(
        f"  status = {STATUS}"
    )
    print(
        f"  relation = {RELATION}"
    )

    print()
    print(
        "Inga price-data har ändrats."
    )

    print(
        "instrument_map.json har inte ändrats."
    )


if __name__ == "__main__":
    main()
