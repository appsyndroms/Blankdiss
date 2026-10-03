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


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "instrument_entity_diagnostic.json"
)

SAMPLE_LIMIT = 50


def _instrument_description(
    record: dict[str, Any],
) -> str:
    parts: list[str] = []

    isin = normalize_isin(
        record.get("isin")
    )

    issuer = normalize_text(
        record.get("issuer")
    )

    lei = normalize_lei(
        record.get("lei")
    )

    yahoo_symbol = normalize_identifier(
        record.get("yahoo_symbol")
    )

    ticker = normalize_identifier(
        record.get("ticker")
    )

    if isin:
        parts.append(
            f"ISIN={isin}"
        )

    if issuer:
        parts.append(
            f"issuer={issuer}"
        )

    if lei:
        parts.append(
            f"lei={lei}"
        )

    if yahoo_symbol:
        parts.append(
            f"yahoo={yahoo_symbol}"
        )

    if ticker:
        parts.append(
            f"ticker={ticker}"
        )

    if not parts:
        return "-"

    return " ".join(
        parts
    )


def _entity_observations(
    identity: InstrumentIdentity,
    isin: str,
) -> list[dict[str, Any]]:
    observations = (
        identity.entities_for_instrument(
            isin
        )
    )

    result: list[
        dict[str, Any]
    ] = []

    seen: set[
        tuple[str, str]
    ] = set()

    for observation in observations:
        entity_id = observation.entity_id

        record = observation.record

        key = (
            entity_id,
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
            ),
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            {
                "entity_id": entity_id,
                "relation": record.get(
                    "relation"
                ),
                "source": record.get(
                    "source"
                ),
                "status": record.get(
                    "status"
                ),
                "valid_from": record.get(
                    "valid_from"
                ),
                "valid_to": record.get(
                    "valid_to"
                ),
                "observed_date": record.get(
                    "observed_date"
                ),
                "issuer": record.get(
                    "issuer"
                ),
                "lei": record.get(
                    "lei"
                ),
            }
        )

    return result


def _classify_instrument(
    identity: InstrumentIdentity,
    record: dict[str, Any],
) -> dict[str, Any]:
    isin = normalize_isin(
        record.get("isin")
    )

    issuer = normalize_text(
        record.get("issuer")
    )

    lei = normalize_lei(
        record.get("lei")
    )

    yahoo_symbol = normalize_identifier(
        record.get("yahoo_symbol")
    )

    ticker = normalize_identifier(
        record.get("ticker")
    )

    result: dict[str, Any] = {
        "isin": isin,
        "issuer": issuer,
        "lei": lei,
        "yahoo_symbol": yahoo_symbol,
        "ticker": ticker,
        "map_key": record.get(
            "map_key"
        ),
        "mapping_source": record.get(
            "mapping_source"
        ),
        "mapping_status": record.get(
            "mapping_status"
        ),
        "mapping_confidence": record.get(
            "mapping_confidence"
        ),
        "description": _instrument_description(
            record
        ),
        "entity_ids": [],
        "entity_observations": [],
    }

    if not isin:
        result[
            "status"
        ] = "missing_isin"

        return result

    observations = _entity_observations(
        identity,
        isin,
    )

    entity_ids = sorted(
        {
            observation[
                "entity_id"
            ]
            for observation in observations
            if observation.get(
                "entity_id"
            )
        }
    )

    result[
        "entity_ids"
    ] = entity_ids

    result[
        "entity_observations"
    ] = observations

    if not observations:
        result[
            "status"
        ] = "no_entity_observation"

    elif len(entity_ids) == 1:
        result[
            "status"
        ] = "unique_entity"

    else:
        result[
            "status"
        ] = "multiple_entities"

    return result


def _print_header() -> None:
    print()
    print("=" * 70)
    print(
        "INSTRUMENT / ENTITY FELSÖKNING"
    )
    print("=" * 70)


def _print_summary(
    counts: dict[str, int],
) -> None:
    print()
    print("=" * 70)
    print(
        "ENTITY-KOPPLING"
    )
    print("=" * 70)

    statuses = (
        "unique_entity",
        "multiple_entities",
        "no_entity_observation",
        "missing_isin",
    )

    for status in statuses:
        print(
            f"{status:30}: "
            f"{counts[status]:,}"
        )


def _print_samples(
    title: str,
    values: list[dict[str, Any]],
) -> None:
    print()
    print(title)
    print("-" * 70)

    if not values:
        print(
            "Inga."
        )

        return

    for value in values[
        :SAMPLE_LIMIT
    ]:
        isin = (
            value.get(
                "isin"
            )
            or "-"
        )

        issuer = (
            value.get(
                "issuer"
            )
            or "-"
        )

        lei = (
            value.get(
                "lei"
            )
            or "-"
        )

        yahoo_symbol = (
            value.get(
                "yahoo_symbol"
            )
            or "-"
        )

        entity_ids = ",".join(
            value.get(
                "entity_ids",
                [],
            )
        ) or "-"

        print(
            f"{isin} | "
            f"{yahoo_symbol} | "
            f"entity={entity_ids} | "
            f"issuer={issuer} | "
            f"lei={lei}"
        )

    if len(
        values
    ) > SAMPLE_LIMIT:
        print()
        print(
            f"... "
            f"{len(values) - SAMPLE_LIMIT:,} "
            "ytterligare poster."
        )


def _build_entity_groups(
    values: list[dict[str, Any]],
) -> dict[str, list[str]]:
    groups: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for value in values:
        for entity_id in value.get(
            "entity_ids",
            [],
        ):
            isin = value.get(
                "isin"
            )

            if not isin:
                continue

            groups[
                entity_id
            ].append(
                isin
            )

    return {
        entity_id: sorted(
            set(isins)
        )
        for entity_id, isins
        in sorted(
            groups.items()
        )
    }


def _build_missing_entity_groups(
    values: list[dict[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    """
    Grupperar instrument som saknar entity-observation.

    LEI används som primär kandidatnyckel eftersom samma LEI
    normalt representerar den juridiska entiteten över flera
    instrument.

    Detta är endast diagnostik.

    Ingen entity skapas automatiskt.
    """

    groups: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for value in values:
        if value.get(
            "status"
        ) != "no_entity_observation":
            continue

        isin = value.get(
            "isin"
        ) or ""

        lei = value.get(
            "lei"
        ) or ""

        issuer = value.get(
            "issuer"
        ) or ""

        yahoo_symbol = value.get(
            "yahoo_symbol"
        ) or ""

        if lei:
            group_key = (
                "LEI:"
                + lei
            )

        elif issuer:
            group_key = (
                "ISSUER:"
                + issuer
            )

        else:
            group_key = (
                "UNIDENTIFIED:"
                + isin
            )

        groups[
            group_key
        ].append(
            {
                "isin": isin,
                "issuer": issuer,
                "lei": lei,
                "yahoo_symbol": yahoo_symbol,
            }
        )

    return {
        key: sorted(
            values,
            key=lambda value: (
                value.get(
                    "issuer"
                ) or "",
                value.get(
                    "isin"
                ) or "",
            ),
        )
        for key, values
        in sorted(
            groups.items()
        )
    }


def _write_report(
    report: dict[str, Any],
) -> None:
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    _print_header()

    identity = InstrumentIdentity()

    print()
    print(
        "IDENTITY-REGISTER"
    )
    print("-" * 70)

    print(
        f"Observationer : "
        f"{len(identity.records):,}"
    )

    print(
        f"Entity-id:n   : "
        f"{len(identity._entity_ids()):,}"
    )

    print(
        f"Instrument map: "
        f"{len(identity.instrument_map):,}"
    )

    results: list[
        dict[str, Any]
    ] = []

    counts: dict[
        str,
        int,
    ] = defaultdict(int)

    examples: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in identity.instrument_map.values():
        result = _classify_instrument(
            identity,
            record,
        )

        results.append(
            result
        )

        status = result[
            "status"
        ]

        counts[
            status
        ] += 1

        if len(
            examples[status]
        ) < SAMPLE_LIMIT:
            examples[
                status
            ].append(
                result
            )

    _print_summary(
        counts
    )

    _print_samples(
        "UNIK ENTITY",
        examples[
            "unique_entity"
        ],
    )

    _print_samples(
        "FLERA ENTITY-KANDIDATER",
        examples[
            "multiple_entities"
        ],
    )

    _print_samples(
        "INSTRUMENT UTAN ENTITY-OBSERVATION",
        examples[
            "no_entity_observation"
        ],
    )

    _print_samples(
        "INSTRUMENT UTAN ISIN",
        examples[
            "missing_isin"
        ],
    )

    entity_groups = _build_entity_groups(
        results
    )

    missing_entity_groups = (
        _build_missing_entity_groups(
            results
        )
    )

    print()
    print("=" * 70)
    print(
        "ENTITY-GRUPPER"
    )
    print("=" * 70)

    for entity_id, isins in entity_groups.items():
        print(
            f"{entity_id}: "
            f"{len(isins):,} ISIN"
        )

    print()
    print("=" * 70)
    print(
        "SAKNADE ENTITY-GRUPPER"
    )
    print("=" * 70)

    print(
        f"Grupper: "
        f"{len(missing_entity_groups):,}"
    )

    for group_key, instruments in list(
        missing_entity_groups.items()
    )[
        :SAMPLE_LIMIT
    ]:
        print()
        print(
            f"{group_key} "
            f"({len(instruments):,} instrument)"
        )

        for instrument in instruments[
            :10
        ]:
            print(
                f"  "
                f"{instrument['isin']} | "
                f"{instrument['yahoo_symbol'] or '-'} | "
                f"{instrument['issuer'] or '-'}"
            )

        if len(
            instruments
        ) > 10:
            print(
                f"  ... "
                f"{len(instruments) - 10:,} fler"
            )

    report = {
        "source": {
            "instrument_map": str(
                identity.instrument_map_path
            ),
            "instrument_aliases": str(
                identity.aliases_path
            ),
        },
        "identity": {
            "observation_count": len(
                identity.records
            ),
            "entity_count": len(
                identity._entity_ids()
            ),
            "instrument_map_count": len(
                identity.instrument_map
            ),
        },
        "classification": {
            "instrument_count": len(
                results
            ),
            "counts": dict(
                sorted(
                    counts.items()
                )
            ),
        },
        "entity_groups": entity_groups,
        "missing_entity_groups": (
            missing_entity_groups
        ),
        "examples": {
            status: values
            for status, values
            in sorted(
                examples.items()
            )
        },
        "instruments": results,
    }

    _write_report(
        report
    )

    print()
    print("=" * 70)
    print(
        "SLUTSATS"
    )
    print("=" * 70)

    print()
    print(
        "Detta var endast diagnostik."
    )

    print(
        "instrument_map.json har inte ändrats."
    )

    print(
        "instrument_aliases.jsonl har inte ändrats."
    )

    print(
        "Inga entity-id:n har skapats."
    )

    print()
    print(
        f"Rapport: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
