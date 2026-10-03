from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.request
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from shared.instrument_identity import (
    InstrumentIdentity,
    normalize_isin,
    normalize_lei,
    normalize_identifier,
    normalize_text,
    parse_date,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "instrument_entity_discovery.json"
)

GLEIF_URL_TEMPLATE = (
    "https://isinmapping.gleif.org/"
    "api/v2/isin-lei/{date}/download"
)

DEFAULT_DATE = date.today().isoformat()

SAMPLE_LIMIT = 50


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Matchar Blankdiss instrument_map mot "
            "GLEIF/ANNA ISIN→LEI mapping."
        )
    )

    parser.add_argument(
        "--date",
        dest="mapping_date",
        default=DEFAULT_DATE,
        help=(
            "Datum för GLEIF-mappingen "
            "(YYYY-MM-DD)."
        ),
    )

    parser.add_argument(
        "--url",
        dest="url",
        default=None,
        help=(
            "Valfri explicit GLEIF-URL. "
            "Överstyr --date."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP-timeout i sekunder.",
    )

    return parser.parse_args()


def _mapping_url(
    mapping_date: str,
) -> str:
    parsed_date = parse_date(
        mapping_date
    )

    if parsed_date is None:
        raise ValueError(
            f"Ogiltigt mappingdatum: "
            f"{mapping_date}"
        )

    return GLEIF_URL_TEMPLATE.format(
        date=parsed_date.isoformat()
    )


def _download_mapping(
    url: str,
    timeout: int,
) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Blankdiss/"
                "instrument-identity"
            ),
            "Accept": (
                "application/zip,"
                "application/octet-stream,"
                "*/*"
            ),
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            return response.read()

    except Exception as exc:
        raise RuntimeError(
            f"Kunde inte hämta GLEIF-mapping: "
            f"{url}"
        ) from exc


def _find_csv(
    payload: bytes,
) -> tuple[str, bytes]:
    try:
        archive = zipfile.ZipFile(
            io.BytesIO(payload)
        )

    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            "GLEIF-svaret var inte en giltig ZIP-fil."
        ) from exc

    csv_names = [
        name
        for name in archive.namelist()
        if name.lower().endswith(
            ".csv"
        )
    ]

    if not csv_names:
        raise RuntimeError(
            "Ingen CSV-fil hittades i "
            "GLEIF-arkivet."
        )

    # Mappingarkivet kan innehålla metadatafiler.
    # Välj den största CSV-filen eftersom själva
    # relationstabellen normalt är den största.
    csv_name = max(
        csv_names,
        key=lambda name: archive.getinfo(
            name
        ).file_size,
    )

    return (
        csv_name,
        archive.read(
            csv_name
        ),
    )


def _decode_csv(
    payload: bytes,
) -> list[dict[str, str]]:
    encodings = (
        "utf-8-sig",
        "utf-8",
        "latin-1",
    )

    last_error: Exception | None = None

    for encoding in encodings:
        try:
            text = payload.decode(
                encoding
            )

            break

        except UnicodeDecodeError as exc:
            last_error = exc

    else:
        raise RuntimeError(
            "Kunde inte avkoda GLEIF CSV."
        ) from last_error

    sample = text[
        :10000
    ]

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=(
                ",",
                ";",
                "\t",
                "|",
            ),
        )

    except csv.Error:
        dialect = csv.excel

    reader = csv.DictReader(
        io.StringIO(text),
        dialect=dialect,
    )

    if not reader.fieldnames:
        raise RuntimeError(
            "GLEIF CSV saknar kolumnrubriker."
        )

    return [
        {
            str(key).strip(): (
                value.strip()
                if isinstance(
                    value,
                    str,
                )
                else ""
            )
            for key, value in row.items()
            if key is not None
        }
        for row in reader
    ]


def _find_column(
    fieldnames: list[str],
    candidates: tuple[str, ...],
) -> str | None:
    normalized = {
        field.strip().lower(): field
        for field in fieldnames
    }

    for candidate in candidates:
        field = normalized.get(
            candidate.lower()
        )

        if field:
            return field

    for field in fieldnames:
        compact = (
            field
            .strip()
            .lower()
            .replace(
                "_",
                "",
            )
            .replace(
                "-",
                "",
            )
            .replace(
                " ",
                "",
            )
        )

        for candidate in candidates:
            candidate_compact = (
                candidate
                .lower()
                .replace(
                    "_",
                    "",
                )
                .replace(
                    "-",
                    "",
                )
                .replace(
                    " ",
                    "",
                )
            )

            if compact == candidate_compact:
                return field

    return None


def _build_gleif_index(
    rows: list[dict[str, str]],
) -> dict[str, set[str]]:
    if not rows:
        return {}

    fieldnames = list(
        rows[0].keys()
    )

    isin_column = _find_column(
        fieldnames,
        (
            "ISIN",
            "isin",
            "isin_code",
            "isinCode",
        ),
    )

    lei_column = _find_column(
        fieldnames,
        (
            "LEI",
            "lei",
            "lei_code",
            "leiCode",
        ),
    )

    if not isin_column:
        raise RuntimeError(
            "Kunde inte hitta ISIN-kolumn "
            "i GLEIF CSV."
        )

    if not lei_column:
        raise RuntimeError(
            "Kunde inte hitta LEI-kolumn "
            "i GLEIF CSV."
        )

    index: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for row in rows:
        isin = normalize_isin(
            row.get(
                isin_column
            )
        )

        lei = normalize_lei(
            row.get(
                lei_column
            )
        )

        if not isin or not lei:
            continue

        index[
            isin
        ].add(
            lei
        )

    return dict(
        index
    )


def _entity_ids_for_lei(
    identity: InstrumentIdentity,
    lei: str,
) -> list[str]:
    normalized_lei = normalize_lei(
        lei
    )

    if not normalized_lei:
        return []

    entity_ids: set[str] = set()

    for record in identity.records:
        record_lei = normalize_lei(
            record.get(
                "lei"
            )
        )

        if record_lei != normalized_lei:
            continue

        entity_id = record.get(
            "entity_id"
        )

        if entity_id:
            entity_ids.add(
                str(entity_id)
            )

    return sorted(
        entity_ids
    )


def _classify_instrument(
    identity: InstrumentIdentity,
    record: dict[str, Any],
    gleif_index: dict[str, set[str]],
) -> dict[str, Any]:
    isin = normalize_isin(
        record.get(
            "isin"
        )
    )

    issuer = normalize_text(
        record.get(
            "issuer"
        )
    )

    instrument_lei = normalize_lei(
        record.get(
            "lei"
        )
    )

    yahoo_symbol = normalize_identifier(
        record.get(
            "yahoo_symbol"
        )
    )

    ticker = normalize_identifier(
        record.get(
            "ticker"
        )
    )

    result: dict[str, Any] = {
        "isin": isin,
        "issuer": issuer,
        "instrument_lei": instrument_lei,
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
        "gleif_leis": [],
        "entity_ids": [],
        "status": None,
    }

    if not isin:
        result[
            "status"
        ] = "missing_isin"

        return result

    gleif_leis = sorted(
        gleif_index.get(
            isin,
            set(),
        )
    )

    result[
        "gleif_leis"
    ] = gleif_leis

    if not gleif_leis:
        result[
            "status"
        ] = "no_gleif_mapping"

        return result

    entity_ids: set[str] = set()

    for lei in gleif_leis:
        entity_ids.update(
            _entity_ids_for_lei(
                identity,
                lei,
            )
        )

    result[
        "entity_ids"
    ] = sorted(
        entity_ids
    )

    if len(
        gleif_leis
    ) > 1:
        result[
            "status"
        ] = "multiple_gleif_leis"

        return result

    if len(
        entity_ids
    ) > 1:
        result[
            "status"
        ] = "multiple_entities"

        return result

    if len(
        entity_ids
    ) == 1:
        result[
            "status"
        ] = "entity_resolved_by_gleif"

        return result

    result[
        "status"
    ] = "gleif_lei_without_entity"

    return result


def _print_summary(
    counts: dict[str, int],
) -> None:
    print()
    print("=" * 70)
    print(
        "GLEIF / ENTITY-STATUS"
    )
    print("=" * 70)

    statuses = (
        "entity_resolved_by_gleif",
        "gleif_lei_without_entity",
        "multiple_entities",
        "multiple_gleif_leis",
        "no_gleif_mapping",
        "missing_isin",
    )

    for status in statuses:
        print(
            f"{status:35}: "
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
        print(
            f"{value.get('isin') or '-'} | "
            f"{value.get('yahoo_symbol') or '-'} | "
            f"GLEIF LEI="
            f"{','.join(value.get('gleif_leis', [])) or '-'} | "
            f"entity="
            f"{','.join(value.get('entity_ids', [])) or '-'} | "
            f"{value.get('issuer') or '-'}"
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


def _build_lei_groups(
    results: list[dict[str, Any]],
) -> dict[str, list[str]]:
    groups: dict[
        str,
        list[str],
    ] = defaultdict(list)

    for result in results:
        isin = result.get(
            "isin"
        )

        if not isin:
            continue

        for lei in result.get(
            "gleif_leis",
            [],
        ):
            groups[
                lei
            ].append(
                isin
            )

    return {
        lei: sorted(
            set(isins)
        )
        for lei, isins
        in sorted(
            groups.items()
        )
    }


def _build_candidate_entities(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Skapar kandidatgrupper för LEI:n som GLEIF
    känner till men som ännu saknas i vårt
    centrala entity-register.

    Detta skapar INTE entities.
    """

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for result in results:
        if result.get(
            "status"
        ) != "gleif_lei_without_entity":
            continue

        for lei in result.get(
            "gleif_leis",
            [],
        ):
            grouped[
                lei
            ].append(
                {
                    "isin": result.get(
                        "isin"
                    ),
                    "issuer": result.get(
                        "issuer"
                    ),
                    "yahoo_symbol": result.get(
                        "yahoo_symbol"
                    ),
                    "instrument_lei": result.get(
                        "instrument_lei"
                    ),
                }
            )

    return [
        {
            "lei": lei,
            "instrument_count": len(
                instruments
            ),
            "instruments": sorted(
                instruments,
                key=lambda value: (
                    value.get(
                        "issuer"
                    )
                    or "",
                    value.get(
                        "isin"
                    )
                    or "",
                ),
            ),
            "candidate_status": "new_entity_candidate",
            "source": "GLEIF_ANNA",
        }
        for lei, instruments
        in sorted(
            grouped.items()
        )
    ]


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
    args = _parse_args()

    print("=" * 70)
    print(
        "INSTRUMENT → ENTITY DISCOVERY"
    )
    print("=" * 70)

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

    url = (
        args.url
        or _mapping_url(
            args.mapping_date
        )
    )

    print()
    print(
        "GLEIF / ANNA"
    )
    print("-" * 70)

    print(
        f"Datum: "
        f"{args.mapping_date}"
    )

    print(
        f"URL: "
        f"{url}"
    )

    payload = _download_mapping(
        url,
        args.timeout,
    )

    csv_name, csv_payload = _find_csv(
        payload
    )

    print(
        f"CSV: "
        f"{csv_name}"
    )

    rows = _decode_csv(
        csv_payload
    )

    print(
        f"GLEIF-rader: "
        f"{len(rows):,}"
    )

    gleif_index = _build_gleif_index(
        rows
    )

    print(
        f"ISIN→LEI-mappningar: "
        f"{len(gleif_index):,}"
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
            gleif_index,
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
        "ENTITY RESOLVED VIA GLEIF",
        examples[
            "entity_resolved_by_gleif"
        ],
    )

    _print_samples(
        "GLEIF-LEI FINNS – ENTITY SAKNAS",
        examples[
            "gleif_lei_without_entity"
        ],
    )

    _print_samples(
        "FLERA ENTITYS",
        examples[
            "multiple_entities"
        ],
    )

    _print_samples(
        "FLERA GLEIF-LEI",
        examples[
            "multiple_gleif_leis"
        ],
    )

    _print_samples(
        "INGEN GLEIF-MAPPNING",
        examples[
            "no_gleif_mapping"
        ],
    )

    _print_samples(
        "SAKNAR ISIN",
        examples[
            "missing_isin"
        ],
    )

    lei_groups = _build_lei_groups(
        results
    )

    candidate_entities = (
        _build_candidate_entities(
            results
        )
    )

    print()
    print("=" * 70)
    print(
        "NYA ENTITY-KANDIDATER"
    )
    print("=" * 70)

    print(
        f"LEI-grupper utan entity: "
        f"{len(candidate_entities):,}"
    )

    for candidate in candidate_entities[
        :SAMPLE_LIMIT
    ]:
        print()
        print(
            f"LEI={candidate['lei']} | "
            f"{candidate['instrument_count']:,} instrument"
        )

        for instrument in candidate[
            "instruments"
        ][
            :10
        ]:
            print(
                f"  "
                f"{instrument['isin']} | "
                f"{instrument['yahoo_symbol'] or '-'} | "
                f"{instrument['issuer'] or '-'}"
            )

        if candidate[
            "instrument_count"
        ] > 10:
            print(
                f"  ... "
                f"{candidate['instrument_count'] - 10:,} fler"
            )

    report = {
        "source": {
            "provider": "GLEIF_ANNA",
            "mapping_date": args.mapping_date,
            "url": url,
            "csv": csv_name,
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
        "mapping": {
            "gleif_row_count": len(
                rows
            ),
            "isin_lei_count": len(
                gleif_index
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
        "lei_groups": lei_groups,
        "candidate_entities": (
            candidate_entities
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
        "Detta var endast discovery/diagnostik."
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
