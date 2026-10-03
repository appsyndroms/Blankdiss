from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from shared.instrument_identity import (
    load_identity,
    normalize_isin,
)


PRICE_DIR = Path(
    "data/raw/prices"
)

ARCHIVE_DIR = PRICE_DIR / "archive"

REPORT_PATH = Path(
    "data/analysis/price_cleanup_report.json"
)

IDENTITY_FIELDS = (
    "yahoo_symbol",
    "lei",
    "issuer",
    "ticker",
)


def _parse_date(
    value: Any,
) -> date | None:
    if isinstance(value, date):
        return value

    if not isinstance(
        value,
        str,
    ):
        return None

    try:
        return date.fromisoformat(
            value[:10]
        )
    except ValueError:
        return None


def _normalise_isin(
    value: Any,
) -> str:
    if value is None:
        return ""

    return str(
        value
    ).strip().upper()


def _normalise_identity_value(
    field: str,
    value: Any,
) -> str:
    if value is None:
        return ""

    value = str(
        value
    ).strip()

    if not value:
        return ""

    if field in (
        "yahoo_symbol",
        "lei",
        "ticker",
    ):
        return value.upper()

    if field == "issuer":
        return " ".join(
            value.upper().split()
        )

    return value


def _normalise_symbol(
    value: Any,
) -> str:
    return _normalise_identity_value(
        "yahoo_symbol",
        value,
    )


def _parse_close(
    value: Any,
) -> float | None:
    if value is None:
        return None

    try:
        numeric = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        numeric
    ):
        return None

    return numeric


def _observation_key(
    record: dict[str, Any],
) -> tuple[str, str] | None:
    record_date = _parse_date(
        record.get("date")
    )

    isin = _normalise_isin(
        record.get("isin")
    )

    if record_date is None:
        return None

    if not isin:
        return None

    return (
        record_date.isoformat(),
        isin,
    )


def _load_price_files() -> list[Path]:
    return sorted(
        path
        for path in PRICE_DIR.glob(
            "prices_*.jsonl"
        )
        if path.is_file()
    )


def _load_records(
    paths: list[Path],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    records: list[
        dict[str, Any]
    ] = []

    statistics = {
        "files": 0,
        "lines": 0,
        "blank_lines": 0,
        "invalid_json": 0,
        "missing_date": 0,
        "invalid_date": 0,
        "missing_isin": 0,
        "missing_close": 0,
        "invalid_close": 0,
    }

    for path in paths:
        statistics[
            "files"
        ] += 1

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            for line_number, line in enumerate(
                handle,
                start=1,
            ):
                statistics[
                    "lines"
                ] += 1

                line = line.strip()

                if not line:
                    statistics[
                        "blank_lines"
                    ] += 1
                    continue

                try:
                    record = json.loads(
                        line
                    )
                except json.JSONDecodeError:
                    statistics[
                        "invalid_json"
                    ] += 1
                    continue

                if not isinstance(
                    record,
                    dict,
                ):
                    statistics[
                        "invalid_json"
                    ] += 1
                    continue

                raw_date = record.get(
                    "date"
                )

                if raw_date is None:
                    statistics[
                        "missing_date"
                    ] += 1
                    continue

                parsed_date = _parse_date(
                    raw_date
                )

                if parsed_date is None:
                    statistics[
                        "invalid_date"
                    ] += 1
                    continue

                close = _parse_close(
                    record.get("close")
                )

                if record.get("close") is None:
                    statistics[
                        "missing_close"
                    ] += 1
                    continue

                if close is None:
                    statistics[
                        "invalid_close"
                    ] += 1
                    continue

                clean_record = dict(
                    record
                )

                clean_record[
                    "date"
                ] = parsed_date.isoformat()

                clean_record[
                    "isin"
                ] = _normalise_isin(
                    record.get("isin")
                )

                clean_record[
                    "close"
                ] = close

                clean_record[
                    "_source_file"
                ] = path.name

                clean_record[
                    "_source_line"
                ] = line_number

                if not clean_record[
                    "isin"
                ]:
                    statistics[
                        "missing_isin"
                    ] += 1

                records.append(
                    clean_record
                )

    return (
        records,
        statistics,
    )


def _records_conflict(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    first_close = _parse_close(
        first.get("close")
    )

    second_close = _parse_close(
        second.get("close")
    )

    if (
        first_close is None
        or second_close is None
    ):
        return True

    return first_close != second_close


def _clean_output_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if not key.startswith("_")
    }


def _build_identity_index(
    records: list[dict[str, Any]],
) -> dict[
    str,
    dict[str, set[str]],
]:
    index: dict[
        str,
        dict[str, set[str]],
    ] = {
        field: defaultdict(set)
        for field in IDENTITY_FIELDS
    }

    for record in records:
        isin = _normalise_isin(
            record.get("isin")
        )

        if not isin:
            continue

        for field in IDENTITY_FIELDS:
            value = _normalise_identity_value(
                field,
                record.get(field),
            )

            if not value:
                continue

            index[
                field
            ][
                value
            ].add(
                isin
            )

    return index


def _resolve_missing_isin(
    record: dict[str, Any],
    identity_index: dict[
        str,
        dict[str, set[str]],
    ],
) -> dict[str, Any]:
    evidence: dict[
        str,
        set[str],
    ] = {}

    for field in IDENTITY_FIELDS:
        value = _normalise_identity_value(
            field,
            record.get(field),
        )

        if not value:
            continue

        known_isins = identity_index[
            field
        ].get(
            value,
            set(),
        )

        if known_isins:
            evidence[
                field
            ] = set(
                known_isins
            )

    all_candidates: set[str] = set()

    for candidates in evidence.values():
        all_candidates.update(
            candidates
        )

    if not evidence:
        status = "unresolved"
        candidate_isins: set[str] = set()

    else:
        intersection: set[str] | None = None

        for candidates in evidence.values():
            if intersection is None:
                intersection = set(
                    candidates
                )
            else:
                intersection &= candidates

        if intersection:
            candidate_isins = intersection

            if len(
                candidate_isins
            ) == 1:
                status = "unique"
            else:
                status = "ambiguous"

        else:
            candidate_isins = all_candidates
            status = "conflict"

    return {
        "status": status,
        "candidate_isins": sorted(
            candidate_isins
        ),
        "evidence": {
            field: sorted(
                candidates
            )
            for field, candidates
            in sorted(
                evidence.items()
            )
        },
    }


def _analyse_missing_isins(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Analyserar ISIN-lösa prisrader mot det centrala identitetslagret.

    Funktionen skriver inte till identitetslagret och ändrar inte
    prisraderna.

    Analysen skiljer mellan:

        unique_isin
            En entity och exakt ett instrument med ISIN gäller på datumet.

        multiple_current_instruments
            En entity är identifierad men flera instrument gäller samtidigt.

        current_instrument_without_isin
            Ett instrument gäller på datumet men identitetslagret saknar ISIN.

        entity_only
            Entityn kan identifieras men inget instrument kan kopplas
            entydigt till datumet.

        multiple_entities
            Identitetsuppgifterna pekar på flera möjliga entitys.

        unresolved
            Ingen central identitet kan hittas.

    Dessutom rapporteras historiska ISIN för identifierade entitys.
    """

    identity = load_identity()

    statistics = {
        "rows": 0,
        "unique_isin": 0,
        "multiple_current_instruments": 0,
        "current_instrument_without_isin": 0,
        "entity_only": 0,
        "multiple_entities": 0,
        "unresolved": 0,
        "historical_multiple_isins": 0,
    }

    examples: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    unique_candidate_isins: dict[
        str,
        int,
    ] = defaultdict(int)

    def add_example(
        status: str,
        record: dict[str, Any],
        details: dict[str, Any],
    ) -> None:
        if len(
            examples[status]
        ) >= 20:
            return

        examples[status].append(
            {
                "date": record.get(
                    "date"
                ),
                "yahoo_symbol": _normalise_symbol(
                    record.get(
                        "yahoo_symbol"
                    )
                ),
                "lei": _normalise_identity_value(
                    "lei",
                    record.get(
                        "lei"
                    ),
                ),
                "issuer": record.get(
                    "issuer"
                ),
                "ticker": _normalise_identity_value(
                    "ticker",
                    record.get(
                        "ticker"
                    ),
                ),
                **details,
                "source_file": record.get(
                    "_source_file"
                ),
                "source_line": record.get(
                    "_source_line"
                ),
            }
        )

    for record in records:
        if _normalise_isin(
            record.get("isin")
        ):
            continue

        statistics[
            "rows"
        ] += 1

        target_date = _parse_date(
            record.get("date")
        )

        matches = identity._matching_records(
            lei=record.get(
                "lei"
            ),
            issuer=record.get(
                "issuer"
            ),
            ticker=record.get(
                "ticker"
            ),
            yahoo_symbol=record.get(
                "yahoo_symbol"
            ),
            target_date=target_date,
        )

        if not matches:
            statistics[
                "unresolved"
            ] += 1

            add_example(
                "unresolved",
                record,
                {
                    "status": "unresolved",
                },
            )

            continue

        priority = {
            "isin": 0,
            "lei": 1,
            "yahoo_symbol": 2,
            "ticker": 3,
            "issuer": 4,
        }

        best_priority = min(
            priority[
                match.resolution
            ]
            for match in matches
        )

        best_matches = [
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
                for match in best_matches
            }
        )

        if len(
            entity_ids
        ) != 1:
            statistics[
                "multiple_entities"
            ] += 1

            add_example(
                "multiple_entities",
                record,
                {
                    "status": "multiple_entities",
                    "entities": entity_ids,
                    "matches": [
                        {
                            "entity_id": match.entity_id,
                            "resolution": match.resolution,
                            "isin": normalize_isin(
                                match.record.get(
                                    "isin"
                                )
                            ),
                            "issuer": match.record.get(
                                "issuer"
                            ),
                            "lei": match.record.get(
                                "lei"
                            ),
                            "ticker": match.record.get(
                                "ticker"
                            ),
                            "yahoo_symbol": match.record.get(
                                "yahoo_symbol"
                            ),
                        }
                        for match in best_matches
                    ],
                },
            )

            continue

        entity_id = entity_ids[0]

        all_instruments = (
            identity.instruments_for_entity(
                entity_id
            )
        )

        historical_isins = sorted(
            {
                normalize_isin(
                    instrument.get(
                        "isin"
                    )
                )
                for instrument
                in all_instruments
                if normalize_isin(
                    instrument.get(
                        "isin"
                    )
                )
            }
        )

        if len(
            historical_isins
        ) > 1:
            statistics[
                "historical_multiple_isins"
            ] += 1

        current_instruments = (
            identity.instruments_for_entity(
                entity_id,
                target_date,
            )
        )

        current_isins = sorted(
            {
                normalize_isin(
                    instrument.get(
                        "isin"
                    )
                )
                for instrument
                in current_instruments
                if normalize_isin(
                    instrument.get(
                        "isin"
                    )
                )
            }
        )

        instruments_without_isin = [
            instrument
            for instrument
            in current_instruments
            if not normalize_isin(
                instrument.get(
                    "isin"
                )
            )
        ]

        if len(
            current_isins
        ) == 1:
            candidate_isin = (
                current_isins[0]
            )

            statistics[
                "unique_isin"
            ] += 1

            unique_candidate_isins[
                candidate_isin
            ] += 1

            add_example(
                "unique_isin",
                record,
                {
                    "status": "unique_isin",
                    "entity_id": entity_id,
                    "candidate_isin": candidate_isin,
                    "current_instruments": [
                        {
                            "isin": normalize_isin(
                                instrument.get(
                                    "isin"
                                )
                            ),
                            "issuer": instrument.get(
                                "issuer"
                            ),
                            "lei": instrument.get(
                                "lei"
                            ),
                            "ticker": instrument.get(
                                "ticker"
                            ),
                            "yahoo_symbol": instrument.get(
                                "yahoo_symbol"
                            ),
                            "valid_from": instrument.get(
                                "valid_from"
                            ),
                            "valid_to": instrument.get(
                                "valid_to"
                            ),
                            "source": instrument.get(
                                "source"
                            ),
                            "status": instrument.get(
                                "status"
                            ),
                        }
                        for instrument
                        in current_instruments
                    ],
                    "historical_isins": historical_isins,
                    "historical_isin_count": len(
                        historical_isins
                    ),
                },
            )

            continue

        if len(
            current_isins
        ) > 1:
            statistics[
                "multiple_current_instruments"
            ] += 1

            add_example(
                "multiple_current_instruments",
                record,
                {
                    "status": "multiple_current_instruments",
                    "entity_id": entity_id,
                    "candidate_isins": current_isins,
                    "historical_isins": historical_isins,
                    "current_instruments": [
                        {
                            "isin": normalize_isin(
                                instrument.get(
                                    "isin"
                                )
                            ),
                            "issuer": instrument.get(
                                "issuer"
                            ),
                            "ticker": instrument.get(
                                "ticker"
                            ),
                            "yahoo_symbol": instrument.get(
                                "yahoo_symbol"
                            ),
                            "valid_from": instrument.get(
                                "valid_from"
                            ),
                            "valid_to": instrument.get(
                                "valid_to"
                            ),
                            "source": instrument.get(
                                "source"
                            ),
                            "status": instrument.get(
                                "status"
                            ),
                        }
                        for instrument
                        in current_instruments
                    ],
                },
            )

            continue

        if instruments_without_isin:
            statistics[
                "current_instrument_without_isin"
            ] += 1

            add_example(
                "current_instrument_without_isin",
                record,
                {
                    "status": (
                        "current_instrument_without_isin"
                    ),
                    "entity_id": entity_id,
                    "historical_isins": historical_isins,
                    "current_instruments": [
                        {
                            "isin": normalize_isin(
                                instrument.get(
                                    "isin"
                                )
                            ),
                            "issuer": instrument.get(
                                "issuer"
                            ),
                            "lei": instrument.get(
                                "lei"
                            ),
                            "ticker": instrument.get(
                                "ticker"
                            ),
                            "yahoo_symbol": instrument.get(
                                "yahoo_symbol"
                            ),
                            "valid_from": instrument.get(
                                "valid_from"
                            ),
                            "valid_to": instrument.get(
                                "valid_to"
                            ),
                            "source": instrument.get(
                                "source"
                            ),
                            "status": instrument.get(
                                "status"
                            ),
                        }
                        for instrument
                        in current_instruments
                    ],
                },
            )

            continue

        statistics[
            "entity_only"
        ] += 1

        add_example(
            "entity_only",
            record,
            {
                "status": "entity_only",
                "entity_id": entity_id,
                "historical_isins": historical_isins,
                "historical_instrument_count": len(
                    all_instruments
                ),
            },
        )

    return {
        "rows": statistics[
            "rows"
        ],
        "unique_isin": statistics[
            "unique_isin"
        ],
        "multiple_current_instruments": statistics[
            "multiple_current_instruments"
        ],
        "current_instrument_without_isin": statistics[
            "current_instrument_without_isin"
        ],
        "entity_only": statistics[
            "entity_only"
        ],
        "multiple_entities": statistics[
            "multiple_entities"
        ],
        "unresolved": statistics[
            "unresolved"
        ],
        "historical_multiple_isins": statistics[
            "historical_multiple_isins"
        ],
        "unique_candidate_isins": dict(
            sorted(
                unique_candidate_isins.items()
            )
        ),
        "examples": dict(
            examples
        ),
    }


def _analyse_isin_records(
    records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in records:
        key = _observation_key(
            record
        )

        if key is None:
            continue

        grouped[key].append(
            record
        )

    canonical: list[
        dict[str, Any]
    ] = []

    duplicate_groups = 0
    duplicate_rows = 0
    conflict_groups = 0
    conflict_rows = 0

    by_isin: dict[
        str,
        dict[str, int],
    ] = defaultdict(
        lambda: {
            "rows": 0,
            "unique_observations": 0,
            "duplicate_rows": 0,
            "conflict_groups": 0,
        }
    )

    conflicts: list[
        dict[str, Any]
    ] = []

    for (
        key,
        observations,
    ) in sorted(
        grouped.items()
    ):
        observation_date, isin = key

        stats = by_isin[
            isin
        ]

        stats[
            "rows"
        ] += len(
            observations
        )

        stats[
            "unique_observations"
        ] += 1

        first = observations[0]

        if len(
            observations
        ) == 1:
            canonical.append(
                _clean_output_record(
                    first
                )
            )
            continue

        duplicate_groups += 1

        duplicate_count = (
            len(observations) - 1
        )

        duplicate_rows += (
            duplicate_count
        )

        stats[
            "duplicate_rows"
        ] += duplicate_count

        has_conflict = any(
            _records_conflict(
                first,
                other,
            )
            for other in observations[
                1:
            ]
        )

        if has_conflict:
            conflict_groups += 1

            conflict_rows += (
                len(observations)
            )

            stats[
                "conflict_groups"
            ] += 1

            conflicts.append(
                {
                    "date": observation_date,
                    "isin": isin,
                    "observations": [
                        {
                            "close": observation.get(
                                "close"
                            ),
                            "yahoo_symbol": (
                                _normalise_symbol(
                                    observation.get(
                                        "yahoo_symbol"
                                    )
                                )
                            ),
                            "issuer": observation.get(
                                "issuer"
                            ),
                            "source_file": observation.get(
                                "_source_file"
                            ),
                            "source_line": observation.get(
                                "_source_line"
                            ),
                        }
                        for observation
                        in observations
                    ],
                }
            )

            continue

        canonical.append(
            _clean_output_record(
                first
            )
        )

    report = {
        "input": {
            "rows": len(
                records
            ),
        },
        "output": {
            "unique_observations": len(
                canonical
            ),
        },
        "deduplication": {
            "duplicate_groups": (
                duplicate_groups
            ),
            "duplicate_rows_removed": (
                duplicate_rows
            ),
        },
        "conflicts": {
            "groups": conflict_groups,
            "rows": conflict_rows,
            "details": conflicts,
        },
        "instruments": {
            "count": len(
                by_isin
            ),
            "by_isin": dict(
                sorted(
                    by_isin.items()
                )
            ),
        },
    }

    return (
        canonical,
        report,
    )


def _build_canonical_dataset(
    records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    records_with_isin = [
        record
        for record in records
        if _normalise_isin(
            record.get("isin")
        )
    ]

    records_without_isin = [
        record
        for record in records
        if not _normalise_isin(
            record.get("isin")
        )
    ]

    deduplicated_isin_records, report = (
        _analyse_isin_records(
            records_with_isin
        )
    )

    if report[
        "conflicts"
    ][
        "groups"
    ]:
        return (
            [],
            report,
        )

    canonical = (
        deduplicated_isin_records
        + [
            _clean_output_record(
                record
            )
            for record in records_without_isin
        ]
    )

    canonical.sort(
        key=lambda record: (
            record.get("date", ""),
            _normalise_isin(
                record.get("isin")
            ),
            _normalise_symbol(
                record.get("yahoo_symbol")
            ),
        )
    )

    report[
        "dataset"
    ] = {
        "input_rows": len(
            records
        ),
        "input_rows_with_isin": len(
            records_with_isin
        ),
        "input_rows_without_isin": len(
            records_without_isin
        ),
        "output_rows": len(
            canonical
        ),
        "output_rows_with_isin": len(
            deduplicated_isin_records
        ),
        "output_rows_without_isin": len(
            records_without_isin
        ),
    }

    return (
        canonical,
        report,
    )


def _output_path(
    records: list[dict[str, Any]],
) -> Path:
    dates = [
        _parse_date(
            record.get("date")
        )
        for record in records
    ]

    valid_dates = [
        value
        for value in dates
        if value is not None
    ]

    if not valid_dates:
        raise ValueError(
            "Det finns inga giltiga datum "
            "i canonical-datasetet."
        )

    first_date = min(
        valid_dates
    )

    last_date = max(
        valid_dates
    )

    return PRICE_DIR / (
        "prices_"
        f"{first_date.isoformat()}_"
        f"{last_date.isoformat()}_cleaned.jsonl"
    )


def _write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
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
                + "\n"
            )


def _write_report(
    path: Path,
    report: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            ensure_ascii=False,
            indent=2,
        )


def _archive_original_files(
    paths: list[Path],
) -> Path:
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    archive_path = (
        ARCHIVE_DIR
        / f"cleanup_{timestamp}"
    )

    archive_path.mkdir(
        parents=True,
        exist_ok=False,
    )

    for path in paths:
        shutil.move(
            str(path),
            str(
                archive_path
                / path.name
            ),
        )

    return archive_path


def _print_conflicts(
    conflicts: list[dict[str, Any]],
) -> None:
    if not conflicts:
        return

    print()
    print(
        "KONFLIKTER - FÖRSTA"
    )
    print(
        "-" * 70
    )

    for conflict in conflicts[:20]:
        print(
            f"{conflict['date']} "
            f"{conflict['isin']}"
        )

        for observation in conflict[
            "observations"
        ]:
            print(
                "  "
                f"close={observation['close']} "
                f"symbol={observation['yahoo_symbol']} "
                f"file={observation['source_file']} "
                f"line={observation['source_line']}"
            )


def _print_missing_isin_analysis(
    report: dict[str, Any],
) -> None:
    print()
    print(
        "CENTRAL IDENTITETSANALYS"
    )
    print(
        "-" * 70
    )

    print(
        "ISIN-lösa rader: "
        f"{report['rows']:,}"
        .replace(",", " ")
    )

    print(
        "Entydig ISIN för datumet: "
        f"{report['unique_isin']:,}"
        .replace(",", " ")
    )

    print(
        "Flera aktuella instrument: "
        f"{report['multiple_current_instruments']:,}"
        .replace(",", " ")
    )

    print(
        "Aktuellt instrument utan ISIN: "
        f"{report['current_instrument_without_isin']:,}"
        .replace(",", " ")
    )

    print(
        "Entity identifierad, inget instrument: "
        f"{report['entity_only']:,}"
        .replace(",", " ")
    )

    print(
        "Flera möjliga entitys: "
        f"{report['multiple_entities']:,}"
        .replace(",", " ")
    )

    print(
        "Ingen identifiering: "
        f"{report['unresolved']:,}"
        .replace(",", " ")
    )

    print(
        "Entity med flera historiska ISIN: "
        f"{report['historical_multiple_isins']:,}"
        .replace(",", " ")
    )

    print()
    print(
        "ENTYDIGA ISIN-KANDIDATER"
    )
    print(
        "-" * 70
    )

    for isin, count in sorted(
        report[
            "unique_candidate_isins"
        ].items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )[:50]:
        print(
            f"{isin}: "
            f"{count:,} rader"
            .replace(",", " ")
        )

    for status in (
        "unique_isin",
        "multiple_current_instruments",
        "current_instrument_without_isin",
        "entity_only",
        "multiple_entities",
        "unresolved",
    ):
        status_examples = report[
            "examples"
        ].get(
            status,
            [],
        )

        if not status_examples:
            continue

        print()
        print(
            f"EXEMPEL - {status.upper()}"
        )
        print(
            "-" * 70
        )

        for example in status_examples:
            print(
                f"{example['date']} "
                f"{example['yahoo_symbol']} "
                f"entity={example.get('entity_id', '-')}"
            )

            if example.get(
                "candidate_isin"
            ):
                print(
                    "  kandidat ISIN: "
                    f"{example['candidate_isin']}"
                )

            if example.get(
                "candidate_isins"
            ):
                print(
                    "  kandidat ISIN: "
                    f"{', '.join(example['candidate_isins'])}"
                )

            if example.get(
                "historical_isins"
            ):
                print(
                    "  historiska ISIN: "
                    f"{', '.join(example['historical_isins'])}"
                )

            if example.get(
                "entities"
            ):
                print(
                    "  entities: "
                    f"{', '.join(example['entities'])}"
                )


def _verify_output(
    path: Path,
    expected_rows: int,
) -> None:
    if not path.exists():
        raise RuntimeError(
            f"Canonical-filen skapades inte: {path}"
        )

    actual_rows = 0

    with path.open(
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
                raise RuntimeError(
                    "Ogiltig JSON i canonical-filen "
                    f"på rad {line_number}: {exc}"
                ) from exc

            if not isinstance(
                record,
                dict,
            ):
                raise RuntimeError(
                    "Canonical-filen innehåller "
                    f"ett ogiltigt objekt på rad {line_number}."
                )

            actual_rows += 1

    if actual_rows != expected_rows:
        raise RuntimeError(
            "Fel antal rader i canonical-filen: "
            f"förväntade {expected_rows}, "
            f"fick {actual_rows}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicerar price-filer på "
            "date + ISIN. "
            "ISIN-lösa rader lämnas orörda."
        )
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Arkivera originalfilerna och "
            "ersätt dem med ett canonical dataset."
        ),
    )

    args = parser.parse_args()

    print(
        "=" * 70
    )

    print(
        "PRICE CLEANUP"
    )

    print(
        "=" * 70
    )

    if args.apply:
        print(
            "MODE: APPLY"
        )
    else:
        print(
            "MODE: DRY-RUN"
        )

    paths = _load_price_files()

    print()
    print(
        "PRISFILER"
    )
    print(
        "-" * 70
    )

    for path in paths:
        print(
            f"  {path.name}"
        )

    if not paths:
        print(
            "Inga prices_*.jsonl hittades."
        )
        return

    records, load_statistics = (
        _load_records(
            paths
        )
    )

    canonical, report = (
        _build_canonical_dataset(
            records
        )
    )

    missing_isin_report = (
        _analyse_missing_isins(
            records
        )
    )

    report[
        "missing_isin_analysis"
    ] = missing_isin_report

    report[
        "load_statistics"
    ] = load_statistics

    print()
    print(
        "DATASET"
    )
    print(
        "-" * 70
    )

    print(
        "Filer: "
        f"{load_statistics['files']}"
    )

    print(
        "Rader: "
        f"{load_statistics['lines']:,}"
        .replace(",", " ")
    )

    print(
        "Giltiga prisrader: "
        f"{len(records):,}"
        .replace(",", " ")
    )

    print(
        "Med ISIN: "
        f"{len([r for r in records if _normalise_isin(r.get('isin'))]):,}"
        .replace(",", " ")
    )

    print(
        "Utan ISIN: "
        f"{len([r for r in records if not _normalise_isin(r.get('isin'))]):,}"
        .replace(",", " ")
    )

    print()
    print(
        "DEDUPE"
    )
    print(
        "-" * 70
    )

    print(
        "Unika date + ISIN: "
        f"{report['output']['unique_observations']:,}"
        .replace(",", " ")
    )

    print(
        "Dubblettgrupper: "
        f"{report['deduplication']['duplicate_groups']:,}"
        .replace(",", " ")
    )

    print(
        "Dubblettrader som tas bort: "
        f"{report['deduplication']['duplicate_rows_removed']:,}"
        .replace(",", " ")
    )

    print()
    print(
        "KONFLIKTER"
    )
    print(
        "-" * 70
    )

    print(
        "Konfliktgrupper: "
        f"{report['conflicts']['groups']:,}"
        .replace(",", " ")
    )

    print(
        "Konfliktrader: "
        f"{report['conflicts']['rows']:,}"
        .replace(",", " ")
    )

    _print_conflicts(
        report[
            "conflicts"
        ][
            "details"
        ]
    )

    print()
    print(
        "INVALIDA RADER"
    )
    print(
        "-" * 70
    )

    print(
        "Ogiltig JSON: "
        f"{load_statistics['invalid_json']}"
    )

    print(
        "Saknat datum: "
        f"{load_statistics['missing_date']}"
    )

    print(
        "Ogiltigt datum: "
        f"{load_statistics['invalid_date']}"
    )

    print(
        "Saknat ISIN: "
        f"{load_statistics['missing_isin']:,}"
        .replace(",", " ")
    )

    print(
        "Saknat close: "
        f"{load_statistics['missing_close']}"
    )

    print(
        "Ogiltigt close: "
        f"{load_statistics['invalid_close']}"
    )

    _print_missing_isin_analysis(
        missing_isin_report
    )

    print()
    print(
        "RESULTAT"
    )
    print(
        "-" * 70
    )

    if report[
        "conflicts"
    ][
        "groups"
    ]:
        print(
            "FAIL - close-konflikter hittades."
        )

        print(
            "Ingen fil ändras."
        )

        report[
            "apply"
        ] = {
            "requested": args.apply,
            "applied": False,
            "reason": "close_conflicts",
        }

        _write_report(
            REPORT_PATH,
            report,
        )

        raise SystemExit(1)

    if not canonical:
        print(
            "FAIL - canonical dataset blev tomt."
        )

        print(
            "Ingen fil ändras."
        )

        report[
            "apply"
        ] = {
            "requested": args.apply,
            "applied": False,
            "reason": "empty_output",
        }

        _write_report(
            REPORT_PATH,
            report,
        )

        raise SystemExit(1)

    output_path = _output_path(
        canonical
    )

    report[
        "output_path"
    ] = str(
        output_path
    )

    report[
        "output"
    ][
        "rows"
    ] = len(
        canonical
    )

    print(
        "Canonical dataset: "
        f"{len(canonical):,} rader"
        .replace(",", " ")
    )

    if not args.apply:
        dryrun_path = (
            Path("data/analysis")
            / "prices_canonical_dryrun.jsonl"
        )

        _write_jsonl(
            dryrun_path,
            canonical,
        )

        report[
            "apply"
        ] = {
            "requested": False,
            "applied": False,
            "reason": "dry_run",
        }

        report[
            "dryrun_output_path"
        ] = str(
            dryrun_path
        )

        _write_report(
            REPORT_PATH,
            report,
        )

        print()
        print(
            "DRY-RUN"
        )
        print(
            "-" * 70
        )

        print(
            f"Canonical testfil: "
            f"{dryrun_path}"
        )

        print(
            "Originalfilerna har INTE ändrats."
        )

        return

    print()
    print(
        "APPLY"
    )
    print(
        "-" * 70
    )

    temporary_output = (
        PRICE_DIR
        / f".{output_path.name}.tmp"
    )

    _write_jsonl(
        temporary_output,
        canonical,
    )

    _verify_output(
        temporary_output,
        len(canonical),
    )

    print(
        "Ny canonical-fil verifierad."
    )

    archive_path = _archive_original_files(
        paths
    )

    try:
        temporary_output.replace(
            output_path
        )
    except Exception:
        print()
        print(
            "FAIL - kunde inte installera "
            "den nya canonical-filen."
        )

        print(
            "Originalfilerna finns i arkivet:"
        )

        print(
            f"  {archive_path}"
        )

        raise

    report[
        "apply"
    ] = {
        "requested": True,
        "applied": True,
        "archive_path": str(
            archive_path
        ),
        "output_path": str(
            output_path
        ),
    }

    report[
        "archived_files"
    ] = [
        path.name
        for path in paths
    ]

    _write_report(
        REPORT_PATH,
        report,
    )

    print()
    print(
        "Originalfiler arkiverade:"
    )

    print(
        f"  {archive_path}"
    )

    print()
    print(
        "Ny canonical prisfil:"
    )

    print(
        f"  {output_path}"
    )

    print()
    print(
        "Dubblettstädningen är genomförd."
    )

    print()
    print(
        "Rapport:"
    )

    print(
        f"{REPORT_PATH}"
    )

    print()
    print(
        "=" * 70
    )

    print(
        "KLAR"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
