from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any


PRICE_DIR = Path(
    "data/raw/prices"
)

OUTPUT_PATH = PRICE_DIR / (
    "prices_deduplicated_dryrun.jsonl"
)

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
        if path.name
        != OUTPUT_PATH.name
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
    identity_index = _build_identity_index(
        records
    )

    statistics = {
        "rows": 0,
        "unique": 0,
        "ambiguous": 0,
        "conflict": 0,
        "unresolved": 0,
        "candidate_isins": 0,
    }

    by_status: dict[
        str,
        dict[str, int],
    ] = defaultdict(
        lambda: {
            "rows": 0,
            "unique_observations": 0,
        }
    )

    unique_candidates: dict[
        str,
        int,
    ] = defaultdict(int)

    ambiguous_candidates: dict[
        str,
        int,
    ] = defaultdict(int)

    conflict_examples: list[
        dict[str, Any]
    ] = []

    ambiguous_examples: list[
        dict[str, Any]
    ] = []

    unresolved_examples: list[
        dict[str, Any]
    ] = []

    seen_observations: dict[
        str,
        set[
            tuple[
                str,
                str,
            ]
        ],
    ] = defaultdict(set)

    for record in records:
        if _normalise_isin(
            record.get("isin")
        ):
            continue

        statistics[
            "rows"
        ] += 1

        resolution = _resolve_missing_isin(
            record,
            identity_index,
        )

        status = resolution[
            "status"
        ]

        statistics[
            status
        ] += 1

        by_status[
            status
        ][
            "rows"
        ] += 1

        observation_key = (
            record.get("date", ""),
            _normalise_identity_value(
                "yahoo_symbol",
                record.get(
                    "yahoo_symbol"
                ),
            ),
        )

        seen_observations[
            status
        ].add(
            observation_key
        )

        candidate_isins = resolution[
            "candidate_isins"
        ]

        if status == "unique":
            candidate_isin = (
                candidate_isins[0]
            )

            unique_candidates[
                candidate_isin
            ] += 1

            statistics[
                "candidate_isins"
            ] += 1

        elif status == "ambiguous":
            for isin in candidate_isins:
                ambiguous_candidates[
                    isin
                ] += 1

            if len(
                ambiguous_examples
            ) < 20:
                ambiguous_examples.append(
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
                        "candidate_isins": candidate_isins,
                        "evidence": resolution[
                            "evidence"
                        ],
                        "source_file": record.get(
                            "_source_file"
                        ),
                        "source_line": record.get(
                            "_source_line"
                        ),
                    }
                )

        elif status == "conflict":
            if len(
                conflict_examples
            ) < 20:
                conflict_examples.append(
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
                        "candidate_isins": candidate_isins,
                        "evidence": resolution[
                            "evidence"
                        ],
                        "source_file": record.get(
                            "_source_file"
                        ),
                        "source_line": record.get(
                            "_source_line"
                        ),
                    }
                )

        elif status == "unresolved":
            if len(
                unresolved_examples
            ) < 20:
                unresolved_examples.append(
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
                        "source_file": record.get(
                            "_source_file"
                        ),
                        "source_line": record.get(
                            "_source_line"
                        ),
                    }
                )

    for status in (
        "unique",
        "ambiguous",
        "conflict",
        "unresolved",
    ):
        by_status[
            status
        ][
            "unique_observations"
        ] = len(
            seen_observations[
                status
            ]
        )

    return {
        "rows": statistics[
            "rows"
        ],
        "unique": statistics[
            "unique"
        ],
        "ambiguous": statistics[
            "ambiguous"
        ],
        "conflict": statistics[
            "conflict"
        ],
        "unresolved": statistics[
            "unresolved"
        ],
        "unique_candidate_rows": statistics[
            "candidate_isins"
        ],
        "by_status": dict(
            by_status
        ),
        "unique_candidate_isins": dict(
            sorted(
                unique_candidates.items()
            )
        ),
        "ambiguous_candidate_isins": dict(
            sorted(
                ambiguous_candidates.items()
            )
        ),
        "examples": {
            "ambiguous": ambiguous_examples,
            "conflict": conflict_examples,
            "unresolved": unresolved_examples,
        },
    }


def _analyse_records(
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
            "files": len(
                {
                    record[
                        "_source_file"
                    ]
                    for record in records
                }
            ),
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


def main() -> None:
    print(
        "=" * 70
    )
    print(
        "PRICE CLEANUP / DRY-RUN"
    )
    print(
        "=" * 70
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

    canonical, report = (
        _analyse_records(
            records_with_isin
        )
    )

    missing_isin_report = (
        _analyse_missing_isins(
            records
        )
    )

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
        f"{len(records_with_isin):,}"
        .replace(",", " ")
    )

    print(
        "Utan ISIN: "
        f"{len(records_without_isin):,}"
        .replace(",", " ")
    )

    print(
        "Instrument/ISIN: "
        f"{report['instruments']['count']}"
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
        "Dubblettrader: "
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

    if report[
        "conflicts"
    ][
        "groups"
    ]:
        print()
        print(
            "KONFLIKTER - FÖRSTA"
        )
        print(
            "-" * 70
        )

        for conflict in report[
            "conflicts"
        ][
            "details"
        ][:20]:
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
        f"{load_statistics['missing_isin']}"
    )

    print(
        "Saknat close: "
        f"{load_statistics['missing_close']}"
    )

    print(
        "Ogiltigt close: "
        f"{load_statistics['invalid_close']}"
    )

    print()
    print(
        "ISIN-ANALYS"
    )
    print(
        "-" * 70
    )

    print(
        "ISIN-lösa rader: "
        f"{missing_isin_report['rows']:,}"
        .replace(",", " ")
    )

    print(
        "Entydig ISIN-kandidat: "
        f"{missing_isin_report['unique']:,}"
        .replace(",", " ")
    )

    print(
        "Flera möjliga ISIN: "
        f"{missing_isin_report['ambiguous']:,}"
        .replace(",", " ")
    )

    print(
        "Motstridiga identiteter: "
        f"{missing_isin_report['conflict']:,}"
        .replace(",", " ")
    )

    print(
        "Ingen identifiering: "
        f"{missing_isin_report['unresolved']:,}"
        .replace(",", " ")
    )

    print()
    print(
        "ISIN-ANALYS - UNIKA KANDIDATER"
    )
    print(
        "-" * 70
    )

    unique_candidates = (
        missing_isin_report[
            "unique_candidate_isins"
        ]
    )

    for isin, count in sorted(
        unique_candidates.items(),
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

    if missing_isin_report[
        "ambiguous"
    ]:
        print()
        print(
            "ISIN-ANALYS - FLERA MÖJLIGA"
        )
        print(
            "-" * 70
        )

        for example in (
            missing_isin_report[
                "examples"
            ][
                "ambiguous"
            ]
        ):
            print(
                f"{example['date']} "
                f"symbol={example['yahoo_symbol']} "
                f"issuer={example['issuer']}"
            )

            print(
                "  Kandidater: "
                + ", ".join(
                    example[
                        "candidate_isins"
                    ]
                )
            )

            print(
                "  Evidence: "
                + json.dumps(
                    example[
                        "evidence"
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

            print(
                "  Källa: "
                f"{example['source_file']}:"
                f"{example['source_line']}"
            )

    if missing_isin_report[
        "conflict"
    ]:
        print()
        print(
            "ISIN-ANALYS - MOTSTRIDIGA"
        )
        print(
            "-" * 70
        )

        for example in (
            missing_isin_report[
                "examples"
            ][
                "conflict"
            ]
        ):
            print(
                f"{example['date']} "
                f"symbol={example['yahoo_symbol']} "
                f"issuer={example['issuer']}"
            )

            print(
                "  Kandidater: "
                + ", ".join(
                    example[
                        "candidate_isins"
                    ]
                )
            )

            print(
                "  Evidence: "
                + json.dumps(
                    example[
                        "evidence"
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

            print(
                "  Källa: "
                f"{example['source_file']}:"
                f"{example['source_line']}"
            )

    if missing_isin_report[
        "unresolved"
    ]:
        print()
        print(
            "ISIN-ANALYS - EJ IDENTIFIERADE"
        )
        print(
            "-" * 70
        )

        for example in (
            missing_isin_report[
                "examples"
            ][
                "unresolved"
            ]
        ):
            print(
                f"{example['date']} "
                f"symbol={example['yahoo_symbol']} "
                f"issuer={example['issuer']} "
                f"lei={example['lei']} "
                f"ticker={example['ticker']}"
            )

            print(
                "  Källa: "
                f"{example['source_file']}:"
                f"{example['source_line']}"
            )

    print()
    print(
        "PER INSTRUMENT"
    )
    print(
        "-" * 70
    )

    for isin, statistics in report[
        "instruments"
    ][
        "by_isin"
    ].items():
        print(
            f"{isin}: "
            f"{statistics['rows']:,} rader, "
            f"{statistics['unique_observations']:,} unika, "
            f"{statistics['duplicate_rows']:,} dubbletter, "
            f"{statistics['conflict_groups']} konflikter"
            .replace(",", " ")
        )

    report[
        "missing_isin_analysis"
    ] = missing_isin_report

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
            "FAIL - konflikter hittades."
        )
        print(
            "Ingen konfliktobservation tas bort."
        )
        print(
            "Ingen slutlig fil ska användas."
        )
    else:
        print(
            "PASS - inga close-konflikter."
        )
        print(
            "Dubbletter med ISIN kan reduceras "
            "säkert till en rad per date + ISIN."
        )

        _write_jsonl(
            OUTPUT_PATH,
            canonical,
        )

        print()
        print(
            "DRY-RUN OUTPUT"
        )
        print(
            "-" * 70
        )

        print(
            f"Föreslagen fil: "
            f"{OUTPUT_PATH}"
        )

        print(
            "Originalfilerna har INTE ändrats."
        )

    report[
        "load_statistics"
    ] = load_statistics

    report[
        "output_path"
    ] = str(
        OUTPUT_PATH
    )

    report[
        "dry_run"
    ] = True

    _write_report(
        REPORT_PATH,
        report,
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
