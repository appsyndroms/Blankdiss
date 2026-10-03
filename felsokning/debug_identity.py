from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from shared.instrument_identity import (
    InstrumentIdentity,
    normalize_identifier,
    normalize_isin,
    normalize_lei,
    normalize_text,
    parse_date,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

PRICE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "prices"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "analysis"
    / "identity_price_diagnostic.json"
)

SAMPLE_LIMIT = 50


def _parse_price_date(
    value: Any,
) -> date | None:
    return parse_date(
        value
    )


def _load_price_files() -> list[Path]:
    return sorted(
        path
        for path in PRICE_DIR.glob(
            "prices_*.jsonl"
        )
        if path.is_file()
    )


def _read_price_records() -> list[dict[str, Any]]:
    paths = _load_price_files()

    if not paths:
        raise FileNotFoundError(
            f"Inga price-filer hittades i: {PRICE_DIR}"
        )

    records: list[
        dict[str, Any]
    ] = []

    for path in paths:
        with path.open(
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
                    record = json.loads(
                        line
                    )

                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Ogiltig JSON på rad "
                        f"{line_number}: {path}"
                    ) from exc

                if not isinstance(
                    record,
                    dict,
                ):
                    continue

                clean_record = dict(
                    record
                )

                clean_record[
                    "_source_file"
                ] = path.name

                clean_record[
                    "_source_line"
                ] = line_number

                records.append(
                    clean_record
                )

    return records


def _value(
    record: dict[str, Any],
    *keys: str,
) -> Any:
    for key in keys:
        value = record.get(
            key
        )

        if value is not None:
            return value

    return None


def _identity_values(
    record: dict[str, Any],
) -> dict[str, str]:
    return {
        "isin": normalize_isin(
            _value(
                record,
                "isin",
            )
        ),
        "lei": normalize_lei(
            _value(
                record,
                "lei",
                "LEI",
            )
        ),
        "issuer": normalize_text(
            _value(
                record,
                "issuer",
            )
        ),
        "ticker": normalize_identifier(
            _value(
                record,
                "ticker",
            )
        ),
        "yahoo_symbol": normalize_identifier(
            _value(
                record,
                "yahoo_symbol",
            )
        ),
    }


def _has_identity(
    values: dict[str, str],
) -> bool:
    return any(
        values[field]
        for field in (
            "isin",
            "lei",
            "issuer",
            "ticker",
            "yahoo_symbol",
        )
    )


def _instrument_isin(
    record: dict[str, Any],
) -> str:
    return normalize_isin(
        record.get("isin")
    )


def _instrument_key(
    record: dict[str, Any],
) -> str:
    isin = _instrument_isin(
        record
    )

    if isin:
        return isin

    return (
        "NO_ISIN:"
        + normalize_text(
            record.get(
                "issuer"
            )
        )
    )


def _instrument_description(
    record: dict[str, Any],
) -> str:
    isin = _instrument_isin(
        record
    )

    issuer = normalize_text(
        record.get(
            "issuer"
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

    parts: list[str] = []

    if isin:
        parts.append(
            f"ISIN={isin}"
        )

    if issuer:
        parts.append(
            f"issuer={issuer}"
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


def _unique_instruments(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: dict[
        str,
        dict[str, Any],
    ] = {}

    for record in records:
        result.setdefault(
            _instrument_key(
                record
            ),
            record,
        )

    return list(
        result.values()
    )


def _classify_record(
    identity: InstrumentIdentity,
    record: dict[str, Any],
) -> dict[str, Any]:
    values = _identity_values(
        record
    )

    target_date = _parse_price_date(
        record.get("date")
    )

    base_result: dict[
        str,
        Any,
    ] = {
        "date": (
            target_date.isoformat()
            if target_date
            else None
        ),
        "yahoo_symbol": values[
            "yahoo_symbol"
        ],
        "lei": values[
            "lei"
        ],
        "issuer": values[
            "issuer"
        ],
        "ticker": values[
            "ticker"
        ],
        "source_file": record.get(
            "_source_file"
        ),
        "source_line": record.get(
            "_source_line"
        ),
    }

    if not target_date:
        base_result[
            "status"
        ] = "invalid_date"

        return base_result

    if not _has_identity(
        values
    ):
        base_result[
            "status"
        ] = "unresolved"

        return base_result

    match = identity.resolve(
        isin=values[
            "isin"
        ] or None,
        lei=values[
            "lei"
        ] or None,
        issuer=values[
            "issuer"
        ] or None,
        ticker=values[
            "ticker"
        ] or None,
        yahoo_symbol=values[
            "yahoo_symbol"
        ] or None,
        target_date=target_date,
    )

    if match is None:
        base_result[
            "status"
        ] = "unresolved"

        return base_result

    entity_id = match.entity_id

    base_result[
        "entity_id"
    ] = entity_id

    base_result[
        "resolution"
    ] = match.resolution

    historical = _unique_instruments(
        identity.instruments_for_entity(
            entity_id
        )
    )

    current = _unique_instruments(
        identity.instruments_for_entity(
            entity_id,
            target_date,
        )
    )

    historical_isins = sorted(
        {
            _instrument_isin(
                instrument
            )
            for instrument in historical
            if _instrument_isin(
                instrument
            )
        }
    )

    current_isins = sorted(
        {
            _instrument_isin(
                instrument
            )
            for instrument in current
            if _instrument_isin(
                instrument
            )
        }
    )

    base_result[
        "historical_instruments"
    ] = [
        _instrument_description(
            instrument
        )
        for instrument in historical
    ]

    base_result[
        "historical_isins"
    ] = historical_isins

    base_result[
        "current_instruments"
    ] = [
        _instrument_description(
            instrument
        )
        for instrument in current
    ]

    base_result[
        "current_isins"
    ] = current_isins

    if not current:
        base_result[
            "status"
        ] = "entity_only"

    elif len(current) > 1:
        base_result[
            "status"
        ] = "multiple_current_instruments"

    elif len(current) == 1:
        current_instrument = current[
            0
        ]

        current_isin = _instrument_isin(
            current_instrument
        )

        if not current_isin:
            base_result[
                "status"
            ] = "instrument_without_isin"

        elif len(
            historical_isins
        ) > 1:
            base_result[
                "status"
            ] = "multiple_historical_isins"

        else:
            base_result[
                "status"
            ] = "unique_isin"

    return base_result


def _print_samples(
    title: str,
    values: list[dict[str, Any]],
) -> None:
    print()
    print(title)
    print("-" * 70)

    if not values:
        print("Inga.")

        return

    for value in values[
        :SAMPLE_LIMIT
    ]:
        date_value = (
            value.get(
                "date"
            )
            or "-"
        )

        symbol = (
            value.get(
                "yahoo_symbol"
            )
            or "-"
        )

        entity = (
            value.get(
                "entity_id"
            )
            or "-"
        )

        resolution = (
            value.get(
                "resolution"
            )
            or "-"
        )

        current_isins = ",".join(
            value.get(
                "current_isins",
                [],
            )
        ) or "-"

        historical_isins = ",".join(
            value.get(
                "historical_isins",
                [],
            )
        ) or "-"

        print(
            f"{date_value} | "
            f"{symbol} | "
            f"entity={entity} | "
            f"resolution={resolution} | "
            f"current={current_isins} | "
            f"historical={historical_isins}"
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


def _print_status_summary(
    counts: dict[str, int],
) -> None:
    print()
    print("=" * 70)
    print(
        "IDENTITETSSTATUS"
    )
    print("=" * 70)

    statuses = (
        "unique_isin",
        "multiple_historical_isins",
        "multiple_current_instruments",
        "entity_only",
        "instrument_without_isin",
        "unresolved",
        "invalid_date",
    )

    for status in statuses:
        print(
            f"{status:30}: "
            f"{counts[status]:,}"
        )


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
    print("=" * 70)
    print(
        "PRICE / IDENTITY FELSÖKNING"
    )
    print("=" * 70)

    identity = InstrumentIdentity()

    print()
    print(
        "BEFINTLIGT IDENTITY-REGISTER"
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

    paths = _load_price_files()

    print()
    print(
        "PRISFILER"
    )
    print("-" * 70)

    for path in paths:
        print(
            f"  {path.name}"
        )

    if not paths:
        raise SystemExit(
            "Inga price-filer hittades."
        )

    records = _read_price_records()

    print()
    print(
        "PRISKÄLLA"
    )
    print("-" * 70)

    print(
        f"Rader: "
        f"{len(records):,}"
    )

    missing_isin = [
        record
        for record in records
        if not normalize_isin(
            record.get("isin")
        )
    ]

    print(
        f"ISIN-lösa rader: "
        f"{len(missing_isin):,}"
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

    for record in missing_isin:
        result = _classify_record(
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

    _print_status_summary(
        counts
    )

    _print_samples(
        "UNIK ISIN",
        examples[
            "unique_isin"
        ],
    )

    _print_samples(
        "FLERA HISTORISKA ISIN",
        examples[
            "multiple_historical_isins"
        ],
    )

    _print_samples(
        "FLERA AKTUELLA INSTRUMENT",
        examples[
            "multiple_current_instruments"
        ],
    )

    _print_samples(
        "ENTITY HITTAD – MEN INGET INSTRUMENT",
        examples[
            "entity_only"
        ],
    )

    _print_samples(
        "INSTRUMENT HITTAT – MEN UTAN ISIN",
        examples[
            "instrument_without_isin"
        ],
    )

    _print_samples(
        "EJ LÖST",
        examples[
            "unresolved"
        ],
    )

    _print_samples(
        "OGILTIGT DATUM",
        examples[
            "invalid_date"
        ],
    )

    report = {
        "source": {
            "price_dir": str(
                PRICE_DIR
            ),
            "files": [
                path.name
                for path in paths
            ],
            "rows": len(
                records
            ),
            "rows_without_isin": len(
                missing_isin
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
            "rows": len(
                results
            ),
            "counts": dict(
                sorted(
                    counts.items()
                )
            ),
        },
        "examples": {
            status: values
            for status, values
            in sorted(
                examples.items()
            )
        },
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
        "Inga prisrader har ändrats."
    )

    print(
        "instrument_aliases.jsonl har inte ändrats."
    )

    print()
    print(
        f"Rapport: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
