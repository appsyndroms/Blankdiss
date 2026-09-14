"""
Kvalitetskontroll av FI:s rekonstruerade aggregatserie.

VIKTIGT:
Resultatet från reconstruct_aggregate.py är en rekonstruerad serie
baserad på publicerade individuella positioner. Den är inte FI:s
officiella aggregatserie.

Den här modulen försöker hitta ekonomiskt eller strukturellt
misstänkta observationer utan att behandla normala konsekvenser
av FI:s publiceringsgräns som databasfel.

QC:n är avsiktligt diagnostisk:
- misstänkta observationer rapporteras
- workflowet failar inte bara för att en observation är ovanlig

Output:
data/processed/fi/aggregate/reconstructed_qc.json
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "fi"
    / "aggregate"
    / "reconstructed.jsonl"
)

OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "fi"
    / "aggregate"
    / "reconstructed_qc.json"
)

# Trösklarna är avsiktligt ganska höga.
# Målet är att hitta kandidater för granskning,
# inte att flagga normala marknadsrörelser.
ABSOLUTE_JUMP_PP = 3.0
RELATIVE_JUMP_PERCENT = 100.0
RELATIVE_JUMP_MIN_PREVIOUS = 0.5

HOLDER_CHANGE_ABSOLUTE = 5
HOLDER_CHANGE_RELATIVE_PERCENT = 100.0

MARKET_ABSOLUTE_CHANGE_PP = 1.0
MARKET_SHARE_THRESHOLD = 0.20

MAX_EXAMPLES_PER_CATEGORY = 200


def finite_float(value: object) -> float | None:
    """Returnerar ett ändligt float-värde eller None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def finite_int(value: object) -> int | None:
    """Returnerar ett int-värde eller None."""

    try:
        number = int(value)
    except (TypeError, ValueError):
        return None

    return number


def add_example(
    target: list[dict],
    value: dict,
) -> None:
    """Lägger till ett exempel upp till maxgränsen."""

    if len(target) < MAX_EXAMPLES_PER_CATEGORY:
        target.append(value)


def load_metadata() -> dict:
    """Läser rekonstruktionsmetadata om den finns."""

    metadata_path = (
        ROOT
        / "data"
        / "processed"
        / "fi"
        / "aggregate"
        / "reconstructed_metadata.json"
    )

    if not metadata_path.exists():
        return {}

    try:
        return json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        return {}


def run_qc() -> dict:
    """Kör hela QC-analysen."""

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Saknar rekonstruerad FI-serie: {INPUT}"
        )

    metadata = load_metadata()

    required_columns = {
        "snapshot_date",
        "issuer",
        "isin",
        "short_interest_pct",
        "active_holders",
        "max_individual_position_pct",
        "max_position_share_pct",
    }

    rows = 0
    invalid_rows = 0

    snapshot_dates: set[str] = set()
    issuers: set[str] = set()
    isins: set[str] = set()

    # Föregående observation per issuer.
    previous_by_issuer: dict[str, dict] = {}

    # Föregående observation per ISIN.
    previous_by_isin: dict[str, dict] = {}

    # ISIN -> issuer-namn.
    isin_to_issuers: dict[str, set[str]] = defaultdict(set)

    # issuer -> ISIN.
    issuer_to_isins: dict[str, set[str]] = defaultdict(set)

    # Statistik per datum.
    market_date_stats: dict[str, dict] = defaultdict(
        lambda: {
            "observations": 0,
            "absolute_changes_ge_1pp": 0,
            "absolute_changes_ge_3pp": 0,
            "absolute_change_sum": 0.0,
        }
    )

    # Kandidater.
    large_jumps: list[dict] = []
    relative_jumps: list[dict] = []
    holder_changes: list[dict] = []
    concentration_changes: list[dict] = []
    invalid_values: list[dict] = []
    issuer_isin_conflicts: list[dict] = []
    isin_issuer_conflicts: list[dict] = []
    market_wide_changes: list[dict] = []

    first_date: str | None = None
    last_date: str | None = None

    with INPUT.open(
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
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                add_example(
                    invalid_values,
                    {
                        "line": line_number,
                        "reason": "invalid_json",
                    },
                )
                continue

            if not required_columns.issubset(row):
                invalid_rows += 1
                add_example(
                    invalid_values,
                    {
                        "line": line_number,
                        "reason": "missing_columns",
                        "missing": sorted(
                            required_columns
                            - set(row)
                        ),
                    },
                )
                continue

            snapshot_date = str(
                row["snapshot_date"]
            ).strip()

            issuer = str(
                row["issuer"]
            ).strip()

            isin_value = row.get("isin")

            if isin_value is None:
                isin = ""
            else:
                isin = str(
                    isin_value
                ).strip()

            short_interest = finite_float(
                row["short_interest_pct"]
            )

            active_holders = finite_int(
                row["active_holders"]
            )

            max_position = finite_float(
                row[
                    "max_individual_position_pct"
                ]
            )

            concentration = finite_float(
                row[
                    "max_position_share_pct"
                ]
            )

            if (
                not snapshot_date
                or not issuer
                or short_interest is None
                or active_holders is None
                or max_position is None
                or concentration is None
            ):
                invalid_rows += 1

                add_example(
                    invalid_values,
                    {
                        "line": line_number,
                        "snapshot_date": snapshot_date,
                        "issuer": issuer,
                        "reason": (
                            "invalid_required_value"
                        ),
                    },
                )
                continue

            rows += 1

            snapshot_dates.add(
                snapshot_date
            )
            issuers.add(issuer)

            if isin:
                isins.add(isin)

                isin_to_issuers[
                    isin
                ].add(issuer)

                issuer_to_isins[
                    issuer
                ].add(isin)

            if first_date is None:
                first_date = snapshot_date

            last_date = snapshot_date

            current = {
                "snapshot_date": snapshot_date,
                "issuer": issuer,
                "isin": isin or None,
                "short_interest_pct": (
                    short_interest
                ),
                "active_holders": (
                    active_holders
                ),
                "max_individual_position_pct": (
                    max_position
                ),
                "max_position_share_pct": (
                    concentration
                ),
            }

            previous = previous_by_issuer.get(
                issuer
            )

            if previous is not None:
                previous_date = pd.Timestamp(
                    previous["snapshot_date"]
                )
                current_date = pd.Timestamp(
                    snapshot_date
                )

                delta_days = (
                    current_date
                    - previous_date
                ).days

                previous_short = previous[
                    "short_interest_pct"
                ]

                short_delta = (
                    short_interest
                    - previous_short
                )

                absolute_delta = abs(
                    short_delta
                )

                market_stats = (
                    market_date_stats[
                        snapshot_date
                    ]
                )

                market_stats[
                    "observations"
                ] += 1

                market_stats[
                    "absolute_change_sum"
                ] += absolute_delta

                if (
                    absolute_delta
                    >= MARKET_ABSOLUTE_CHANGE_PP
                ):
                    market_stats[
                        "absolute_changes_ge_1pp"
                    ] += 1

                if (
                    absolute_delta
                    >= ABSOLUTE_JUMP_PP
                ):
                    market_stats[
                        "absolute_changes_ge_3pp"
                    ] += 1

                if (
                    absolute_delta
                    >= ABSOLUTE_JUMP_PP
                ):
                    add_example(
                        large_jumps,
                        {
                            **current,
                            "previous_snapshot_date": (
                                previous[
                                    "snapshot_date"
                                ]
                            ),
                            "previous_short_interest_pct": (
                                previous_short
                            ),
                            "delta_pp": round(
                                short_delta,
                                6,
                            ),
                            "absolute_delta_pp": round(
                                absolute_delta,
                                6,
                            ),
                            "gap_days": delta_days,
                            "reason": (
                                "large_absolute_jump"
                            ),
                        },
                    )

                if (
                    previous_short
                    >= RELATIVE_JUMP_MIN_PREVIOUS
                ):
                    relative_change = (
                        absolute_delta
                        / previous_short
                        * 100.0
                    )

                    if (
                        relative_change
                        >= RELATIVE_JUMP_PERCENT
                    ):
                        add_example(
                            relative_jumps,
                            {
                                **current,
                                "previous_snapshot_date": (
                                    previous[
                                        "snapshot_date"
                                    ]
                                ),
                                "previous_short_interest_pct": (
                                    previous_short
                                ),
                                "delta_pp": round(
                                    short_delta,
                                    6,
                                ),
                                "relative_change_percent": round(
                                    (
                                        short_delta
                                        / previous_short
                                        * 100.0
                                    ),
                                    6,
                                ),
                                "gap_days": (
                                    delta_days
                                ),
                                "reason": (
                                    "large_relative_jump"
                                ),
                            },
                        )

                previous_holders = previous[
                    "active_holders"
                ]

                holder_delta = (
                    active_holders
                    - previous_holders
                )

                holder_absolute = abs(
                    holder_delta
                )

                holder_relative = (
                    (
                        holder_absolute
                        / previous_holders
                        * 100.0
                    )
                    if previous_holders > 0
                    else (
                        100.0
                        if active_holders > 0
                        else 0.0
                    )
                )

                if (
                    holder_absolute
                    >= HOLDER_CHANGE_ABSOLUTE
                    or (
                        holder_relative
                        >= HOLDER_CHANGE_RELATIVE_PERCENT
                        and (
                            previous_holders
                            > 0
                        )
                    )
                ):
                    add_example(
                        holder_changes,
                        {
                            **current,
                            "previous_snapshot_date": (
                                previous[
                                    "snapshot_date"
                                ]
                            ),
                            "previous_active_holders": (
                                previous_holders
                            ),
                            "holder_delta": (
                                holder_delta
                            ),
                            "holder_relative_change_percent": (
                                round(
                                    holder_relative,
                                    6,
                                )
                            ),
                            "gap_days": (
                                delta_days
                            ),
                            "reason": (
                                "large_holder_count_change"
                            ),
                        },
                    )

                previous_concentration = (
                    previous[
                        "max_position_share_pct"
                    ]
                )

                concentration_delta = (
                    concentration
                    - previous_concentration
                )

                # Stor förändring i koncentration är inte
                # automatiskt fel. Den kan däremot vara
                # mycket informativ när vi senare analyserar
                # blankarnas beteende.
                if abs(
                    concentration_delta
                ) >= 25.0:
                    add_example(
                        concentration_changes,
                        {
                            **current,
                            "previous_snapshot_date": (
                                previous[
                                    "snapshot_date"
                                ]
                            ),
                            "previous_max_position_share_pct": (
                                previous_concentration
                            ),
                            "concentration_delta_pp": round(
                                concentration_delta,
                                6,
                            ),
                            "gap_days": (
                                delta_days
                            ),
                            "reason": (
                                "large_concentration_change"
                            ),
                        },
                    )

            previous_by_issuer[
                issuer
            ] = current

            if isin:
                previous_by_isin[
                    isin
                ] = current

    # ISIN/issuer-mappningar.
    for isin, names in sorted(
        isin_to_issuers.items()
    ):
        if len(names) > 1:
            add_example(
                isin_issuer_conflicts,
                {
                    "isin": isin,
                    "issuers": sorted(names),
                    "reason": (
                        "same_isin_multiple_issuers"
                    ),
                },
            )

    for issuer, issuer_isins in sorted(
        issuer_to_isins.items()
    ):
        if len(issuer_isins) > 1:
            add_example(
                issuer_isin_conflicts,
                {
                    "issuer": issuer,
                    "isins": sorted(
                        issuer_isins
                    ),
                    "reason": (
                        "same_issuer_multiple_isins"
                    ),
                },
            )

    # Marknadsövergripande förändringar.
    for snapshot_date, stats in sorted(
        market_date_stats.items()
    ):
        observations = stats[
            "observations"
        ]

        if observations == 0:
            continue

        share_ge_1pp = (
            stats[
                "absolute_changes_ge_1pp"
            ]
            / observations
        )

        share_ge_3pp = (
            stats[
                "absolute_changes_ge_3pp"
            ]
            / observations
        )

        if share_ge_1pp >= MARKET_SHARE_THRESHOLD:
            add_example(
                market_wide_changes,
                {
                    "snapshot_date": (
                        snapshot_date
                    ),
                    "observations_with_previous": (
                        observations
                    ),
                    "changes_ge_1pp": (
                        stats[
                            "absolute_changes_ge_1pp"
                        ]
                    ),
                    "changes_ge_3pp": (
                        stats[
                            "absolute_changes_ge_3pp"
                        ]
                    ),
                    "share_ge_1pp": round(
                        share_ge_1pp,
                        6,
                    ),
                    "share_ge_3pp": round(
                        share_ge_3pp,
                        6,
                    ),
                    "mean_absolute_change_pp": round(
                        (
                            stats[
                                "absolute_change_sum"
                            ]
                            / observations
                        ),
                        6,
                    ),
                    "reason": (
                        "market_wide_change"
                    ),
                },
            )

    # Kontrollera monotoni i den förväntade sorteringen.
    # Rekonstruktionen produceras sorterad på snapshot_date,
    # issuer och ISIN. Här behöver vi bara dokumentera
    # eventuella avvikelser; filen bör normalt vara sorterad.
    date_order_ok = True
    previous_date_value: pd.Timestamp | None = None

    with INPUT.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
                current_date = pd.Timestamp(
                    row["snapshot_date"]
                )
            except (
                json.JSONDecodeError,
                KeyError,
                ValueError,
                TypeError,
            ):
                continue

            if (
                previous_date_value is not None
                and current_date
                < previous_date_value
            ):
                date_order_ok = False
                break

            previous_date_value = current_date

    category_counts = {
        "large_absolute_jumps": len(
            large_jumps
        ),
        "large_relative_jumps": len(
            relative_jumps
        ),
        "large_holder_count_changes": len(
            holder_changes
        ),
        "large_concentration_changes": len(
            concentration_changes
        ),
        "issuer_isin_conflicts": len(
            issuer_isin_conflicts
        ),
        "isin_issuer_conflicts": len(
            isin_issuer_conflicts
        ),
        "market_wide_changes": len(
            market_wide_changes
        ),
        "invalid_rows": invalid_rows,
    }

    suspicious_count = sum(
        category_counts.values()
    )

    result = {
        "dataset": (
            "reconstructed_fi_aggregate"
        ),
        "input_file": str(
            INPUT.relative_to(ROOT)
        ),
        "output_file": str(
            OUTPUT.relative_to(ROOT)
        ),
        "method": (
            "economic_and_structural_qc"
        ),
        "thresholds": {
            "absolute_jump_pp": (
                ABSOLUTE_JUMP_PP
            ),
            "relative_jump_percent": (
                RELATIVE_JUMP_PERCENT
            ),
            "relative_jump_min_previous_pct": (
                RELATIVE_JUMP_MIN_PREVIOUS
            ),
            "holder_change_absolute": (
                HOLDER_CHANGE_ABSOLUTE
            ),
            "holder_change_relative_percent": (
                HOLDER_CHANGE_RELATIVE_PERCENT
            ),
            "concentration_change_pp": 25.0,
            "market_absolute_change_pp": (
                MARKET_ABSOLUTE_CHANGE_PP
            ),
            "market_share_threshold": (
                MARKET_SHARE_THRESHOLD
            ),
        },
        "dataset_summary": {
            "rows": rows,
            "unique_snapshot_dates": len(
                snapshot_dates
            ),
            "unique_issuers": len(
                issuers
            ),
            "unique_isins": len(
                isins
            ),
            "first_snapshot_date": (
                first_date
            ),
            "last_snapshot_date": (
                last_date
            ),
        },
        "integrity": {
            "invalid_rows": invalid_rows,
            "date_order_ok": date_order_ok,
            "passed": (
                invalid_rows == 0
                and date_order_ok
            ),
        },
        "findings": {
            "suspicious_category_count": (
                suspicious_count
            ),
            "large_absolute_jumps": (
                large_jumps
            ),
            "large_relative_jumps": (
                relative_jumps
            ),
            "large_holder_count_changes": (
                holder_changes
            ),
            "large_concentration_changes": (
                concentration_changes
            ),
            "issuer_isin_conflicts": (
                issuer_isin_conflicts
            ),
            "isin_issuer_conflicts": (
                isin_issuer_conflicts
            ),
            "market_wide_changes": (
                market_wide_changes
            ),
            "invalid_rows": (
                invalid_values
            ),
        },
        "interpretation": {
            "large_absolute_jumps": (
                "Observationer med en stor "
                "förändring i total synlig "
                "blankning. Kan vara riktiga "
                "positionförändringar eller "
                "indikera rekonstruktionsproblem."
            ),
            "large_relative_jumps": (
                "Relativt stora förändringar "
                "från en redan etablerad "
                "blankningsnivå."
            ),
            "large_holder_count_changes": (
                "Stora förändringar i antalet "
                "synliga aktiva blankare."
            ),
            "large_concentration_changes": (
                "Stor förändring i hur mycket "
                "den största blankaren står "
                "för av den totala synliga "
                "blankningen."
            ),
            "issuer_isin_conflicts": (
                "Samma emittör förekommer med "
                "flera ISIN. Kan vara en "
                "corporate action eller "
                "identitetsfråga och är därför "
                "en kandidat, inte automatiskt "
                "ett fel."
            ),
            "isin_issuer_conflicts": (
                "Samma ISIN förekommer med "
                "olika emittörnamn. Detta är "
                "mer misstänkt och bör granskas."
            ),
            "market_wide_changes": (
                "Datum där en ovanligt stor "
                "andel av observationerna "
                "förändras kraftigt samtidigt."
            ),
            "invalid_rows": (
                "Tekniskt ogiltiga observationer."
            ),
        },
        "reconstruction_metadata": metadata,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return result


def main() -> None:
    """CLI-entrypoint."""

    result = run_qc()

    print(
        "=========================================="
    )
    print(
        "QC AV REKONSTRUERAT FI-AGGREGAT"
    )
    print(
        "=========================================="
    )

    summary = result[
        "dataset_summary"
    ]

    print(
        f"Rows: {summary['rows']}"
    )
    print(
        "Snapshot dates: "
        f"{summary['unique_snapshot_dates']}"
    )
    print(
        "Unique issuers: "
        f"{summary['unique_issuers']}"
    )
    print(
        "Unique ISIN: "
        f"{summary['unique_isins']}"
    )
    print(
        "Period: "
        f"{summary['first_snapshot_date']} "
        f"to "
        f"{summary['last_snapshot_date']}"
    )

    print()
    print(
        "Fynd:"
    )

    findings = result["findings"]

    print(
        "  stora absoluta hopp: "
        f"{len(findings['large_absolute_jumps'])}"
    )
    print(
        "  stora relativa hopp: "
        f"{len(findings['large_relative_jumps'])}"
    )
    print(
        "  stora förändringar i aktiva blankare: "
        f"{len(findings['large_holder_count_changes'])}"
    )
    print(
        "  stora koncentrationsförändringar: "
        f"{len(findings['large_concentration_changes'])}"
    )
    print(
        "  issuer -> flera ISIN: "
        f"{len(findings['issuer_isin_conflicts'])}"
    )
    print(
        "  ISIN -> flera issuer: "
        f"{len(findings['isin_issuer_conflicts'])}"
    )
    print(
        "  marknadsövergripande förändringar: "
        f"{len(findings['market_wide_changes'])}"
    )
    print(
        "  ogiltiga rader: "
        f"{len(findings['invalid_rows'])}"
    )

    print()
    print(
        "Strukturell integritet: "
        f"{result['integrity']['passed']}"
    )
    print(
        "QC-resultat sparat till:"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
