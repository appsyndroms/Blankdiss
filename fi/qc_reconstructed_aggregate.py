"""Economic and structural QC for the reconstructed FI aggregate series."""
from __future__ import annotations
import json
import math
from collections import defaultdict
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
ABSOLUTE_JUMP_PP = 3.0
RELATIVE_JUMP_PERCENT = 100.0
RELATIVE_JUMP_MIN_PREVIOUS = 0.5
HOLDER_CHANGE_ABSOLUTE = 5
HOLDER_CHANGE_RELATIVE_PERCENT = 100.0
CONCENTRATION_CHANGE_PP = 25.0
MARKET_ABSOLUTE_CHANGE_PP = 1.0
MARKET_SHARE_THRESHOLD = 0.20
MAX_EXAMPLES = 200
def finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
def finite_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
def add(target, value):
    if len(target) < MAX_EXAMPLES:
        target.append(value)
def load_metadata():
    path = (
        ROOT
        / "data"
        / "processed"
        / "fi"
        / "aggregate"
        / "reconstructed_metadata.json"
    )
    if not path.exists():
        return {}
    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}
def run_qc():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Saknar rekonstruerad FI-serie: {INPUT}"
        )
    required = {
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
    dates = set()
    issuers = set()
    isins = set()
    # Time-series identity:
    # issuer + ISIN
    previous_by_security = {}
    # Multiple observations for the same security/date
    # are tracked separately and are not treated as
    # time-series changes.
    observations_by_security_date = defaultdict(int)
    isin_to_issuers = defaultdict(set)
    issuer_to_isins = defaultdict(set)
    market = defaultdict(
        lambda: {
            "observations": 0,
            "changes_ge_1pp": 0,
            "changes_ge_3pp": 0,
            "absolute_change_sum": 0.0,
        }
    )
    findings = {
        "large_absolute_jumps": [],
        "large_relative_jumps": [],
        "large_holder_count_changes": [],
        "large_concentration_changes": [],
        "issuer_isin_conflicts": [],
        "isin_issuer_conflicts": [],
        "market_wide_changes": [],
        "same_security_same_date": [],
        "invalid_rows": [],
    }
    # Full counters are not limited by MAX_EXAMPLES.
    holder_stats = {
        "total_changes": 0,
        "changes_ge_5": 0,
        "changes_ge_10": 0,
        "changes_ge_20": 0,
        "changes_ge_50": 0,
        "relative_changes_ge_100_percent": 0,
        "largest_absolute_change": 0,
        "largest_relative_change_percent": 0.0,
    }
    concentration_stats = {
        "total_changes": 0,
        "changes_ge_25pp": 0,
        "changes_ge_40pp": 0,
        "changes_ge_60pp": 0,
        "changes_ge_80pp": 0,
        "largest_absolute_change_pp": 0.0,
    }
    jump_stats = {
        "total_time_series_changes": 0,
        "absolute_changes_ge_1pp": 0,
        "absolute_changes_ge_2pp": 0,
        "absolute_changes_ge_3pp": 0,
        "absolute_changes_ge_5pp": 0,
        "relative_changes_ge_100_percent": 0,
    }
    top_holder_changes = []
    top_concentration_changes = []
    first_date = None
    last_date = None
    with INPUT.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                add(
                    findings["invalid_rows"],
                    {
                        "line": line_number,
                        "reason": "invalid_json",
                    },
                )
                continue
            if not required.issubset(row):
                invalid_rows += 1
                add(
                    findings["invalid_rows"],
                    {
                        "line": line_number,
                        "reason": "missing_columns",
                        "missing": sorted(
                            required - set(row)
                        ),
                    },
                )
                continue
            date = str(row["snapshot_date"]).strip()
            issuer = str(row["issuer"]).strip()
            isin = str(row.get("isin") or "").strip()
            short_interest = finite_float(
                row["short_interest_pct"]
            )
            holders = finite_int(
                row["active_holders"]
            )
            max_position = finite_float(
                row["max_individual_position_pct"]
            )
            concentration = finite_float(
                row["max_position_share_pct"]
            )
            if (
                not date
                or not issuer
                or short_interest is None
                or holders is None
                or max_position is None
                or concentration is None
            ):
                invalid_rows += 1
                add(
                    findings["invalid_rows"],
                    {
                        "line": line_number,
                        "snapshot_date": date,
                        "issuer": issuer,
                        "reason": "invalid_required_value",
                    },
                )
                continue
            rows += 1
            dates.add(date)
            issuers.add(issuer)
            if first_date is None:
                first_date = date
            last_date = date
            if isin:
                isins.add(isin)
                isin_to_issuers[isin].add(issuer)
                issuer_to_isins[issuer].add(isin)
            current = {
                "snapshot_date": date,
                "issuer": issuer,
                "isin": isin or None,
                "short_interest_pct": short_interest,
                "active_holders": holders,
                "max_individual_position_pct": max_position,
                "max_position_share_pct": concentration,
            }
            security_key = (issuer, isin)
            security_date_key = (
                issuer,
                isin,
                date,
            )
            observations_by_security_date[
                security_date_key
            ] += 1
            if (
                observations_by_security_date[
                    security_date_key
                ]
                > 1
            ):
                add(
                    findings["same_security_same_date"],
                    {
                        "snapshot_date": date,
                        "issuer": issuer,
                        "isin": isin or None,
                        "reason": (
                            "multiple_observations_same_security_same_date"
                        ),
                    },
                )
                continue
            previous = previous_by_security.get(
                security_key
            )
            if previous is not None:
                previous_date = pd.Timestamp(
                    previous["snapshot_date"]
                )
                current_date = pd.Timestamp(date)
                if current_date <= previous_date:
                    continue
                gap_days = (
                    current_date - previous_date
                ).days
                previous_short = previous[
                    "short_interest_pct"
                ]
                delta = (
                    short_interest
                    - previous_short
                )
                absolute_delta = abs(delta)
                jump_stats[
                    "total_time_series_changes"
                ] += 1
                if absolute_delta >= 1.0:
                    jump_stats[
                        "absolute_changes_ge_1pp"
                    ] += 1
                if absolute_delta >= 2.0:
                    jump_stats[
                        "absolute_changes_ge_2pp"
                    ] += 1
                if absolute_delta >= 3.0:
                    jump_stats[
                        "absolute_changes_ge_3pp"
                    ] += 1
                if absolute_delta >= 5.0:
                    jump_stats[
                        "absolute_changes_ge_5pp"
                    ] += 1
                stats = market[date]
                stats["observations"] += 1
                stats["absolute_change_sum"] += (
                    absolute_delta
                )
                if (
                    absolute_delta
                    >= MARKET_ABSOLUTE_CHANGE_PP
                ):
                    stats["changes_ge_1pp"] += 1
                if (
                    absolute_delta
                    >= ABSOLUTE_JUMP_PP
                ):
                    stats["changes_ge_3pp"] += 1
                base = {
                    **current,
                    "previous_snapshot_date": (
                        previous["snapshot_date"]
                    ),
                    "previous_short_interest_pct": (
                        previous_short
                    ),
                    "delta_pp": round(
                        delta,
                        6,
                    ),
                    "gap_days": gap_days,
                }
                # Short-interest absolute jump.
                if absolute_delta >= ABSOLUTE_JUMP_PP:
                    add(
                        findings[
                            "large_absolute_jumps"
                        ],
                        {
                            **base,
                            "absolute_delta_pp": round(
                                absolute_delta,
                                6,
                            ),
                            "reason": (
                                "large_absolute_jump"
                            ),
                        },
                    )
                # Short-interest relative jump.
                if (
                    previous_short
                    >= RELATIVE_JUMP_MIN_PREVIOUS
                ):
                    relative = (
                        absolute_delta
                        / previous_short
                        * 100.0
                    )
                    if (
                        relative
                        >= RELATIVE_JUMP_PERCENT
                    ):
                        jump_stats[
                            "relative_changes_ge_100_percent"
                        ] += 1
                        add(
                            findings[
                                "large_relative_jumps"
                            ],
                            {
                                **base,
                                "relative_change_percent": round(
                                    delta
                                    / previous_short
                                    * 100.0,
                                    6,
                                ),
                                "reason": (
                                    "large_relative_jump"
                                ),
                            },
                        )
                # Active holder changes.
                previous_holders = previous[
                    "active_holders"
                ]
                holder_delta = (
                    holders - previous_holders
                )
                holder_abs = abs(holder_delta)
                holder_stats[
                    "total_changes"
                ] += 1
                if holder_abs >= 5:
                    holder_stats[
                        "changes_ge_5"
                    ] += 1
                if holder_abs >= 10:
                    holder_stats[
                        "changes_ge_10"
                    ] += 1
                if holder_abs >= 20:
                    holder_stats[
                        "changes_ge_20"
                    ] += 1
                if holder_abs >= 50:
                    holder_stats[
                        "changes_ge_50"
                    ] += 1
                if previous_holders:
                    holder_relative = (
                        holder_abs
                        / previous_holders
                        * 100.0
                    )
                else:
                    holder_relative = (
                        100.0
                        if holders
                        else 0.0
                    )
                if (
                    holder_relative
                    >= HOLDER_CHANGE_RELATIVE_PERCENT
                ):
                    holder_stats[
                        "relative_changes_ge_100_percent"
                    ] += 1
                holder_stats[
                    "largest_absolute_change"
                ] = max(
                    holder_stats[
                        "largest_absolute_change"
                    ],
                    holder_abs,
                )
                holder_stats[
                    "largest_relative_change_percent"
                ] = max(
                    holder_stats[
                        "largest_relative_change_percent"
                    ],
                    holder_relative,
                )
                holder_record = {
                    **base,
                    "previous_active_holders": (
                        previous_holders
                    ),
                    "holder_delta": holder_delta,
                    "holder_absolute_change": (
                        holder_abs
                    ),
                    "holder_relative_change_percent": (
                        round(
                            holder_relative,
                            6,
                        )
                    ),
                    "reason": (
                        "large_holder_count_change"
                    ),
                }
                if (
                    holder_abs
                    >= HOLDER_CHANGE_ABSOLUTE
                    or (
                        previous_holders > 0
                        and holder_relative
                        >= HOLDER_CHANGE_RELATIVE_PERCENT
                    )
                ):
                    add(
                        findings[
                            "large_holder_count_changes"
                        ],
                        holder_record,
                    )
                top_holder_changes.append(
                    holder_record
                )
                if (
                    len(top_holder_changes)
                    > MAX_EXAMPLES * 2
                ):
                    top_holder_changes = sorted(
                        top_holder_changes,
                        key=lambda item: (
                            item[
                                "holder_absolute_change"
                            ],
                            item[
                                "holder_relative_change_percent"
                            ],
                        ),
                        reverse=True,
                    )[:MAX_EXAMPLES]
                # Concentration changes.
                previous_concentration = previous[
                    "max_position_share_pct"
                ]
                concentration_delta = (
                    concentration
                    - previous_concentration
                )
                concentration_abs = abs(
                    concentration_delta
                )
                concentration_stats[
                    "total_changes"
                ] += 1
                if concentration_abs >= 25.0:
                    concentration_stats[
                        "changes_ge_25pp"
                    ] += 1
                if concentration_abs >= 40.0:
                    concentration_stats[
                        "changes_ge_40pp"
                    ] += 1
                if concentration_abs >= 60.0:
                    concentration_stats[
                        "changes_ge_60pp"
                    ] += 1
                if concentration_abs >= 80.0:
                    concentration_stats[
                        "changes_ge_80pp"
                    ] += 1
                concentration_stats[
                    "largest_absolute_change_pp"
                ] = max(
                    concentration_stats[
                        "largest_absolute_change_pp"
                    ],
                    concentration_abs,
                )
                concentration_record = {
                    **base,
                    "previous_max_position_share_pct": (
                        previous_concentration
                    ),
                    "concentration_delta_pp": round(
                        concentration_delta,
                        6,
                    ),
                    "concentration_absolute_change_pp": (
                        round(
                            concentration_abs,
                            6,
                        )
                    ),
                    "reason": (
                        "large_concentration_change"
                    ),
                }
                if (
                    concentration_abs
                    >= CONCENTRATION_CHANGE_PP
                ):
                    add(
                        findings[
                            "large_concentration_changes"
                        ],
                        concentration_record,
                    )
                top_concentration_changes.append(
                    concentration_record
                )
                if (
                    len(top_concentration_changes)
                    > MAX_EXAMPLES * 2
                ):
                    top_concentration_changes = sorted(
                        top_concentration_changes,
                        key=lambda item: item[
                            "concentration_absolute_change_pp"
                        ],
                        reverse=True,
                    )[:MAX_EXAMPLES]
            previous_by_security[
                security_key
            ] = current
    # Keep the largest holder/concentration changes.
    top_holder_changes = sorted(
        top_holder_changes,
        key=lambda item: (
            item["holder_absolute_change"],
            item[
                "holder_relative_change_percent"
            ],
        ),
        reverse=True,
    )[:MAX_EXAMPLES]
    top_concentration_changes = sorted(
        top_concentration_changes,
        key=lambda item: item[
            "concentration_absolute_change_pp"
        ],
        reverse=True,
    )[:MAX_EXAMPLES]
    # ISIN / issuer consistency.
    for isin, names in sorted(
        isin_to_issuers.items()
    ):
        if len(names) > 1:
            add(
                findings["isin_issuer_conflicts"],
                {
                    "isin": isin,
                    "issuers": sorted(names),
                    "issuer_count": len(names),
                    "reason": (
                        "same_isin_multiple_issuers"
                    ),
                },
            )
    for issuer, issuer_isins in sorted(
        issuer_to_isins.items()
    ):
        if len(issuer_isins) > 1:
            add(
                findings["issuer_isin_conflicts"],
                {
                    "issuer": issuer,
                    "isins": sorted(
                        issuer_isins
                    ),
                    "isin_count": len(
                        issuer_isins
                    ),
                    "reason": (
                        "same_issuer_multiple_isins"
                    ),
                },
            )
    # Market-wide changes.
    for date, stats in sorted(
        market.items()
    ):
        if stats["observations"] == 0:
            continue
        share = (
            stats["changes_ge_1pp"]
            / stats["observations"]
        )
        if share >= MARKET_SHARE_THRESHOLD:
            add(
                findings["market_wide_changes"],
                {
                    "snapshot_date": date,
                    "observations_with_previous": (
                        stats["observations"]
                    ),
                    "changes_ge_1pp": (
                        stats["changes_ge_1pp"]
                    ),
                    "changes_ge_3pp": (
                        stats["changes_ge_3pp"]
                    ),
                    "share_ge_1pp": round(
                        share,
                        6,
                    ),
                    "share_ge_3pp": round(
                        (
                            stats["changes_ge_3pp"]
                            / stats["observations"]
                        ),
                        6,
                    ),
                    "mean_absolute_change_pp": round(
                        (
                            stats[
                                "absolute_change_sum"
                            ]
                            / stats["observations"]
                        ),
                        6,
                    ),
                    "reason": (
                        "market_wide_change"
                    ),
                },
            )
    # Dataset ordering.
    date_order_ok = True
    previous_date = None
    with INPUT.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                current_date = pd.Timestamp(
                    json.loads(line)[
                        "snapshot_date"
                    ]
                )
            except (
                json.JSONDecodeError,
                KeyError,
                ValueError,
                TypeError,
            ):
                continue
            if (
                previous_date is not None
                and current_date < previous_date
            ):
                date_order_ok = False
                break
            previous_date = current_date
    category_counts = {
        key: len(value)
        for key, value in findings.items()
    }
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
            "concentration_change_pp": (
                CONCENTRATION_CHANGE_PP
            ),
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
                dates
            ),
            "unique_issuers": len(
                issuers
            ),
            "unique_isins": len(
                isins
            ),
            "first_snapshot_date": first_date,
            "last_snapshot_date": last_date,
        },
        "integrity": {
            "invalid_rows": invalid_rows,
            "date_order_ok": date_order_ok,
            "passed": (
                invalid_rows == 0
                and date_order_ok
            ),
        },
        "change_statistics": {
            "short_interest": jump_stats,
            "active_holders": holder_stats,
            "concentration": concentration_stats,
        },
        "finding_counts": category_counts,
        "findings": findings,
        "top_examples": {
            "holder_changes": (
                top_holder_changes
            ),
            "concentration_changes": (
                top_concentration_changes
            ),
        },
        "conflict_statistics": {
            "issuers_with_multiple_isins": sum(
                1
                for values in issuer_to_isins.values()
                if len(values) > 1
            ),
            "isins_with_multiple_issuers": sum(
                1
                for values in isin_to_issuers.values()
                if len(values) > 1
            ),
            "total_unique_issuers": len(
                issuer_to_isins
            ),
            "total_unique_isins": len(
                isin_to_issuers
            ),
        },
        "interpretation": {
            "note": (
                "Findings are candidates for review, "
                "not automatic data errors."
            ),
            "data_mutation": (
                "This QC does not modify, remove, "
                "correct or impute any observations "
                "in the input dataset."
            ),
            "time_series_identity": (
                "Time-series changes are calculated "
                "per issuer + ISIN, not issuer alone."
            ),
            "same_day_observations": (
                "Multiple observations for the same "
                "issuer + ISIN on the same date are "
                "not treated as time-series changes."
            ),
            "threshold_transitions": (
                "A missing issuer observation is not "
                "treated as zero because FI's individual "
                "publication threshold can make positions "
                "disappear from the visible data."
            ),
            "relative_changes": (
                "Large relative changes can be caused "
                "by small starting values near the FI "
                "publication threshold and are therefore "
                "not automatically suspicious."
            ),
            "issuer_isin": (
                "Corporate actions can legitimately "
                "produce multiple ISIN for one issuer; "
                "the reverse mapping is more suspicious."
            ),
        },
        "reconstruction_metadata": load_metadata(),
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
def main():
    result = run_qc()
    summary = result["dataset_summary"]
    integrity = result["integrity"]
    changes = result["change_statistics"]
    conflicts = result["conflict_statistics"]
    counts = result["finding_counts"]
    short_interest = changes[
        "short_interest"
    ]
    holders = changes[
        "active_holders"
    ]
    concentration = changes[
        "concentration"
    ]
    print("FI QC")
    print(
        f"Rows: {summary['rows']} | "
        f"Dates: {summary['unique_snapshot_dates']} | "
        f"Issuers: {summary['unique_issuers']} | "
        f"ISIN: {summary['unique_isins']}"
    )
    print(
        f"Period: "
        f"{summary['first_snapshot_date']} "
        f"-> "
        f"{summary['last_snapshot_date']}"
    )
    print(
        f"Integrity: "
        f"{'PASS' if integrity['passed'] else 'FAIL'} | "
        f"Invalid rows: {integrity['invalid_rows']} | "
        f"Date order: "
        f"{'OK' if integrity['date_order_ok'] else 'FAIL'}"
    )
    print(
        f"Short interest: "
        f">=1pp {short_interest['absolute_changes_ge_1pp']} | "
        f">=3pp {short_interest['absolute_changes_ge_3pp']} | "
        f">=5pp {short_interest['absolute_changes_ge_5pp']}"
    )
    print(
        f"Holders: "
        f">=5 {holders['changes_ge_5']} | "
        f"largest abs {holders['largest_absolute_change']}"
    )
    print(
        f"Concentration: "
        f">=25pp {concentration['changes_ge_25pp']} | "
        f">=60pp {concentration['changes_ge_60pp']} | "
        f"largest {concentration['largest_absolute_change_pp']:.2f}pp"
    )
    print(
        f"ISIN: "
        f"issuer->multiple {conflicts['issuers_with_multiple_isins']} | "
        f"ISIN->multiple {conflicts['isins_with_multiple_issuers']}"
    )
    print(
        f"Market-wide changes: "
        f"{counts['market_wide_changes']} | "
        f"Same security/date: "
        f"{counts['same_security_same_date']}"
    )
    print(
        f"QC report: {OUTPUT}"
    )
if __name__ == "__main__":
    main()
