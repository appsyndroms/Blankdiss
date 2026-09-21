from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.config import TARGETS, WALK_FORWARD_WINDOWS
from ml.dataset import build_target, load_features
from ml.research.bootstrap import bootstrap_mean_difference
from ml.research.signals import build_signal


DISCOVERY_END = pd.Timestamp("2025-12-19")

TARGET_NAME = "down_10pct_5d"
SI_SIGNAL_NAME = "short_interest_change"
PRIOR_RETURN_SIGNAL_NAME = "price_momentum_5d"

SI_TAILS = (
    0.20,
    0.10,
    0.05,
    0.025,
    0.01,
)

PRIOR_RETURN_TAILS = (
    0.20,
    0.10,
    0.05,
    0.025,
    0.01,
)

FUTURE_RETURN_HORIZONS = (
    1,
    3,
    5,
    10,
    20,
)

OUTPUT_DIR = Path(
    "data/processed/ml/research/si_context"
)


def target_config():
    for target in TARGETS:
        if target.name == TARGET_NAME:
            return target

    raise ValueError(
        f"Unknown target: {TARGET_NAME}"
    )


def build_splits() -> list[dict[str, Any]]:
    splits = []

    for index, window in enumerate(
        WALK_FORWARD_WINDOWS,
        start=1,
    ):
        splits.append(
            {
                "name": f"window_{index}",
                "train_end": pd.Timestamp(
                    window.train_end
                ),
                "validation_end": pd.Timestamp(
                    window.validation_end
                ),
                "test_end": pd.Timestamp(
                    window.test_end
                ),
            }
        )

    return splits


def cross_sectional_tail_mask(
    frame: pd.DataFrame,
    values: pd.Series,
    fraction: float,
    *,
    direction: str,
) -> pd.Series:
    """
    Creates a cross-sectional tail mask independently for each
    snapshot date.

    Grouping is performed only with information available on the
    snapshot date.
    """
    if not 0 < fraction <= 1:
        raise ValueError(
            f"Invalid tail fraction: {fraction}"
        )

    if direction not in {
        "upper",
        "lower",
    }:
        raise ValueError(
            f"Unknown direction: {direction}"
        )

    result = pd.Series(
        False,
        index=frame.index,
        dtype=bool,
    )

    working = pd.DataFrame(
        {
            "date": frame["snapshot_date"],
            "value": pd.to_numeric(
                values,
                errors="coerce",
            ),
        },
        index=frame.index,
    )

    for _, index in working.groupby(
        "date",
        sort=False,
    ).groups.items():
        local = working.loc[
            index,
            "value",
        ]

        valid = local.notna()

        if not valid.any():
            continue

        ranks = local.loc[
            valid
        ].rank(
            method="average",
            pct=True,
        )

        if direction == "upper":
            selected = (
                ranks
                >= (1.0 - fraction)
            )
        else:
            selected = (
                ranks
                <= fraction
            )

        result.loc[
            selected.index
        ] = selected.astype(bool)

    return result


def event_rate(
    target: np.ndarray,
) -> float | None:
    target = target[
        np.isfinite(target)
    ]

    if target.size == 0:
        return None

    return float(
        target.mean()
    )


def safe_mean(
    values: np.ndarray,
) -> float | None:
    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return None

    return float(
        values.mean()
    )


def stable_seed(
    *parts: object,
) -> int:
    payload = "|".join(
        str(part)
        for part in parts
    ).encode("utf-8")

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    ) % (2**32 - 1)


def analyse_condition(
    frame: pd.DataFrame,
    target: np.ndarray,
    si: pd.Series,
    prior_return: pd.Series,
    *,
    prior_tail: float,
    si_tail: float,
    base_mask: np.ndarray,
    split_name: str,
    evaluation_name: str,
) -> dict[str, Any]:
    """
    Compare high SI-change stocks against other stocks that have
    already experienced a large positive 5-day price move.

    Primary comparison:

        strong prior return + high SI change
        versus
        strong prior return + other SI change

    Both groups therefore come from the same prior-return regime.
    """

    prior_mask = cross_sectional_tail_mask(
        frame,
        prior_return,
        prior_tail,
        direction="upper",
    )

    si_mask = cross_sectional_tail_mask(
        frame,
        si,
        si_tail,
        direction="upper",
    )

    base = (
        base_mask
        & np.isfinite(target)
        & prior_mask.to_numpy()
    )

    high_si = (
        base
        & si_mask.to_numpy()
    )

    other_si = (
        base
        & ~si_mask.to_numpy()
    )

    high_target = target[
        high_si
    ]

    other_target = target[
        other_si
    ]

    high_event_rate = event_rate(
        high_target
    )

    other_event_rate = event_rate(
        other_target
    )

    target_difference = None
    target_ci_low = None
    target_ci_high = None

    if (
        high_target.size > 0
        and other_target.size > 0
    ):
        target_difference = (
            float(
                high_target.mean()
            )
            - float(
                other_target.mean()
            )
        )

        (
            target_ci_low,
            target_ci_high,
        ) = bootstrap_mean_difference(
            high_target,
            other_target,
            seed=stable_seed(
                split_name,
                evaluation_name,
                prior_tail,
                si_tail,
                "target",
            ),
        )

    row: dict[str, Any] = {
        "split": split_name,
        "evaluation": evaluation_name,
        "prior_return_tail": prior_tail,
        "si_tail": si_tail,
        "prior_return_signal": (
            PRIOR_RETURN_SIGNAL_NAME
        ),
        "si_signal": SI_SIGNAL_NAME,
        "target": TARGET_NAME,
        "base_n": int(base.sum()),
        "high_si_n": int(high_si.sum()),
        "other_si_n": int(other_si.sum()),
        "high_si_events": int(
            high_target.sum()
        ),
        "other_si_events": int(
            other_target.sum()
        ),
        "high_si_event_rate": (
            high_event_rate
        ),
        "other_si_event_rate": (
            other_event_rate
        ),
        "event_rate_difference": (
            (
                high_event_rate
                - other_event_rate
            )
            if (
                high_event_rate is not None
                and other_event_rate is not None
            )
            else None
        ),
        "event_rate_lift": (
            (
                high_event_rate
                / other_event_rate
            )
            if (
                high_event_rate is not None
                and other_event_rate is not None
                and other_event_rate > 0
            )
            else None
        ),
        "target_difference": (
            target_difference
        ),
        "target_bootstrap_ci_low": (
            target_ci_low
        ),
        "target_bootstrap_ci_high": (
            target_ci_high
        ),
    }

    for horizon in FUTURE_RETURN_HORIZONS:
        column = (
            f"forward_return_{horizon}d"
        )

        if column not in frame.columns:
            continue

        returns = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        high_returns = returns[
            high_si
            & np.isfinite(returns)
        ]

        other_returns = returns[
            other_si
            & np.isfinite(returns)
        ]

        difference = None
        ci_low = None
        ci_high = None

        if (
            high_returns.size > 0
            and other_returns.size > 0
        ):
            difference = (
                float(
                    high_returns.mean()
                )
                - float(
                    other_returns.mean()
                )
            )

            (
                ci_low,
                ci_high,
            ) = bootstrap_mean_difference(
                high_returns,
                other_returns,
                seed=stable_seed(
                    split_name,
                    evaluation_name,
                    prior_tail,
                    si_tail,
                    horizon,
                ),
            )

        row[
            f"high_si_mean_return_{horizon}d"
        ] = safe_mean(
            high_returns
        )

        row[
            f"other_si_mean_return_{horizon}d"
        ] = safe_mean(
            other_returns
        )

        row[
            f"return_difference_{horizon}d"
        ] = difference

        row[
            f"return_ci_low_{horizon}d"
        ] = ci_low

        row[
            f"return_ci_high_{horizon}d"
        ] = ci_high

    return row


def build_evaluation_masks(
    frame: pd.DataFrame,
    split: dict[str, Any],
) -> dict[str, np.ndarray]:
    """
    Build train/validation/test masks while respecting the frozen
    discovery cutoff.

    A future test period is allowed to exist in the configured
    walk-forward definition, but if the discovery cutoff has not
    reached that period the mask is empty and the evaluation is
    explicitly reported as unavailable.
    """

    dates = frame[
        "snapshot_date"
    ]

    cutoff_mask = (
        dates <= DISCOVERY_END
    ).to_numpy()

    train_mask = (
        dates <= split["train_end"]
    ).to_numpy()

    validation_mask = (
        (
            dates
            > split["train_end"]
        )
        & (
            dates
            <= split["validation_end"]
        )
    ).to_numpy()

    test_mask = (
        (
            dates
            > split["validation_end"]
        )
        & (
            dates
            <= split["test_end"]
        )
    ).to_numpy()

    return {
        "train": (
            train_mask
            & cutoff_mask
        ),
        "validation": (
            validation_mask
            & cutoff_mask
        ),
        "test": (
            test_mask
            & cutoff_mask
        ),
    }


def run_analysis() -> dict[str, Any]:
    print(
        "Loading features...",
        flush=True,
    )

    frame = load_features().copy()

    frame["snapshot_date"] = pd.to_datetime(
        frame["snapshot_date"],
        errors="coerce",
    )

    frame = frame.loc[
        frame["snapshot_date"]
        <= DISCOVERY_END
    ].copy()

    frame.reset_index(
        drop=True,
        inplace=True,
    )

    print(
        f"Rows through "
        f"{DISCOVERY_END.date()}: "
        f"{len(frame):,}",
        flush=True,
    )

    target_cfg = target_config()

    target = build_target(
        frame,
        target_cfg,
    ).to_numpy(
        dtype=float
    )

    # Use Blankdiss' canonical signal definitions rather than
    # assuming that signal names are physical feature columns.
    si = build_signal(
        frame,
        SI_SIGNAL_NAME,
    )

    prior_return = build_signal(
        frame,
        PRIOR_RETURN_SIGNAL_NAME,
    )

    print(
        "Running momentum-conditioned SI analysis...",
        flush=True,
    )

    all_rows: list[dict[str, Any]] = []
    split_status: list[dict[str, Any]] = []

    for split in build_splits():
        masks = build_evaluation_masks(
            frame,
            split,
        )

        train_n = int(
            masks["train"].sum()
        )

        validation_n = int(
            masks["validation"].sum()
        )

        test_n = int(
            masks["test"].sum()
        )

        if test_n == 0:
            test_status = (
                "NOT_AVAILABLE"
            )
        else:
            test_status = "AVAILABLE"

        if validation_n == 0:
            validation_status = (
                "NOT_AVAILABLE"
            )
        else:
            validation_status = "AVAILABLE"

        split_status.append(
            {
                "name": split["name"],
                "train_end": str(
                    split["train_end"].date()
                ),
                "validation_end": str(
                    split[
                        "validation_end"
                    ].date()
                ),
                "test_end": str(
                    split["test_end"].date()
                ),
                "train_rows": train_n,
                "validation_rows": validation_n,
                "test_rows": test_n,
                "validation_status": (
                    validation_status
                ),
                "test_status": test_status,
                "test_available_through": str(
                    min(
                        DISCOVERY_END,
                        split["test_end"],
                    ).date()
                ),
            }
        )

        for evaluation_name in (
            "validation",
            "test",
        ):
            mask = masks[
                evaluation_name
            ]

            if not mask.any():
                print(
                    f"Context evaluation: "
                    f"{split['name']} / "
                    f"{evaluation_name} "
                    f"SKIPPED - no data before "
                    f"{DISCOVERY_END.date()}",
                    flush=True,
                )
                continue

            print(
                f"Context evaluation: "
                f"{split['name']} / "
                f"{evaluation_name}",
                flush=True,
            )

            for prior_tail in (
                PRIOR_RETURN_TAILS
            ):
                for si_tail in SI_TAILS:
                    all_rows.append(
                        analyse_condition(
                            frame,
                            target,
                            si,
                            prior_return,
                            prior_tail=prior_tail,
                            si_tail=si_tail,
                            base_mask=mask,
                            split_name=split[
                                "name"
                            ],
                            evaluation_name=(
                                evaluation_name
                            ),
                        )
                    )

    available_test_windows = [
        row
        for row in split_status
        if row["test_status"] == "AVAILABLE"
    ]

    return {
        "analysis": (
            "momentum_conditioned_short_interest"
        ),
        "discovery_end": str(
            DISCOVERY_END.date()
        ),
        "target": TARGET_NAME,
        "prior_return_signal": (
            PRIOR_RETURN_SIGNAL_NAME
        ),
        "si_signal": SI_SIGNAL_NAME,
        "prior_return_tails": list(
            PRIOR_RETURN_TAILS
        ),
        "si_tails": list(
            SI_TAILS
        ),
        "future_return_horizons": list(
            FUTURE_RETURN_HORIZONS
        ),
        "method": {
            "conditioning": (
                "large positive prior "
                "5-day return"
            ),
            "cross_sectional_grouping": (
                "independently per snapshot_date"
            ),
            "comparison": (
                "high SI-change versus "
                "other SI-change within "
                "the same prior-return tail"
            ),
            "signal_resolution": (
                "canonical ml.research.signals "
                "definitions"
            ),
            "walk_forward": True,
            "future_information_used_for_grouping": False,
            "discovery_cutoff": (
                str(
                    DISCOVERY_END.date()
                )
            ),
        },
        "split_status": split_status,
        "available_test_windows": (
            available_test_windows
        ),
        "rows": all_rows,
    }


def write_outputs(
    document: dict[str, Any],
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        OUTPUT_DIR
        / "results.json"
    ).write_text(
        json.dumps(
            document,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    rows = document["rows"]

    pd.DataFrame(
        rows
    ).to_csv(
        OUTPUT_DIR
        / "context_analysis.csv",
        index=False,
    )

    test_rows = [
        row
        for row in rows
        if row["evaluation"] == "test"
    ]

    pd.DataFrame(
        test_rows
    ).to_csv(
        OUTPUT_DIR
        / "test_results.csv",
        index=False,
    )

    lines = [
        "# Momentum-conditioned Short Interest",
        "",
        f"- Target: `{TARGET_NAME}`",
        f"- Prior return signal: "
        f"`{PRIOR_RETURN_SIGNAL_NAME}`",
        f"- SI signal: `{SI_SIGNAL_NAME}`",
        f"- Frozen cutoff: "
        f"`{DISCOVERY_END.date()}`",
        "",
        "## Data availability",
        "",
        "| Window | Train rows | Validation rows | Test rows | Validation status | Test status |",
        "|---|---:|---:|---:|---|---|",
    ]

    for split in document[
        "split_status"
    ]:
        lines.append(
            "| "
            f"{split['name']} | "
            f"{split['train_rows']} | "
            f"{split['validation_rows']} | "
            f"{split['test_rows']} | "
            f"{split['validation_status']} | "
            f"{split['test_status']} |"
        )

    lines.extend(
        [
            "",
            "Only periods that exist before the frozen discovery "
            "cutoff are evaluated. A configured future test window "
            "with no observations is reported as "
            "`NOT_AVAILABLE`, not as a failed OOS test.",
            "",
            "## OOS results",
            "",
            "| Window | Eval | Prior tail | SI tail | N | High SI N | Other SI N | High SI event rate | Other event rate | Event lift | 5d return diff | 20d return diff |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in test_rows:
        lines.append(
            "| "
            f"{row['split']} | "
            f"{row['evaluation']} | "
            f"{row['prior_return_tail']:.3f} | "
            f"{row['si_tail']:.3f} | "
            f"{row['base_n']} | "
            f"{row['high_si_n']} | "
            f"{row['other_si_n']} | "
            f"{_fmt(row['high_si_event_rate'])} | "
            f"{_fmt(row['other_si_event_rate'])} | "
            f"{_fmt(row['event_rate_lift'])} | "
            f"{_fmt(row.get('return_difference_5d'))} | "
            f"{_fmt(row.get('return_difference_20d'))} |"
        )

    if not test_rows:
        lines.extend(
            [
                "",
                "No OOS test observations are available "
                "before the frozen discovery cutoff.",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This analysis is descriptive and OOS. It does not "
            "select a trading rule automatically.",
            "",
            "The main question is whether high SI change remains "
            "associated with negative subsequent returns after "
            "conditioning on a large prior price increase.",
            "",
            "The configured walk-forward windows may extend beyond "
            "the frozen discovery cutoff. Such future portions are "
            "not evaluated until corresponding observations exist.",
        ]
    )

    (
        OUTPUT_DIR
        / "report.md"
    ).write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    metadata = {
        "analysis": (
            "momentum_conditioned_short_interest"
        ),
        "created_at_utc": pd.Timestamp.now(
            tz="UTC"
        ).isoformat(),
        "discovery_end": str(
            DISCOVERY_END.date()
        ),
        "target": TARGET_NAME,
        "prior_return_signal": (
            PRIOR_RETURN_SIGNAL_NAME
        ),
        "si_signal": SI_SIGNAL_NAME,
        "row_count": len(rows),
        "test_row_count": len(
            test_rows
        ),
        "available_test_window_count": len(
            document[
                "available_test_windows"
            ]
        ),
        "walk_forward_windows": (
            document[
                "split_status"
            ]
        ),
    }

    (
        OUTPUT_DIR
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _fmt(
    value: Any,
) -> str:
    if value is None:
        return ""

    if isinstance(
        value,
        float,
    ):
        return f"{value:.6f}"

    return str(value)


def main() -> None:
    document = run_analysis()

    write_outputs(
        document
    )

    print(
        f"Written to {OUTPUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
