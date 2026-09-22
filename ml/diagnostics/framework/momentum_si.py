from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ExperimentResult
from ml.research.signals import build_signal


# These are the previously identified cells.
# They remain locked for this follow-up analysis.
FOCUS_CELLS = (
    (1, 3),
    (10, 6),
    (6, 9),
    (9, 10),
    (7, 10),
)

# Event definition used throughout this follow-up.
EVENT_THRESHOLD = -0.05

# Event-risk bands are deliberately broad and fixed.
EVENT_RISK_BANDS = (
    ("top_5pct", 0.00, 0.05),
    ("5_20pct", 0.05, 0.20),
    ("20_50pct", 0.20, 0.50),
    ("bottom_50pct", 0.50, 1.00),
)

HORIZONS = (1, 3, 5, 10, 20)

# The 9x10 follow-up focuses on the actual SI change rather than
# treating SI decile as sufficient.
SI_CHANGE_BUCKETS = (
    "negative",
    "zero",
    "positive_normal",
    "positive_extreme",
)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(index=frame.index, dtype=float)

    return pd.to_numeric(frame[column], errors="coerce")


def _numeric_mean(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column).dropna()
    return float(values.mean()) if not values.empty else float("nan")


def _numeric_median(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column).dropna()
    return float(values.median()) if not values.empty else float("nan")


def _event_rate(frame: pd.DataFrame) -> float:
    values = _numeric(frame, "forward_return_5d").dropna()

    if values.empty:
        return float("nan")

    return float((values <= EVENT_THRESHOLD).mean())


def _cross_sectional_deciles(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Assign 0-9 deciles independently for each snapshot date."""

    values = _numeric(frame, column)

    ranks = values.groupby(frame["snapshot_date"]).rank(
        method="first",
        pct=True,
    )

    deciles = np.ceil(ranks * 10).astype("Int64") - 1
    deciles = deciles.clip(lower=0, upper=9)

    return deciles.fillna(-1).astype(int)


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    if "snapshot_date" not in result.columns:
        raise ValueError(
            "Momentum/SI analysis requires snapshot_date"
        )

    result["snapshot_date"] = pd.to_datetime(
        result["snapshot_date"],
        errors="coerce",
    )

    result = result.loc[
        result["snapshot_date"].notna()
    ].copy()

    result["price_momentum_5d"] = build_signal(
        result,
        "price_momentum_5d",
    )

    result["short_interest_change"] = build_signal(
        result,
        "short_interest_change",
    )

    result["momentum_decile"] = _cross_sectional_deciles(
        result,
        "price_momentum_5d",
    )

    result["si_decile"] = _cross_sectional_deciles(
        result,
        "short_interest_change",
    )

    # Prefer the existing event-risk score if the surrounding research
    # context already provides it.
    if "event_score" in result.columns:
        result["event_score"] = _numeric(
            result,
            "event_score",
        )

    return result


def _cell_mask(
    frame: pd.DataFrame,
    momentum_decile: int,
    si_decile: int,
) -> pd.Series:
    return (
        (frame["momentum_decile"] == momentum_decile - 1)
        & (frame["si_decile"] == si_decile - 1)
    )


def _momentum_control_mask(
    frame: pd.DataFrame,
    momentum_decile: int,
    exclude_si_decile: int | None = None,
) -> pd.Series:
    mask = frame["momentum_decile"] == momentum_decile - 1

    if exclude_si_decile is not None:
        mask &= frame["si_decile"] != exclude_si_decile - 1

    return mask


def _event_rate_row(
    *,
    label: str,
    frame: pd.DataFrame,
) -> dict:
    return {
        "group": label,
        "n": int(len(frame)),
        "event_rate": _event_rate(frame),
        "momentum_mean": _numeric_mean(
            frame,
            "price_momentum_5d",
        ),
        "momentum_median": _numeric_median(
            frame,
            "price_momentum_5d",
        ),
        "si_change_mean": _numeric_mean(
            frame,
            "short_interest_change",
        ),
        "si_change_median": _numeric_median(
            frame,
            "short_interest_change",
        ),
        "si_level_mean": _numeric_mean(
            frame,
            "short_interest_pct",
        ),
        "si_level_median": _numeric_median(
            frame,
            "short_interest_pct",
        ),
        "volatility_mean": _numeric_mean(
            frame,
            "volatility_20d",
        ),
        "volatility_median": _numeric_median(
            frame,
            "volatility_20d",
        ),
    }


def _delta_pp(
    focal: pd.DataFrame,
    control: pd.DataFrame,
) -> float:
    focal_rate = _event_rate(focal)
    control_rate = _event_rate(control)

    if not np.isfinite(focal_rate) or not np.isfinite(control_rate):
        return float("nan")

    return float(focal_rate - control_rate)


def _event_risk_cutoffs(
    frame: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate event-risk cutoffs cross-sectionally from the available
    frame.

    The cutoffs are descriptive here; this experiment does not fit
    another model. If event_score is absent, no bands are generated.
    """

    if "event_score" not in frame.columns:
        return {}

    values = _numeric(frame, "event_score").dropna()

    if values.empty:
        return {}

    return {
        name: float(values.quantile(1.0 - upper))
        for name, _, upper in EVENT_RISK_BANDS
    }


def _event_risk_band(
    frame: pd.DataFrame,
    band_name: str,
    lower: float,
    upper: float,
) -> pd.Series:
    if "event_score" not in frame.columns:
        return pd.Series(
            False,
            index=frame.index,
        )

    values = _numeric(frame, "event_score")

    if lower == 0.0:
        cutoff = float(values.quantile(1.0 - upper))
        return values >= cutoff

    if upper == 1.0:
        cutoff = float(values.quantile(1.0 - lower))
        return values < cutoff

    upper_cut = float(values.quantile(1.0 - upper))
    lower_cut = float(values.quantile(1.0 - lower))

    return (
        (values >= upper_cut)
        & (values < lower_cut)
    )


def _add_si_change_buckets(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    change = _numeric(
        result,
        "short_interest_change",
    )

    positive = change[change > 0].dropna()

    if positive.empty:
        extreme_cutoff = float("inf")
    else:
        # Extreme = top 20% of positive SI changes.
        extreme_cutoff = float(
            positive.quantile(0.80)
        )

    result["si_change_bucket"] = np.select(
        [
            change < 0,
            change == 0,
            change > 0,
        ],
        [
            "negative",
            "zero",
            "positive_normal",
        ],
        default="missing",
    )

    extreme_mask = (
        change >= extreme_cutoff
    ) & change.notna()

    result.loc[
        extreme_mask,
        "si_change_bucket",
    ] = "positive_extreme"

    result["si_change_extreme_cutoff"] = extreme_cutoff

    return result


def _cell_context_rows(
    frame: pd.DataFrame,
    *,
    focus_cells: tuple[tuple[int, int], ...],
) -> list[dict]:
    rows: list[dict] = []

    for momentum_decile, si_decile in focus_cells:
        focal_mask = _cell_mask(
            frame,
            momentum_decile,
            si_decile,
        )

        focal = frame.loc[focal_mask].copy()

        momentum_control_mask = _momentum_control_mask(
            frame,
            momentum_decile,
            exclude_si_decile=si_decile,
        )

        momentum_control = frame.loc[
            momentum_control_mask
        ].copy()

        row = _event_rate_row(
            label=f"{momentum_decile}x{si_decile}",
            frame=focal,
        )

        row.update(
            {
                "momentum_decile": momentum_decile,
                "si_decile": si_decile,
                "control_n": int(
                    len(momentum_control)
                ),
                "control_event_rate": _event_rate(
                    momentum_control
                ),
                "delta_vs_same_momentum_pp": _delta_pp(
                    focal,
                    momentum_control,
                ),
            }
        )

        rows.append(row)

    return rows


def _conditional_rows(
    frame: pd.DataFrame,
    *,
    focus_cells: tuple[tuple[int, int], ...],
) -> list[dict]:
    """
    Compare each locked cell against the same momentum decile while
    controlling for event-risk band.

    This is deliberately descriptive. It answers:
        Does SI still matter inside the same momentum + risk regime?
    """

    if "event_score" not in frame.columns:
        return []

    rows: list[dict] = []

    for momentum_decile, si_decile in focus_cells:
        focal_mask = _cell_mask(
            frame,
            momentum_decile,
            si_decile,
        )

        focal_all = frame.loc[focal_mask].copy()

        for band_name, lower, upper in EVENT_RISK_BANDS:
            band_mask = _event_risk_band(
                frame,
                band_name,
                lower,
                upper,
            )

            focal = frame.loc[
                focal_mask & band_mask
            ].copy()

            control_mask = (
                _momentum_control_mask(
                    frame,
                    momentum_decile,
                    exclude_si_decile=si_decile,
                )
                & band_mask
            )

            control = frame.loc[
                control_mask
            ].copy()

            rows.append(
                {
                    "momentum_decile": momentum_decile,
                    "si_decile": si_decile,
                    "cell": (
                        f"{momentum_decile}x"
                        f"{si_decile}"
                    ),
                    "event_risk_band": band_name,
                    "n": int(len(focal)),
                    "event_rate": _event_rate(
                        focal
                    ),
                    "control_n": int(len(control)),
                    "control_event_rate": _event_rate(
                        control
                    ),
                    "delta_vs_same_momentum_and_risk_pp": (
                        _delta_pp(
                            focal,
                            control,
                        )
                    ),
                    "event_score_mean": _numeric_mean(
                        focal,
                        "event_score",
                    ),
                    "event_score_median": _numeric_median(
                        focal,
                        "event_score",
                    ),
                }
            )

    return rows


def _nine_by_ten_si_change_rows(
    frame: pd.DataFrame,
) -> list[dict]:
    """
    Mechanism follow-up for the 9x10 cell.

    The purpose is to determine whether 9x10 is driven by the
    SI-decile label itself or by unusually large actual SI changes.
    """

    focal = frame.loc[
        _cell_mask(frame, 9, 10)
    ].copy()

    if focal.empty:
        return []

    focal = _add_si_change_buckets(
        focal
    )

    rows: list[dict] = []

    for bucket in SI_CHANGE_BUCKETS:
        local = focal.loc[
            focal["si_change_bucket"] == bucket
        ].copy()

        if local.empty:
            continue

        row = _event_rate_row(
            label=bucket,
            frame=local,
        )

        row.update(
            {
                "cell": "9x10",
                "si_change_bucket": bucket,
                "n": int(len(local)),
                "event_rate": _event_rate(
                    local
                ),
                "forward_return_5d_mean": _numeric_mean(
                    local,
                    "forward_return_5d",
                ),
                "forward_return_5d_median": _numeric_median(
                    local,
                    "forward_return_5d",
                ),
            }
        )

        rows.append(row)

    return rows


def _nine_by_ten_vs_si_level_rows(
    frame: pd.DataFrame,
) -> list[dict]:
    """
    Compare the 9x10 cell against 9x1..9 by actual SI change and
    SI level. This helps identify whether the decile is merely acting
    as a proxy for a few extreme observations.
    """

    focal = frame.loc[
        frame["momentum_decile"] == 8
    ].copy()

    if focal.empty:
        return []

    focal = _add_si_change_buckets(
        focal
    )

    rows: list[dict] = []

    for si_decile in range(1, 11):
        local = focal.loc[
            focal["si_decile"] == si_decile - 1
        ].copy()

        if local.empty:
            continue

        row = _event_rate_row(
            label=f"9x{si_decile}",
            frame=local,
        )

        row.update(
            {
                "momentum_decile": 9,
                "si_decile": si_decile,
                "cell": f"9x{si_decile}",
                "n": int(len(local)),
                "event_rate": _event_rate(
                    local
                ),
                "positive_si_change_rate": (
                    float(
                        (
                            _numeric(
                                local,
                                "short_interest_change",
                            )
                            > 0
                        ).mean()
                    )
                ),
                "extreme_si_change_rate": (
                    float(
                        (
                            local["si_change_bucket"]
                            == "positive_extreme"
                        ).mean()
                    )
                ),
            }
        )

        rows.append(row)

    return rows


def run_momentum_si_cell_context(
    context,
    *,
    focus_cells: tuple[tuple[int, int], ...] = FOCUS_CELLS,
    context_columns: tuple[str, ...] = (),
    horizons: tuple[int, ...] = HORIZONS,
) -> ExperimentResult:
    """
    Locked Momentum × SI follow-up.

    Main questions:

    1. Does each locked cell have elevated event risk?
    2. Does the elevation remain against the same momentum decile?
    3. Does it remain within the same event-risk regime?
    4. For 9x10, is the effect actually driven by extreme SI changes?
    """

    frame = _prepare_frame(context.test)

    result = ExperimentResult(
        name="momentum_si_cell_context",
        description=(
            "Fördjupad kontroll av fem låsta momentum × SI-celler. "
            "Testar först kontroll mot samma momentumdecil och därefter "
            "samma momentumdecil + event-riskregim. För 9x10 analyseras "
            "dessutom faktisk SI-förändring för att skilja decileffekt "
            "från extrema SI-hopp."
        ),
    )

    # ------------------------------------------------------------------
    # 1. Main locked-cell table
    # ------------------------------------------------------------------

    result.add_table(
        "cell_context",
        pd.DataFrame(
            _cell_context_rows(
                frame,
                focus_cells=focus_cells,
            )
        ),
    )

    # ------------------------------------------------------------------
    # 2. Conditional event-risk control
    # ------------------------------------------------------------------

    conditional = _conditional_rows(
        frame,
        focus_cells=focus_cells,
    )

    result.add_table(
        "momentum_event_risk_control",
        pd.DataFrame(conditional),
    )

    # ------------------------------------------------------------------
    # 3. 9x10 mechanism: actual SI change
    # ------------------------------------------------------------------

    result.add_table(
        "nine_by_ten_si_change",
        pd.DataFrame(
            _nine_by_ten_si_change_rows(
                frame
            )
        ),
    )

    # ------------------------------------------------------------------
    # 4. Full momentum-9 SI surface
    # ------------------------------------------------------------------

    result.add_table(
        "momentum_9_si_surface",
        pd.DataFrame(
            _nine_by_ten_vs_si_level_rows(
                frame
            )
        ),
    )

    # ------------------------------------------------------------------
    # 5. Optional context columns
    # ------------------------------------------------------------------

    if context_columns:
        context_rows: list[dict] = []

        for momentum_decile, si_decile in focus_cells:
            local = frame.loc[
                _cell_mask(
                    frame,
                    momentum_decile,
                    si_decile,
                )
            ].copy()

            row = {
                "momentum_decile": momentum_decile,
                "si_decile": si_decile,
                "cell": (
                    f"{momentum_decile}x"
                    f"{si_decile}"
                ),
                "n": int(len(local)),
            }

            for column in context_columns:
                row[f"{column}_mean"] = _numeric_mean(
                    local,
                    column,
                )
                row[f"{column}_median"] = _numeric_median(
                    local,
                    column,
                )

            for horizon in horizons:
                column = (
                    f"forward_return_{horizon}d"
                )

                if column in local.columns:
                    row[f"{column}_mean"] = _numeric_mean(
                        local,
                        column,
                    )
                    row[f"{column}_median"] = _numeric_median(
                        local,
                        column,
                    )

            context_rows.append(row)

        result.add_table(
            "cell_price_context",
            pd.DataFrame(context_rows),
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    result.add_metric(
        "test_rows",
        int(len(frame)),
    )

    result.add_metric(
        "focus_cell_count",
        int(len(focus_cells)),
    )

    result.add_metric(
        "focus_cells",
        [
            f"{momentum}x{si}"
            for momentum, si in focus_cells
        ],
    )

    result.add_metric(
        "event_threshold",
        EVENT_THRESHOLD,
    )

    result.add_metric(
        "event_risk_bands",
        [
            name
            for name, _, _ in EVENT_RISK_BANDS
        ],
    )

    result.add_metric(
        "mechanism_focus",
        "9x10",
    )

    result.add_metric(
        "si_change_extreme_definition",
        "top 20% of positive SI changes within the analysed frame",
    )

    return result
