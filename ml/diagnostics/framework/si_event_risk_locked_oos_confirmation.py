from __future__ import annotations

import pandas as pd

from .base import ExperimentResult
from .event_risk_grid import (
    LOCKED_DOWNSIDE_TARGET,
    LOCKED_RISK_CUTOFF,
    LOCKED_SI_CHANGE_CUTOFF,
    _evaluate_locked_configuration,
)


LOCKED_HORIZON = 1
LOCKED_EVENT_THRESHOLD = 0.07


def run_locked_oos_confirmation(
    context,
) -> ExperimentResult:
    """
    Förutbestämd OOS-bekräftelse av SI × event-risk-
    interaktionen.

    Alla hypotesparametrar är låsta före testperioden.
    Event-risk-modellen väljs på train/validation och
    kalibreras på pretest. Testperioden används endast
    för slutlig utvärdering.
    """

    row = _evaluate_locked_configuration(
        context=context,
        horizon=LOCKED_HORIZON,
        event_threshold=LOCKED_EVENT_THRESHOLD,
    )

    if row is None:
        raise ValueError(
            "SI × event-risk OOS-bekräftelsen "
            "kunde inte beräknas för den aktuella "
            "walk-forward-perioden."
        )

    result = ExperimentResult(
        name="si_event_risk_locked_oos_confirmation",
        description=(
            "Förutbestämd OOS-bekräftelse av "
            "interaktionen mellan short-interest-"
            "förändring och event-risk utan "
            "parameteroptimering på testperioden."
        ),
    )

    result.add_metric(
        "analysis_type",
        "locked_oos_confirmation",
    )

    result.add_metric(
        "parameter_selection",
        "predeclared",
    )

    result.add_metric(
        "is_discovery_grid",
        False,
    )

    result.add_metric(
        "horizon_days",
        LOCKED_HORIZON,
    )

    result.add_metric(
        "event_threshold",
        LOCKED_EVENT_THRESHOLD,
    )

    result.add_metric(
        "downside_target",
        LOCKED_DOWNSIDE_TARGET,
    )

    result.add_metric(
        "risk_cutoff",
        LOCKED_RISK_CUTOFF,
    )

    result.add_metric(
        "si_change_cutoff",
        LOCKED_SI_CHANGE_CUTOFF,
    )

    result.add_metric(
        "interaction",
        row["interaction"],
    )

    result.add_metric(
        "interaction_ci_low",
        row["interaction_ci_low"],
    )

    result.add_metric(
        "interaction_ci_high",
        row["interaction_ci_high"],
    )

    result.add_metric(
        "interaction_p_positive",
        row["interaction_p_positive"],
    )

    result.add_table(
        "oos_confirmation",
        pd.DataFrame([row]),
    )

    return result
