from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment
from ml.diagnostics.framework.event_risk import (
    run_event_risk_interaction,
)


class FIShortInterestEventRiskInteractionExperiment(
    DiagnosticExperiment
):
    name = "fi_short_interest_event_risk_interaction"

    description = (
        "Testar om effekten av short-interest-förändring "
        "är starkare vid extrem event-risk."
    )

    event_risk_cutoffs = (
        0.20,
        0.10,
        0.05,
        0.025,
        0.01,
    )

    positive_change_cutoff = 0.20

    def analyze_window(self, context):
        return run_event_risk_interaction(
            context,
            risk_cutoffs=self.event_risk_cutoffs,
            positive_change_cutoff=self.positive_change_cutoff,
        )
