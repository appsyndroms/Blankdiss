from __future__ import annotations

from ml.diagnostics.framework import (
    DiagnosticExperiment,
)
from ml.diagnostics.framework.event_risk import (
    run_si_level_confirmation,
)


class SILevelEventRiskConfirmationExperiment(
    DiagnosticExperiment
):
    name = "si_level_event_risk_confirmation"

    description = (
        "Bekräftar om hög short-interest-nivå "
        "tillför information inom extrem event-risk."
    )

    event_risk_cutoffs = (
        0.20,
        0.10,
        0.05,
        0.025,
        0.01,
    )

    short_interest_cutoff = 0.20

    def analyze_window(self, context):
        return run_si_level_confirmation(
            context,
            risk_cutoffs=self.event_risk_cutoffs,
            short_interest_cutoff=self.short_interest_cutoff,
        )
