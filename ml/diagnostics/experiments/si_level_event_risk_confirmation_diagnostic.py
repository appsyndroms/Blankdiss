from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment


class SILevelEventRiskConfirmationExperiment(
    DiagnosticExperiment
):
    name = "si_level_event_risk_confirmation"

    description = (
        "Bekräftar om short-interest-nivå tillför "
        "information inom extrem event-risk."
    )

    risk_cutoffs = (
        0.10,
        0.05,
        0.02,
    )

    short_interest_cutoff = 0.20

    def analyze_window(self, context):
        return self.run_si_level_confirmation(
            context,
            risk_cutoffs=self.risk_cutoffs,
            short_interest_cutoff=self.short_interest_cutoff,
        )
