from __future__ import annotations
from ml.diagnostics.framework import (
    DiagnosticExperiment,
)
from ml.diagnostics.framework.event_risk_grid import (
    run_locked_event_risk_follow_up,
)
class SIEventRiskLockedFollowUpExperiment(
    DiagnosticExperiment
):
    name = "si_event_risk_locked_follow_up"
    description = (
        "Låst uppföljning av hypotesen att stor "
        "positiv short-interest-förändring förstärker "
        "risken för ett snabbt kraftigt kursfall när "
        "aktien redan befinner sig i hög event-risk."
    )
    def analyze_window(
        self,
        context,
    ):
        return run_locked_event_risk_follow_up(
            context
        )
