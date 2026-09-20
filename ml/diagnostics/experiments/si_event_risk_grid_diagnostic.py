from __future__ import annotations
from ml.diagnostics.framework import (
    DiagnosticExperiment,
)
from ml.diagnostics.framework.event_risk_grid import (
    run_event_risk_grid,
)
class SIEventRiskGridExperiment(
    DiagnosticExperiment
):
    name = "si_event_risk_grid"
    description = (
        "Discovery-grid för att identifiera robusta "
        "interaktioner mellan short-interest-"
        "förändring och event-risk över flera "
        "horisonter, risknivåer och nedgångsmål. "
        "Resultaten ska användas för hypotesbildning "
        "och inte blandas ihop med låst uppföljning."
    )
    def analyze_window(
        self,
        context,
    ):
        return run_event_risk_grid(
            context
        )
