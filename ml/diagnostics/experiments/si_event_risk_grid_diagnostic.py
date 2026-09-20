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
        "Systematiskt testar om effekten av "
        "short-interest-förändring är starkare "
        "vid extrem event-risk över flera "
        "horisonter och nedgångsmål."
    )
    def analyze_window(
        self,
        context,
    ):
        return run_event_risk_grid(
            context
        )
