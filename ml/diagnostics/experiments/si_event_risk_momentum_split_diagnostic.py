from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment
from ml.diagnostics.framework.si_event_risk_momentum_split import (
    run_momentum_split,
)


class SIEventRiskMomentumSplitExperiment(
    DiagnosticExperiment
):
    name = "si_event_risk_momentum_split"

    description = (
        "2×2-mekanismanalys av den låsta "
        "SI × event-risk-signalen mot "
        "tidigare momentum."
    )

    def analyze_window(
        self,
        context,
    ):
        return run_momentum_split(
            context
        )
