from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment
from ml.diagnostics.framework.si_event_risk_positive_momentum_context import (
    run_positive_momentum_context,
)


class SIEventRiskPositiveMomentumContextExperiment(
    DiagnosticExperiment
):
    name = (
        "si_event_risk_positive_momentum_context"
    )

    description = (
        "Mekanismanalys av hög SI-förändring "
        "inom hög event-risk och icke-negativt "
        "20-dagars momentum."
    )

    def analyze_window(
        self,
        context,
    ):
        return run_positive_momentum_context(
            context
        )
