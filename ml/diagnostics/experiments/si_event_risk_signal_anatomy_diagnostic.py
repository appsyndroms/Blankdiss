from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment
from ml.diagnostics.framework.si_event_risk_signal_anatomy import (
    run_signal_anatomy,
)


class SIEventRiskSignalAnatomyExperiment(
    DiagnosticExperiment
):
    name = "si_event_risk_signal_anatomy"

    description = (
        "Mekanismanalys av den låsta "
        "SI × event-risk-signalen."
    )

    def analyze_window(
        self,
        context,
    ):
        return run_signal_anatomy(
            context
        )
