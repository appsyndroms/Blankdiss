from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment
from ml.diagnostics.framework.si_event_risk_locked_oos_confirmation import (
    run_locked_oos_confirmation,
)


class SIEventRiskLockedOOSConfirmationExperiment(
    DiagnosticExperiment
):
    name = "si_event_risk_locked_oos_confirmation"
    description = (
        "Låst OOS-bekräftelse av SI × event-risk."
    )

    def analyze_window(
        self,
        context,
    ):
        return run_locked_oos_confirmation(
            context
        )
