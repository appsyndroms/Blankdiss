from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment
from ml.diagnostics.framework.si_event_risk_locked_robustness import (
    run_locked_robustness,
)


class SIEventRiskLockedRobustnessExperiment(
    DiagnosticExperiment
):
    name = "si_event_risk_locked_robustness"
    description = (
        "Robusthetskontroll av låst "
        "SI × event-risk-signal."
    )

    def analyze_window(
        self,
        context,
    ):
        return run_locked_robustness(
            context
        )
