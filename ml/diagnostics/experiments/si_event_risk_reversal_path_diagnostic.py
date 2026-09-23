from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment
from ml.diagnostics.framework.si_event_risk_reversal_path import (
    run_reversal_path,
)


class SIEventRiskReversalPathExperiment(
    DiagnosticExperiment
):
    name = "si_event_risk_reversal_path"

    description = (
        "Prisbananalys av B-cellen från den låsta "
        "SI × event-risk-signalen, med fokus på "
        "långsiktig svaghet, kortsiktig återhämtning "
        "och prisacceleration före signalen."
    )

    def analyze_window(
        self,
        context,
    ):
        return run_reversal_path(
            context
        )
