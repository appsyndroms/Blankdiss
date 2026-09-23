from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment
from ml.diagnostics.framework.si_event_risk_b_outcome_anatomy import (
    run_b_outcome_anatomy,
)


class SIEventRiskBOutcomeAnatomyExperiment(
    DiagnosticExperiment
):
    name = "si_event_risk_b_outcome_anatomy"

    description = (
        "Deskriptiv analys av vad som skiljer "
        "B-observationer med olika efterföljande "
        "utfall, utan att använda utfallet för "
        "parameterselektion."
    )

    def analyze_window(
        self,
        context,
    ):
        return run_b_outcome_anatomy(
            context
        )
