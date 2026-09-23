from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment
from ml.diagnostics.framework.si_event_risk_b_continuous_anatomy import (
    run_b_continuous_anatomy,
)


class SIEventRiskBContinuousAnatomyExperiment(
    DiagnosticExperiment
):
    name = "si_event_risk_b_continuous_anatomy"

    description = (
        "Kontinuerlig deskriptiv analys av "
        "B-cellen, med Spearman-samband och "
        "fasta kvartiler för att undersöka "
        "om det finns en intern struktur i B."
    )

    def analyze_window(
        self,
        context,
    ):
        return run_b_continuous_anatomy(
            context
        )
