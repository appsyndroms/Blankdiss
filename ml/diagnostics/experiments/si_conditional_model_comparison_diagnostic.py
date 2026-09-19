from __future__ import annotations

from ml.diagnostics.framework import (
    DiagnosticExperiment,
)
from ml.diagnostics.framework.conditional_models import (
    run_conditional_model_comparison,
)


class SIConditionalModelComparisonExperiment(
    DiagnosticExperiment
):
    name = "si_conditional_model_comparison"

    description = (
        "Jämför SI, event-risk, kombinationen "
        "och kombinationen med tidigare avkastning "
        "som OOS-prediktorer för 5d-nedgång."
    )

    def analyze_window(self, context):
        return run_conditional_model_comparison(
            context
        )
