from __future__ import annotations

from ml.diagnostics.framework import (
    DiagnosticExperiment,
)


classSIConditionalModelComparisonExperiment(
    DiagnosticExperiment
):
    name = "si_conditional_model_comparison"

    description = (
        "Jämför SI, event-risk, kombinationen "
        "och kombinationen med tidigare avkastning "
        "som OOS-prediktorer för 5d-nedgång."
    )

    def analyze_window(self, context):
        return self.run_conditional_model_comparison(
            context
        )
