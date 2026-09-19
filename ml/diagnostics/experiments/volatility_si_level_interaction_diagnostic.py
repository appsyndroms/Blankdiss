from __future__ import annotations

from ml.diagnostics.framework import (
    DiagnosticExperiment,
)
from ml.diagnostics.framework.volatility_si_level import (
    run_volatility_si_level_interaction,
)


class VolatilitySILevelInteractionExperiment(
    DiagnosticExperiment
):
    name = "volatility_si_level_interaction"

    description = (
        "Testar om hög short interest tillför "
        "information inom extrem event-risk när "
        "volatiliteten redan är hög."
    )

    def analyze_window(self, context):
        return run_volatility_si_level_interaction(
            context
        )
