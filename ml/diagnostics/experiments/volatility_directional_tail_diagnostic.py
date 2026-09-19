from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment
from ml.diagnostics.framework.volatility import (
    run_directional_tail_analysis,
)


class VolatilityDirectionalTailExperiment(
    DiagnosticExperiment
):
    name = "volatility_directional_tail"

    description = (
        "Analyserar om volatility skiljer DOWN från UP "
        "inom event-risk tails."
    )

    event_tail_fractions = (
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
    )

    volatility_column = "price_volatility_20d"

    def analyze_window(self, context):
        return run_directional_tail_analysis(
            context,
            volatility_column=self.volatility_column,
            event_tail_fractions=self.event_tail_fractions,
        )
