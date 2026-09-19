from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment
from ml.diagnostics.framework.volatility import (
    run_descriptive_volatility_analysis,
)


class VolatilityDiagnosticsExperiment(
    DiagnosticExperiment
):
    name = "volatility_diagnostics"

    description = (
        "Deskriptiv analys av volatilitet, FI och "
        "framtida downside."
    )

    targets = (
        "down_3pct_5d",
        "down_5pct_5d",
        "down_7pct_5d",
        "down_10pct_5d",
    )

    volatility_column = "price_volatility_20d"

    def analyze_window(self, context):
        return run_descriptive_volatility_analysis(
            context,
            volatility_column=self.volatility_column,
            targets=self.targets,
        )
