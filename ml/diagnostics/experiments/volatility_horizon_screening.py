from __future__ import annotations

from ml.diagnostics.framework import (
    DiagnosticExperiment,
)
from ml.diagnostics.framework.volatility_horizon import (
    run_volatility_horizon,
)


class VolatilityHorizonExperiment(
    DiagnosticExperiment
):
    name = "volatility_horizon_screening"

    description = (
        "Jämför 20d-volatilitet med relativ "
        "och förändrad volatilitet mot 60d."
    )

    economic_target = "down_5pct_5d"

    def analyze_window(self, context):
        return run_volatility_horizon(
            context,
            economic_target=self.economic_target,
        )
