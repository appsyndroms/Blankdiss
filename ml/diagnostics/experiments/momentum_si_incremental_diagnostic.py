from __future__ import annotations

from ml.diagnostics.framework import (
    DiagnosticExperiment,
)
from ml.diagnostics.framework.momentum_si_incremental import (
    run_momentum_si_incremental,
)


class MomentumSIIncrementalExperiment(
    DiagnosticExperiment
):
    name = "momentum_si_incremental"

    targets = (
        "down_5pct_5d",
        "down_7pct_5d",
        "down_10pct_5d",
    )

    description = (
        "Testar om SI-förändring tillför "
        "inkrementell prediktiv information utöver "
        "20-dagars momentum, samt om momentum × "
        "SI-förändring tillför ytterligare information."
    )

    def analyze_window(self, context):
        return run_momentum_si_incremental(
            context
        )
