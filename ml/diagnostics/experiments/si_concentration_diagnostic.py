from __future__ import annotations

from ml.diagnostics.framework import (
    DiagnosticExperiment,
)
from ml.diagnostics.framework.si_additional import (
    run_si_concentration,
)


class SIConcentrationExperiment(
    DiagnosticExperiment
):
    name = "si_concentration"

    description = (
        "Testar om stora förändringar i short interest "
        "ger olika signal beroende på hur koncentrerad "
        "blankningen är."
    )

    concentration_columns = (
        "short_interest_pct",
    )

    concentration_quantile = 0.80

    horizons = (
        1,
        3,
        5,
        10,
        20,
    )

    change_cutoff = 0.20

    def analyze_window(self, context):
        return run_si_concentration(
            context,
            concentration_columns=self.concentration_columns,
            concentration_quantile=self.concentration_quantile,
            horizons=self.horizons,
            change_cutoff=self.change_cutoff,
        )
