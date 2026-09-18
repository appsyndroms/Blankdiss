from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment


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
        "max_individual_position_pct",
        "max_position_share_pct",
        "active_holders",
    )

    concentration_quantile = 0.80

    change_cutoff = 0.10

    horizons = (
        1,
        3,
        5,
        10,
        20,
    )

    def analyze_window(self, context):
        from ml.diagnostics.framework.si_additional import (
            run_si_concentration,
        )

        return run_si_concentration(
            context,
            concentration_columns=(
                self.concentration_columns
            ),
            concentration_quantile=(
                self.concentration_quantile
            ),
            horizons=self.horizons,
            change_cutoff=self.change_cutoff,
        )
