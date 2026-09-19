from __future__ import annotations

from ml.diagnostics.framework import (
    DiagnosticExperiment,
)
from ml.diagnostics.framework.si_additional import (
    run_si_level_change_joint,
)


class SILevelChangeJointExperiment(
    DiagnosticExperiment
):
    name = "si_level_change_joint"

    description = (
        "Separera effekten av hög blankningsnivå "
        "från effekten av en stor positiv förändring "
        "i short interest."
    )

    level_quantiles = (
        0.80,
    )

    change_cutoffs = (
        0.10,
        0.20,
        0.30,
    )

    horizons = (
        1,
        3,
        5,
        10,
        20,
    )

    def analyze_window(self, context):
        return run_si_level_change_joint(
            context,
            horizons=self.horizons,
            level_quantiles=self.level_quantiles,
            change_cutoffs=self.change_cutoffs,
        )
