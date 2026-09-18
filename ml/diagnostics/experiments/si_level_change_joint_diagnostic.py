from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment


class SILevelChangeJointExperiment(
    DiagnosticExperiment
):
    name = "si_level_change_joint"

    description = (
        "Separera effekten av hög blankningsnivå "
        "från effekten av en stor positiv förändring "
        "i short interest."
    )

    horizons = (
        1,
        3,
        5,
        10,
        20,
    )

    level_quantiles = (
        0.80,
        0.90,
    )

    change_cutoffs = (
        0.10,
        0.20,
    )

    def analyze_window(self, context):
        from ml.diagnostics.framework.si_additional import (
            run_si_level_change_joint,
        )

        return run_si_level_change_joint(
            context,
            horizons=self.horizons,
            level_quantiles=self.level_quantiles,
            change_cutoffs=self.change_cutoffs,
        )
