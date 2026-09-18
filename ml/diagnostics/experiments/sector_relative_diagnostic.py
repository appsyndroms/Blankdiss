from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment


class SectorRelativeReturnExperiment(
    DiagnosticExperiment
):
    name = "sector_relative_return"

    description = (
        "Testar om effekten av stora förändringar i "
        "short interest kvarstår relativt sektor och "
        "marknad."
    )

    horizons = (
        1,
        3,
        5,
        10,
        20,
    )

    si_change_cutoffs = (
        0.10,
        0.20,
        0.30,
    )

    def analyze_window(self, context):
        return self.run_sector_relative_return(
            context,
            horizons=self.horizons,
            si_change_cutoffs=self.si_change_cutoffs,
        )
