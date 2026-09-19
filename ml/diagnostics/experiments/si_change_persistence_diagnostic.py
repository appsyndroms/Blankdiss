from __future__ import annotations

from ml.diagnostics.framework import (
    DiagnosticExperiment,
)
from ml.diagnostics.framework.si_additional import (
    run_si_change_persistence,
)


class SIChangePersistenceExperiment(
    DiagnosticExperiment
):
    name = "si_change_persistence"

    description = (
        "Testar om stora ökningar i short interest "
        "följs av fortsatt förändring i blankningen "
        "och hur detta hänger ihop med efterföljande "
        "kursutveckling."
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
        return run_si_change_persistence(
            context,
            change_cutoffs=self.change_cutoffs,
            horizons=self.horizons,
        )
