from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment


class SIPriceDynamicsExperiment(
    DiagnosticExperiment
):
    name = "si_price_dynamics"

    description = (
        "Testar hur priset utvecklas före och efter "
        "förändringar i short interest, inklusive "
        "event-risk och kontroll för tidigare prisrörelse."
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

    event_risk_cutoffs = (
        0.20,
        0.10,
        0.05,
    )

    prior_return_columns = (
        "price_return_5d",
        "price_return_20d",
        "price_return_60d",
    )

    def analyze_window(self, context):
        return self.run_si_price_dynamics(
            context,
            horizons=self.horizons,
            si_change_cutoffs=self.si_change_cutoffs,
            event_risk_cutoffs=self.event_risk_cutoffs,
            prior_return_columns=self.prior_return_columns,
        )
