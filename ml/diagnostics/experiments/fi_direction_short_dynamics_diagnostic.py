from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment


classFIDirectionShortDynamicsExperiment(
    DiagnosticExperiment
):
    name = "fi_direction_short_dynamics"

    description = (
        "Testar om förändringar i short interest "
        "skiljer DOWN från UP inom 5% event-risk."
    )

    event_tail = 0.05

    change_columns = (
        "short_interest_pct_change",
        "short_interest_pct_change_pct",
    )

    positive_cutoffs = (
        0.10,
        0.20,
        0.30,
        0.40,
    )

    def analyze_window(self, context):
        return self.run_short_interest_dynamics(
            context,
            event_tail=self.event_tail,
            change_columns=self.change_columns,
            positive_cutoffs=self.positive_cutoffs,
        )
