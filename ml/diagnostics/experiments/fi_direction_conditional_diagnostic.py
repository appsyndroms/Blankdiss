from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment


class FIDirectionConditionalExperiment(
    DiagnosticExperiment
):
    name = "fi_direction_conditional"

    description = (
        "Testar om short interest skiljer DOWN från UP "
        "inom hög event-risk."
    )

    targets = (
        "down_10pct_5d",
    )

    event_tail_fractions = (
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
    )

    fi_column = "short_interest_pct"

    def analyze_window(self, context):
        return self.run_conditional_direction_analysis(
            context,
            fi_column=self.fi_column,
            event_tail_fractions=self.event_tail_fractions,
        )
