from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment


class VolatilitySILevelInteractionExperiment(
    DiagnosticExperiment
):
    name = "volatility_si_level_interaction"

    description = (
        "Testar interaktionen mellan volatilitet och "
        "short-interest-nivå inom extrem event-risk."
    )

    targets = (
        "down_10pct_5d",
        "down_7pct_5d",
        "down_5pct_5d",
    )

    event_tail = 0.05
    volatility_tail = 0.20
    short_interest_tail = 0.20

    def analyze_window(self, context):
        volatility_bins = self.make_pretest_bins(
            context.test,
            "price_volatility_20d",
        )

        short_interest_bins = self.make_pretest_bins(
            context.test,
            "short_interest_pct",
        )

        return self.run_event_risk_interaction(
            context,
            x_bins=volatility_bins,
            y_bins=short_interest_bins,
            targets=self.targets,
            event_tail=self.event_tail,
        )
