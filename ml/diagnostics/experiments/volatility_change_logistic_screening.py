from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment


class VolatilityChangeLogisticExperiment(
    DiagnosticExperiment
):
    name = "volatility_change_logistic"

    description = (
        "Jämför 20d-volatilitet med förändring "
        "mot 60d-volatilitet."
    )

    feature_sets = {
        "volatility_20d": (
            "price_volatility_20d",
        ),
        "volatility_20d_plus_change": (
            "price_volatility_20d",
            "volatility_change_20d_60d",
        ),
    }

    def analyze_window(self, context):
        return self.run_logistic_screen(
            context,
            feature_sets=self.feature_sets,
            target="down_5pct_5d",
        )
