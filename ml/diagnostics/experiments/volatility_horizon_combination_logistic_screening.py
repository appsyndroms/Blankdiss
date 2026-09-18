from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment


class VolatilityHorizonExperiment(
    DiagnosticExperiment
):
    name = "volatility_horizon"

    description = (
        "Jämför absolut och relativ volatilitet "
        "över 20 och 60 dagar."
    )

    feature_sets = {
        "volatility_20d": (
            "price_volatility_20d",
        ),
        "volatility_relative_20d_60d": (
            "volatility_relative_20d_60d",
        ),
        "volatility_change_20d_60d": (
            "volatility_change_20d_60d",
        ),
        "volatility_20d_plus_relative": (
            "price_volatility_20d",
            "volatility_relative_20d_60d",
        ),
        "volatility_20d_plus_change": (
            "price_volatility_20d",
            "volatility_change_20d_60d",
        ),
        "volatility_20d_relative_plus_change": (
            "price_volatility_20d",
            "volatility_relative_20d_60d",
            "volatility_change_20d_60d",
        ),
    }

    def analyze_window(self, context):
        return self.run_logistic_screen(
            context,
            feature_sets=self.feature_sets,
            target="down_5pct_5d",
        )
