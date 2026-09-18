from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment


class VolatilityHorizonCombinationLogisticExperiment(
    DiagnosticExperiment
):
    name = "volatility_horizon_combination_logistic"

    description = (
        "Testar kombinationer av 20d/60d-volatilitet "
        "och dess relativa förändring."
    )

    feature_sets = {
        "volatility_20d": (
            "price_volatility_20d",
        ),
        "volatility_20d_plus_60d": (
            "price_volatility_20d",
            "volatility_60d",
        ),
        "volatility_20d_plus_relative": (
            "price_volatility_20d",
            "volatility_relative_20d_60d",
        ),
        "volatility_20d_plus_change": (
            "price_volatility_20d",
            "volatility_change_20d_60d",
        ),
        "volatility_all": (
            "price_volatility_20d",
            "volatility_60d",
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
