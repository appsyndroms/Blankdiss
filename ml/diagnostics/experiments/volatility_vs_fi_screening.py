from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment


class VolatilityVsFIExperiment(
    DiagnosticExperiment
):
    name = "volatility_vs_fi"

    description = (
        "Jämför FI-only, volatility-only och "
        "FI + volatility."
    )

    feature_sets = {
        "fi_only": "fi",
        "volatility_only": "volatility",
        "fi_plus_volatility_20d": "fi_plus_volatility",
    }

    def analyze_window(self, context):
        return self.run_feature_set_comparison(
            context,
            feature_sets=self.feature_sets,
            target="down_5pct_5d",
        )
