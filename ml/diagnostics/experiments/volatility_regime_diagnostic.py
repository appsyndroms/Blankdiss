from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment


class VolatilityRegimeExperiment(
    DiagnosticExperiment
):
    name = "volatility_regime"

    description = (
        "Testar om en volatility-regime-gate förbättrar "
        "OOS-rankning jämfört med fasta modeller."
    )

    volatility_column = (
        "price_volatility_20d"
    )

    strategies = (
        "fi_only",
        "fi_plus_volatility",
        "gate_validation",
        "gate_vol_mid_high",
    )

    def analyze_window(self, context):
        return self.run_volatility_regime(
            context,
            volatility_column=(
                self.volatility_column
            ),
            strategies=self.strategies,
        )
