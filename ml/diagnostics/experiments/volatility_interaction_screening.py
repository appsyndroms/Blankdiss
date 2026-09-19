from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment
from ml.diagnostics.framework.volatility import (
    run_interaction_screen,
)


class VolatilityInteractionExperiment(
    DiagnosticExperiment
):
    name = "volatility_interaction"

    description = (
        "Screenar interaktioner mellan short interest "
        "och price volatility."
    )

    base_features = (
        "price_volatility_20d",
    )

    interactions = {
        "short_interest_x_volatility": (
            "short_interest_pct",
            "price_volatility_20d",
        ),
        "short_interest_delta_x_volatility": (
            "short_interest_delta_pp",
            "price_volatility_20d",
        ),
        "short_interest_acceleration_x_volatility": (
            "short_interest_acceleration_pp",
            "price_volatility_20d",
        ),
    }

    def analyze_window(self, context):
        return run_interaction_screen(
            context,
            base_features=self.base_features,
            interactions=self.interactions,
            target="down_5pct_5d",
        )
