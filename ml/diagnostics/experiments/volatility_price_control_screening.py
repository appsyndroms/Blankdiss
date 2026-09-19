from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment
from ml.diagnostics.framework.volatility import (
    run_logistic_screen,
)


class VolatilityPriceControlExperiment(
    DiagnosticExperiment
):
    name = "volatility_price_control"

    description = (
        "Kontrollerar om volatility_20d är en proxy "
        "för annan historisk prisinformation."
    )

    feature_sets = {
        "volatility_only": (
            "price_volatility_20d",
        ),
        "volatility_plus_return_5d": (
            "price_volatility_20d",
            "price_return_5d",
        ),
        "volatility_plus_return_20d": (
            "price_volatility_20d",
            "price_return_20d",
        ),
        "volatility_plus_return_60d": (
            "price_volatility_20d",
            "price_return_60d",
        ),
        "volatility_plus_all_returns": (
            "price_volatility_20d",
            "price_return_5d",
            "price_return_20d",
            "price_return_60d",
        ),
        "volatility_plus_all_other_price": (
            "price_volatility_20d",
            "price_return_5d",
            "price_return_20d",
            "price_return_60d",
            "price_distance_from_20d_high",
            "price_distance_from_60d_high",
        ),
    }

    def analyze_window(self, context):
        return run_logistic_screen(
            context,
            feature_sets=self.feature_sets,
            target="down_5pct_5d",
        )
