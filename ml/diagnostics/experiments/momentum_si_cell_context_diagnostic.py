from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment
from ml.diagnostics.framework.momentum_si import (
    run_momentum_si_cell_context,
)


class MomentumSICellContextExperiment(
    DiagnosticExperiment
):
    name = "momentum_si_cell_context"

    description = (
        "Testar om de förut identifierade momentum × SI-cellerna "
        "har en särskild pris- och riskkontext."
    )

    focus_cells = (
        (1, 3),
        (10, 6),
        (6, 9),
        (9, 10),
        (7, 10),
    )

    horizons = (
        1,
        3,
        5,
        10,
        20,
    )

    context_columns = (
        "price_momentum_20d",
        "price_momentum_60d",
        "price_volatility_20d",
        "price_distance_from_20d_high",
        "price_distance_from_60d_high",
        "short_interest_pct",
    )

    def analyze_window(self, context):
        return run_momentum_si_cell_context(
            context,
            focus_cells=self.focus_cells,
            horizons=self.horizons,
            context_columns=self.context_columns,
        )
