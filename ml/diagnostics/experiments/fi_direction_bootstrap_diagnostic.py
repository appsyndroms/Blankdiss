from __future__ import annotations

from ml.diagnostics.framework import DiagnosticExperiment
from ml.diagnostics.framework.fi_direction import (
    run_incremental_fi_bootstrap,
)


class FIDirectionBootstrapExperiment(
    DiagnosticExperiment
):
    name = "fi_direction_bootstrap"

    description = (
        "Paired bootstrap av inkrementell FI-information "
        "ovanpå event-risk."
    )

    bootstrap_iterations = 2_000

    tail_fractions = (
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
    )

    fi_columns = (
        "short_interest",
        "short_interest_change",
        "short_interest_pct",
        "short_interest_delta",
        "short_interest_rank",
        "short_interest_zscore",
        "short_interest_acceleration",
        "short_interest_days",
        "short_interest_ratio",
        "short_interest_change_5d",
        "short_interest_change_20d",
        "short_interest_change_60d",
        "short_interest_trend",
        "short_interest_volatility",
    )

    def analyze_window(self, context):
        return run_incremental_fi_bootstrap(
            context,
            fi_columns=self.fi_columns,
            tail_fractions=self.tail_fractions,
            bootstrap_iterations=self.bootstrap_iterations,
        )
