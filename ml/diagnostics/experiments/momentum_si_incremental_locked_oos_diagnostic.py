from __future__ import annotations
from ml.diagnostics.framework import DiagnosticExperiment
from ml.diagnostics.framework.momentum_si import (
    run_momentum_si_incremental_locked_oos,
)
class MomentumSIIncrementalLockedOOSExperiment(
    DiagnosticExperiment
):
    name = "momentum_si_incremental_locked_oos"
    description = (
        "Låser hypotesen hög momentum + mycket hög SI "
        "+ ytterligare SI-ökning och testar den med "
        "fyra grupper i 2025 lock-period och helt "
        "orörd 2026 OOS."
    )
    def analyze_window(self, context):
        return run_momentum_si_incremental_locked_oos(
            context,
        )
