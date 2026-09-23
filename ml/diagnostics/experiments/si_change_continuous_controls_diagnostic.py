from __future__ import annotations

from ml.diagnostics.framework import (
    DiagnosticExperiment,
)

from ml.diagnostics.framework.si_change_continuous_controls import (
    run_si_change_continuous_controls,
)


class SIChangeContinuousControlsExperiment(
    DiagnosticExperiment
):
    name = "si_change_continuous_controls"

    description = (
        "Testar om förändringen i short interest har "
        "inkrementell information om downside-risk "
        "efter kontroll för momentum och SI-nivå."
    )

    def analyze_window(
        self,
        context,
    ):
        return run_si_change_continuous_controls(
            context,
        )
