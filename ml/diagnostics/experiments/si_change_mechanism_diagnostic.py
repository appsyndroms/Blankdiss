from __future__ import annotations

from ml.diagnostics.framework import (
    DiagnosticExperiment,
)

from ml.diagnostics.framework.si_change_mechanism import (
    run_si_change_mechanism,
)


class SIChangeMechanismExperiment(
    DiagnosticExperiment
):
    name = "si_change_mechanism"

    description = (
        "Deskriptiv analys av hur förändringar i "
        "short interest relaterar till efterföljande "
        "downside-risk, med uppdelning efter momentum "
        "och SI-nivå."
    )

    def analyze_window(
        self,
        context,
    ):
        return run_si_change_mechanism(
            context
        )
