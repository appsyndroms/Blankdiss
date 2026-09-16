"""Kör Blankdiss ekonomiska OOS-backtests."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from analysis.signal_backtest.economic import (
    run_all_economic_backtests,
)
from analysis.signal_backtest.reporting import (
    print_economic_results,
)
from ml.config import ML_OUTPUT_DIR
ECONOMIC_RESULTS_PATH = (
    ML_OUTPUT_DIR
    / "economic_results.json"
)
def write_results(
    results: list[dict],
) -> None:
    """Skriv ekonomiska resultat som aktuell snapshot."""
    document = {
        "created_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "experiments": results,
        "experiment_count": len(
            results
        ),
    }
    ECONOMIC_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with ECONOMIC_RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            document,
            handle,
            ensure_ascii=False,
            indent=2,
        )
def main() -> None:
    print(
        "Blankdiss economic backtest: "
        "startar."
    )
    results = (
        run_all_economic_backtests()
    )
    write_results(
        results
    )
    print_economic_results(
        results
    )
    print()
    print(
        "Ekonomiskt backtest klart."
    )
    print(
        f"Experiment: "
        f"{len(results)}"
    )
    print(
        f"Resultat: "
        f"{ECONOMIC_RESULTS_PATH}"
    )
if __name__ == "__main__":
    main()
