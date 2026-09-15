"""Entry point för Blankdiss signal-backtest."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from ml.dataset import load_features
from analysis.signal_backtest.config import (
    BACKTEST_FEATURE_SETS,
    BACKTEST_TARGETS,
    TOP_FRACTIONS,
)
from analysis.signal_backtest.reporting import (
    print_experiment,
    print_header,
)
from analysis.signal_backtest.runner import (
    run_all,
)
OUTPUT_PATH = Path(
    "data/processed/ml/signal_backtest.json"
)
def main() -> None:
    """Kör signal-backtest och spara resultat."""
    print_header()
    print()
    features = load_features()
    print(
        "Features loaded: "
        f"{len(features):,}"
    )
    results = run_all(
        features,
        BACKTEST_FEATURE_SETS,
    )
    for result in results:
        print_experiment(result)
    now = datetime.now(
        timezone.utc
    )
    document = {
        "run_id": now.strftime(
            "%Y%m%dT%H%M%SZ"
        ),
        "created_at": now.isoformat(),
        "experiment": "signal_backtest",
        "top_fractions": [
            float(value)
            for value in TOP_FRACTIONS
        ],
        "feature_sets": [
            name
            for name, _
            in BACKTEST_FEATURE_SETS
        ],
        "targets": list(
            BACKTEST_TARGETS
        ),
        "results": results,
    }
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            document,
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print()
    print("================================")
    print("SIGNAL BACKTEST KLAR")
    print(
        "Resultat: "
        f"{OUTPUT_PATH}"
    )
    print("================================")
if __name__ == "__main__":
    main()
