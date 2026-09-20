from __future__ import annotations
import json
from pathlib import Path
from ml.research.discovery.config import (
    DiscoveryConfig,
    ValidationConfig,
)
from ml.research.discovery.engine import (
    Candidate,
    prepare_data,
)
from ml.research.frozen.config import load_config
from ml.research.frozen.subgroup_analysis import (
    run_subgroup_analysis,
)
ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "ml"
    / "research"
    / "frozen"
    / "subgroups"
)
def _build_candidate(config) -> Candidate:
    candidate = config.candidate
    return Candidate(
        candidate_id=candidate.candidate_id,
        target_name=candidate.target_name,
        signal_name=candidate.signal_name,
        signal_tail=candidate.signal_tail,
        stress_feature=candidate.stress_feature,
        stress_tail=candidate.stress_tail,
        stress_direction=candidate.stress_direction,
    )
def _build_discovery_config(config) -> DiscoveryConfig:
    candidate = config.candidate
    return DiscoveryConfig(
        enabled=True,
        discovery_end_date=config.discovery_end_date,
        targets=(candidate.target_name,),
        signals=(candidate.signal_name,),
        stress_features=(candidate.stress_feature,),
        tails=(
            candidate.signal_tail,
            candidate.stress_tail,
        ),
        stress_directions={
            candidate.stress_feature: (
                candidate.stress_direction
            )
        },
        validation=ValidationConfig(),
    )
def main() -> None:
    config = load_config()
    discovery_config = _build_discovery_config(
        config
    )
    data = prepare_data(
        discovery_config,
        apply_discovery_end=False,
    )
    candidate = _build_candidate(config)
    results = run_subgroup_analysis(
        data=data,
        candidate=candidate,
        evaluation_start=(
            config.evaluation.start_date.isoformat()
        ),
        evaluation_end=(
            config.evaluation.end_date.isoformat()
        ),
        output_dir=OUTPUT_DIR,
    )
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    metadata = {
        "hypothesis_id": config.hypothesis_id,
        "discovery_run": config.discovery_run,
        "candidate_id": candidate.candidate_id,
        "target_name": candidate.target_name,
        "signal_name": candidate.signal_name,
        "signal_tail": candidate.signal_tail,
        "stress_feature": candidate.stress_feature,
        "stress_tail": candidate.stress_tail,
        "stress_direction": candidate.stress_direction,
        "discovery_end_date": (
            config.discovery_end_date.isoformat()
        ),
        "evaluation_start": (
            config.evaluation.start_date.isoformat()
        ),
        "evaluation_end": (
            config.evaluation.end_date.isoformat()
        ),
        "analysis": [
            "period",
            "sector",
            "market_regime",
        ],
        "method": (
            "Diagnostic subgroup analysis of the frozen "
            "OOS hypothesis. No thresholds or candidate "
            "parameters are optimized."
        ),
    }
    with (
        OUTPUT_DIR / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(
        "Frozen subgroup analysis completed."
    )
    print(
        f"Candidate: {candidate.candidate_id}"
    )
    for name, frame in results.items():
        print()
        print(f"=== {name} ===")
        if frame.empty:
            print(
                "No compatible data found."
            )
        else:
            print(
                frame.to_string(
                    index=False,
                )
            )
if __name__ == "__main__":
    main()
