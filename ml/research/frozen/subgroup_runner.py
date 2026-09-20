from __future__ import annotations
import json
import os
from pathlib import Path
import yaml
from ml.research.discovery.config import DiscoveryConfig
from ml.research.discovery.engine import (
    build_candidates,
    prepare_data,
)
from ml.research.frozen.subgroup_analysis import (
    run_subgroup_analysis,
)
ROOT = Path(__file__).resolve().parents[3]
HYPOTHESIS_PATH = (
    ROOT
    / "ml"
    / "research"
    / "frozen"
    / "hypothesis.yml"
)
OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "ml"
    / "research"
    / "frozen"
    / "subgroups"
)
def load_hypothesis() -> dict:
    with HYPOTHESIS_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return yaml.safe_load(handle)
def load_discovery_config(
    discovery_end_date: str,
) -> DiscoveryConfig:
    raw_config = os.environ.get(
        "DISCOVERY_CONFIG"
    )
    if not raw_config:
        raise RuntimeError(
            "DISCOVERY_CONFIG saknas i miljön. "
            "Subgroup analysis behöver samma discovery-konfiguration "
            "som användes för att skapa kandidaten."
        )
    config_data = yaml.safe_load(raw_config)
    if not isinstance(config_data, dict):
        raise RuntimeError(
            "DISCOVERY_CONFIG kunde inte tolkas som ett YAML-objekt."
        )
    config_data = dict(config_data)
    # Den frysta hypotesen anger exakt vilken discovery-period
    # som låg till grund för kandidaten.
    config_data["discovery_end_date"] = (
        discovery_end_date
    )
    return DiscoveryConfig(
        **config_data
    )
def main() -> None:
    hypothesis = load_hypothesis()
    candidate_cfg = hypothesis[
        "hypothesis"
    ]["candidate"]
    evaluation_cfg = hypothesis[
        "hypothesis"
    ]["evaluation"]
    source_cfg = hypothesis[
        "hypothesis"
    ]["source"]
    discovery_config = load_discovery_config(
        discovery_end_date=source_cfg[
            "discovery_end_date"
        ],
    )
    # Important:
    # The frozen OOS period must remain available.
    data = prepare_data(
        discovery_config,
        apply_discovery_end=False,
    )
    candidates = build_candidates(
        discovery_config,
    )
    candidate_id = candidate_cfg[
        "candidate_id"
    ]
    matches = [
        candidate
        for candidate in candidates
        if candidate.candidate_id == candidate_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one candidate "
            f"'{candidate_id}', found {len(matches)}."
        )
    candidate = matches[0]
    results = run_subgroup_analysis(
        data=data,
        candidate=candidate,
        evaluation_start=evaluation_cfg[
            "start_date"
        ],
        evaluation_end=evaluation_cfg[
            "end_date"
        ],
        output_dir=OUTPUT_DIR,
    )
    metadata = {
        "candidate_id": candidate_id,
        "target_name": candidate_cfg[
            "target_name"
        ],
        "signal_name": candidate_cfg[
            "signal_name"
        ],
        "signal_tail": candidate_cfg[
            "signal_tail"
        ],
        "stress_feature": candidate_cfg[
            "stress_feature"
        ],
        "stress_tail": candidate_cfg[
            "stress_tail"
        ],
        "stress_direction": candidate_cfg[
            "stress_direction"
        ],
        "discovery_end_date": source_cfg[
            "discovery_end_date"
        ],
        "evaluation_start": evaluation_cfg[
            "start_date"
        ],
        "evaluation_end": evaluation_cfg[
            "end_date"
        ],
        "analysis": [
            "period",
            "sector",
            "market_regime",
        ],
        "method": (
            "Diagnostic subgroup analysis of the frozen OOS "
            "hypothesis. No thresholds or candidate parameters "
            "are optimized."
        ),
    }
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
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
        f"Candidate: {candidate_id}"
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
