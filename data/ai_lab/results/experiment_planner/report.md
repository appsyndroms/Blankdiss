# AI Lab experiment

## Experiment

**ID:** `experiment_planner`

**Type:** `experiment_planner`

**Description:** Controlled experiment proposal generated from the current Blankdiss research state.

## Status

**PASS**

## Execution

- Started: `2026-09-25T19:44:48.871680+00:00`
- Finished: `2026-09-25T19:44:48.871713+00:00`

## Result

```json
{
  "planner_version": 1,
  "research_state_version": 1,
  "status": "proposal_created",
  "selection": {
    "source_spec": "momentum_si_prospective_confirmation",
    "stage": "prospective_confirmation",
    "locked": true
  },
  "proposal": {
    "proposal_version": 1,
    "research_question": "Bekräftar nya, tidigare inte använda data att förändring i short interest tillför downside-risk inom de 20 procent mest negativa 5-dagars momentumobservationerna?\n",
    "hypothesis": {
      "baseline": "negative_5d_momentum_20pct",
      "incremental_signal": "short_interest_change_top_2_5pct",
      "primary_target": "down_10pct_5d",
      "horizon_days": 5
    },
    "stage": "prospective_confirmation",
    "source_specs": [
      "momentum_si_prospective_confirmation"
    ],
    "data_requirements": {
      "windows": [
        "window_2"
      ],
      "splits": [
        "test"
      ],
      "required_signals": [
        {
          "name": "price_momentum_5d",
          "direction": "lower",
          "bins": [
            0.2
          ]
        },
        {
          "name": "short_interest_change",
          "direction": "upper",
          "bins": [
            0.025
          ]
        }
      ],
      "targets": [
        "down_10pct_5d",
        "down_5pct_5d",
        "down_7pct_5d"
      ]
    },
    "parameters": {
      "source_spec": "momentum_si_prospective_confirmation",
      "locked": true
    },
    "validation": {
      "analysis": {
        "type": "regime_comparison",
        "bootstrap": true,
        "bootstrap_iterations": 2000
      },
      "declared_by_source_spec": true,
      "research_run_count_visible": 0
    },
    "constraints": [
      "Do not modify research specifications.",
      "Do not execute arbitrary code.",
      "Do not optimize parameters against test data.",
      "Do not declare a hypothesis confirmed.",
      "Use the selected specification as declared; do not silently change its parameters.",
      "Treat confirmation parameters as locked.",
      "Do not search additional bins.",
      "Do not optimize thresholds on confirmation data.",
      "Do not switch the primary endpoint after test start.",
      "Do not use confirmation data for parameter selection."
    ],
    "rationale": "A locked prospective confirmation specification already exists. The planner preserves its predefined parameters and endpoint and proposes no optimization.",
    "expected_observation": "Determine what the declared research specification actually observes without changing its design.",
    "status": "proposed"
  },
  "warnings": [
    "No research-engine run manifests are visible in the current research-state snapshot.",
    "3 migration-stage specification(s) are visible but are not treated as new evidence."
  ],
  "research_context": {
    "spec_count": 10,
    "locked_spec_count": 1,
    "research_run_count": 0
  }
}
```
