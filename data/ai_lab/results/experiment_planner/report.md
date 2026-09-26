# AI Lab experiment

## Experiment

**ID:** `experiment_planner`

**Type:** `experiment_planner`

**Description:** Controlled experiment proposal generated from the current Blankdiss research state.

## Status

**PASS**

## Execution

- Started: `2026-09-26T05:16:00.527298+00:00`
- Finished: `2026-09-26T05:16:00.527351+00:00`

## Result

```json
{
  "planner_version": 2,
  "status": "proposal_created",
  "selection": {
    "source_spec": "fi_volatility_interaction_scan",
    "stage": "discovery",
    "locked": false,
    "question": "Är downside-event-risk särskilt hög när short interest och marknadsvolatilitet samtidigt ligger högt?\n"
  },
  "parameters": {},
  "constraints": [
    "Do not modify the research specification.",
    "Do not introduce arbitrary parameters.",
    "Do not optimize thresholds on the test set.",
    "Do not select the test data based on observed outcomes.",
    "Do not declare the research hypothesis confirmed from the planner output.",
    "Do not automatically select a locked prospective-confirmation specification."
  ],
  "research_context": {
    "spec_count": 10,
    "research_run_count": 5,
    "completed_spec_count": 5,
    "open_question_count": 6,
    "locked_spec_count": 1
  },
  "warnings": [
    "Locked prospective-confirmation specification(s) exist but are not automatically selected by the planner.",
    "Migration specifications are visible in the research state but are not treated as current research evidence or automatic planning candidates."
  ]
}
```
