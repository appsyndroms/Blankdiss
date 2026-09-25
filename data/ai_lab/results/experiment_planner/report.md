# AI Lab experiment

## Experiment

**ID:** `experiment_planner`

**Type:** `experiment_planner`

**Description:** Controlled experiment proposal generated from the current Blankdiss research state.

## Status

**PASS**

## Execution

- Started: `2026-09-25T20:47:21.149359+00:00`
- Finished: `2026-09-25T20:47:21.149409+00:00`

## Result

```json
{
  "planner_version": 2,
  "status": "proposal_created",
  "selection": {
    "source_spec": "direction_signal_scan",
    "stage": "discovery",
    "locked": false,
    "question": "Har signalerna en riktad relation till framtida upp- respektive nedgång, eller beskriver de huvudsakligen generell event-risk?\n"
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
    "research_run_count": 2,
    "completed_spec_count": 2,
    "open_question_count": 6,
    "locked_spec_count": 1
  },
  "warnings": [
    "Locked prospective-confirmation specification(s) exist but are not automatically selected by the planner.",
    "Migration specifications are visible in the research state but are not treated as current research evidence or automatic planning candidates."
  ]
}
```
