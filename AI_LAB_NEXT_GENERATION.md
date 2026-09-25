# Blankdiss AI Lab — Next Generation

## Purpose

Blankdiss AI Lab is intended to become an autonomous research and experimentation
environment for systematic analysis.

The goal is not simply to run Python scripts from GitHub Actions.

The goal is to build a controlled feedback loop in which an AI can:

1. inspect the current research state
2. identify an unanswered question or promising hypothesis
3. formulate an experiment
4. generate the required experiment specification/code
5. execute the experiment
6. inspect the results
7. determine what the results actually say
8. decide what the next research question should be
9. run the next experiment
10. preserve the complete research history

The human should primarily define the research boundaries and approve important
transitions, rather than manually moving files and starting individual scripts.

---

# 1. The vision

The intended architecture is:

```text
                    ┌──────────────────────┐
                    │       AI Lab         │
                    │                      │
                    │  Research reasoning  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Experiment Spec     │
                    │                      │
                    │  hypothesis           │
                    │  data                 │
                    │  parameters           │
                    │  target               │
                    │  validation           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Experiment Runner  │
                    │                      │
                    │  controlled execution│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       Results        │
                    │                      │
                    │  metrics              │
                    │  tables               │
                    │  diagnostics          │
                    │  report               │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    AI evaluation     │
                    │                      │
                    │  What did we learn?  │
                    │  What failed?        │
                    │  What is next?       │
                    └──────────┬───────────┘
                               │
                               ▼
                         Next experiment
                               │
                               └──────────────►
