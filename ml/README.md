# Blankdiss ML

Detta är den övergripande dokumentationen för Blankdiss ML-system.

ML-systemet består av:

```text
ml/
├── config.py
├── dataset.py
├── walk_forward.py
│
├── research/
│   ├── spec.py
│   ├── session.py
│   ├── cache.py
│   ├── signals.py
│   ├── engine.py
│   ├── runner.py
│   ├── reporting.py
│   ├── bootstrap.py
│   ├── specs/
│   └── custom/
│
└── diagnostics/
    ├── framework/
    │   ├── base.py
    │   ├── context.py
    │   ├── metrics.py
    │   ├── reporting.py
    │   ├── runner.py
    │   └── stratification.py
    │
    └── experiments/
        └── *_diagnostic.py
