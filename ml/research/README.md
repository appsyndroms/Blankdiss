# Blankdiss Research Engine

`ml/research/` är Blankdiss forskningsmotor.

Målet är att göra forskningsloopar snabba:

    Hypotes
        ↓
    YAML research spec
        ↓
    SCAN
        ↓
    intressant signal?
      ↙       ↘
    nej       ja
              ↓
            DEEP
              ↓
          resultat
              ↓
        nästa hypotes

En ny normal forskningsfråga ska därför inte kräva:

- ny Python-wrapper
- ny registry-post
- ny diagnostic-klass
- ny runner
- ny workflow-konfiguration

I stället ska frågan normalt beskrivas som en YAML-spec.

---

## Arkitektur

Den avsedda strukturen är:

    ml/
    ├── config.py
    ├── dataset.py
    ├── models.py
    ├── walk_forward.py
    │
    └── research/
        ├── bootstrap.py
        ├── cache.py
        ├── engine.py
        ├── session.py
        ├── signals.py
        ├── spec.py
        ├── runner.py
        │
        ├── specs/
        │   ├── ...
        │
        └── custom/
            └── ...

### Ansvar

| Fil | Ansvar |
|---|---|
| `spec.py` | Läser och validerar research specs |
| `session.py` | Bygger en gemensam research-session |
| `cache.py` | Förbereder data som kan återanvändas |
| `signals.py` | Gemensamma signaldefinitioner |
| `engine.py` | Kör generiska analyser |
| `bootstrap.py` | Bootstrap/CI för DEEP-analyser |
| `runner.py` | CLI och batch-körning |
| `specs/*.yaml` | Själva forskningsfrågorna |
| `custom/` | Endast analyser som inte kan uttryckas generiskt |

---

# Research specs

En research spec beskriver **vad** som ska undersökas.

Exempel:

```yaml
id: si_event_risk

mode: scan

signals:
  - name: short_interest_change
    direction: high
    fractions: [0.01, 0.05, 0.10]

  - name: price_volatility_20d
    direction: high
    fractions: [0.10, 0.20]

target:
  name: down_5pct_5d

analysis:
  type: interaction
