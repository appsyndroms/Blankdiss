# Blankdiss

Blankdiss är ett forskningsprojekt för att undersöka om offentlig information om bolag, blankning, prisrörelser, rapporter och andra marknadsvariabler innehåller statistiskt och ekonomiskt användbara mönster.

Projektet kombinerar:

- datainsamling
- feature engineering
- datakvalitet
- ML
- hypotesdriven research
- walk-forward evaluation
- prospective evaluation
- automatiserade analyser
- reproducerbara resultat
- AI-assisterad forskning

Målet är inte att bygga en samling fristående analyser.

Målet är att bygga en återanvändbar forskningsmotor där:

    hypotes
        ↓
    deklarativ spec
        ↓
    research
        ↓
    verification
        ↓
    OOS / prospective evaluation
        ↓
    resultat
        ↓
    nästa hypotes

kan genomföras med så lite specialkod som möjligt.

---

## Översikt

```text
                         Blankdiss
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
       Data layer                         Research layer
          │                                   │
    ┌─────┼─────┐                       ┌─────┴─────┐
    │     │     │                       │           │
   FI   Prices Market                 Research   Diagnostics
    │     │     │                       │           │
    └─────┼─────┘                       │           │
          ▼                             │           │
   Feature generation                  │           │
          │                             │           │
          ▼                             ▼           ▼
    Feature dataset              Research Engine
          │                             │
          ▼                             │
      Feature QC                       │
          │                             │
          └──────────────┬──────────────┘
                         ▼
                    Verification
                         │
                         ▼
                    OOS / Evaluation
                         │
                         ▼
                    AI / analysis
