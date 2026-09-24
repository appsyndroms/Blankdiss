Blankdiss

Blankdiss är ett forskningsprojekt för att undersöka om offentlig information om bolag, blankning, prisrörelser, rapporter och andra marknadsvariabler innehåller statistiskt och ekonomiskt användbara mönster.

Projektet kombinerar:

* datainsamling
* feature engineering
* datakvalitet
* ML
* walk-forward evaluation
* hypotesdriven research
* automatiserade analyser
* reproducerbara resultat

Målet är inte att bygga en samling fristående analyser, utan en återanvändbar forskningspipeline där nya hypoteser kan testas snabbt, systematiskt och utan onödig specialkod.

⸻

Översikt

                         Blankdiss
                            │
              ┌─────────────┴─────────────┐
              │                           │
        Data collection                ML / Research
              │                           │
       ┌──────┴──────┐             ┌──────┴──────┐
       │             │             │             │
      FI           Prices       Research     Diagnostics
       │             │             │             │
       └──────┬──────┘             │             │
              ▼                    │             │
       Feature generation           │             │
              │                    │             │
              ▼                    ▼             ▼
       Feature dataset        ml/research/   ml/diagnostics/
              │                    │             │
              ▼                    └──────┬──────┘
         Feature QC                      │
              │                          │
              └──────────────┬───────────┘
                             ▼
                        OOS results
                             │
                             ▼
                       AI / analysis

⸻

Forskningsflödet

Den normala vägen från idé till resultat är:

Ny hypotes
    ↓
Kan den uttryckas deklarativt?
    ↓
YAML research spec
    ↓
SCAN
    ↓
Intressant resultat?
    ↓
DEEP
    ↓
Robusthet / specialanalys
    ↓
OOS-resultat

Målet är att en ny vanlig hypotes ska kunna testas utan att en ny Python-fil, experimentklass eller workflow behöver byggas.

⸻

Data → Features → Research

FI data
   │
   ├──────────────┐
   │              │
   ▼              ▼
FI aggregation   Prices
   │              │
   └──────┬───────┘
          ▼
   Feature generation
          │
          ▼
   Feature dataset
          │
          ▼
      Feature QC
          │
          ▼
     ML / Research
          │
     ┌────┴────┐
     │         │
     ▼         ▼
 Research  Diagnostics
     │         │
     └────┬────┘
          ▼
      OOS results

⸻

ML-systemet

ML-systemet finns under ml/.

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
    └── experiments/

config.py innehåller gemensamma targets och walk-forward-konfiguration.

dataset.py ansvarar för att läsa och förbereda feature-datasetet.

walk_forward.py innehåller den gemensamma walk-forward-logiken.

research/ är den generiska forskningsmotorn.

diagnostics/ innehåller analyser som kräver mer specialiserad experimentlogik.

⸻

Research och Diagnostics

Research

Research används för generisk och bred hypotes-screening.

Exempel:

signal
  ×
tail
  ×
target
  ×
window

Research är deklarativ när det är möjligt.

Diagnostics

Diagnostics används när frågan kräver egen analyslogik.

Exempel:

* komplexa interaktioner
* mekanismanalyser
* specialiserade modeller
* event-sekvenser
* path dependence
* avancerad ekonomisk analys

Principen är:

Använd Research när frågan är generell. Använd Diagnostics när frågan kräver speciallogik.

⸻

Walk-forward och OOS

All modell- och hypotesutvärdering ska respektera tidsordningen:

TRAIN
   ↓
VALIDATION
   ↓
MODEL SELECTION
   ↓
REFIT
   ↓
OOS TEST

Testdata får inte användas för:

* modellval
* feature selection
* threshold selection
* optimering
* efterhandsjustering av hypotesen

Detta gäller även diagnostics.

⸻

Resultat

Research-resultat skrivs under:

data/processed/ml/research/

Nya deklarativa körningar använder:

data/processed/ml/research/spec_runs/

Resultaten ska vara maskinläsbara och lämpade för vidare analys.

Den avsedda kedjan är:

Research
   ↓
Machine-readable results
   ↓
AI / human analysis
   ↓
Next hypothesis

Terminaloutput är främst för körningsstatus och felsökning.

⸻

Designprinciper

1. Hypotes före implementation.
2. Generisk research före specialkod.
3. SCAN före dyra analyser.
4. OOS före slutsats.
5. Ingen test leakage.
6. Gemensam logik ska återanvändas.
7. Resultat ska vara maskinläsbara.
8. Specialanalys ska vara explicit.
9. Död och duplicerad kod ska inte ligga kvar.
10. Optimera för:

idé → information

inte för mängden kod.

⸻

Dokumentation

* ml/README.md – ML-arkitekturen
* ml/research/README.md – Research Engine
* ml/diagnostics/README.md – Diagnostics
* bolagsverket/README.md – Bolagsverket-ingestion
* README.md – övergripande projektarkitektur
