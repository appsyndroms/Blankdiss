Scripts

scripts/ innehåller tunna shell-wrapperar för återkommande kommandon i Blankdiss.

Syftet är främst att göra vanliga körningar enklare och mer konsekventa. Den egentliga data-, analys- och forskningslogiken ska ligga i projektets Python-kod.

Innehåll

Katalogen innehåller bland annat:

scripts/
├── build_features.sh
├── fetch_fi.sh
├── fetch_prices.sh
├── git_push_with_retry.sh
└── update_market.sh

build_features.sh

Wrapper för feature-byggandet.

Den används för att starta den centrala feature-pipelinen i:

analysis/build_features.py

Feature-resultatet skrivs till:

data/processed/analysis/

Själva feature-logiken ska inte ligga i shell-scriptet.

fetch_fi.sh

Wrapper för hämtning eller uppdatering av Finansinspektionens blankningsdata.

Scriptet kapslar in det praktiska kommandot för datainsamlingen så att det kan köras på ett konsekvent sätt.

Den centrala datalogiken ska ligga i projektets Python-/datainsamlingskod.

fetch_prices.sh

Wrapper för hämtning eller uppdatering av prisdata.

På samma sätt som övriga scripts ska detta främst vara ett enkelt startlager ovanpå den egentliga implementationen.

update_market.sh

Wrapper för uppdatering av marknadsdata.

Marknadsdata används senare av feature-pipelinen som en del av det gemensamma feature-datasetet.

git_push_with_retry.sh

Hjälpscript för Git push med retry-hantering.

Det är ett operativt verktyg och innehåller inte någon del av Blankdiss forsknings- eller analyslogik.

Princip

Scripts-katalogen ska hållas tunn.

Exempel:

scripts/build_features.sh
          │
          ▼
analysis/build_features.py
          │
          ▼
feature dataset

Shell-scriptet ska alltså inte utvecklas till ytterligare ett applikationslager.

Varför wrappers?

Wrappers kan vara användbara för:

* kortare kommandon
* konsekventa argument
* enklare lokal körning
* automatisering
* CI/CD
* återkommande datauppdateringar

De gör också att vanliga arbetsflöden kan dokumenteras med ett enkelt kommando utan att den underliggande implementationen behöver dupliceras.

Förhållande till övriga projektet

scripts/
   │
   ├── hämtning ──────────► data
   │
   ├── feature build ─────► analysis/
   │
   └── git-hjälp ─────────► repository
analysis/
   │
   ▼
feature dataset
   │
   ▼
ml/

Scripts är därmed ett hjälp- och integrationslager, inte en del av den centrala forskningsarkitekturen.
