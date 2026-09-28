# Blankdiss

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

## Översikt

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

## Forskningsflödet

Den normala vägen från idé till resultat är:

Ny hypotes
    ↓
Kan befintlig Research Engine uttrycka den?
    │
    ├── JA
    │    ↓
    │  YAML research spec
    │    ↓
    │  SCAN
    │
    └── NEJ
         ↓
    Behövs en ny generell analysform?
         │
         ├── JA
         │    ↓
         │  Utöka Research Engine
         │    ↓
         │  YAML research spec
         │    ↓
         │  SCAN
         │
         └── NEJ
              ↓
         custom / Diagnostics

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

## Viktig arkitekturregel

En ny hypotes och en ny analysform är två olika saker.

Ny hypotes:
    YAML

Ny generell analysform:
    Engine + YAML

Unik specialanalys:
    custom/ eller Diagnostics

Man ska inte skapa standalone-analyser bara för att den första hypotesen av en viss typ kräver ny Engine-funktionalitet.

Om Research Engine exempelvis saknar stöd för en generell modelljämförelse ska modelljämförelsen implementeras som en generell Engine-funktion.

Därefter uttrycks den konkreta hypotesen som YAML.

Det ska alltså inte bli:

Ny hypotes
    ↓
ny Python-fil
    ↓
ny experimentklass
    ↓
ny registry
    ↓
ny workflow

⸻

## Data → Features → Research

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

## Operational pipeline

Den aktiva datapipelinen körs i:

.github/workflows/blankdiss.yml

Den normala upstream-kedjan är:

FI aggregate
    ↓
Prices
    ↓
OMXSPI
    ↓
analysis.build_features
    ↓
Feature dataset
    ↓
Feature QC
    ↓
ML / Research

Feature generation är alltså en aktiv del av pipeline-kedjan och ska
inte betraktas som legacy.

Feature-datasetet skrivs under:

data/processed/analysis/

med:

* features_*.jsonl – kanoniska feature-chunks
* features_metadata.json – metadata och source fingerprint
* features_qc.json – resultat från Feature QC

`ml/dataset.py` ansvarar för att läsa det kanoniska feature-datasetet.

Research Engine använder samma feature-dataset som grund för
forskningskörningarna.

### Workflowansvar

`.github/workflows/blankdiss.yml` ansvarar för den gemensamma
datapipelinen:

    FI
     ↓
    Prices
     ↓
    OMXSPI
     ↓
    Feature generation
     ↓
    Feature QC

`.github/workflows/ml-research-new.yml` startar AI Lab och Research Engine
men bygger inte feature-datasetet själv.

Den normala kedjan är därför:

Data workflow
    ↓
Feature dataset
    ↓
Research Engine / AI Lab

Research Engine ska inte själv börja bygga om feature-datasetet som en
del av varje research-spec.

⸻

## Feature generation

Den kanoniska feature-byggaren är:

analysis/build_features.py

Den använder de separata featuremodulerna för:

* FI-features
* prisfeatures
* forward returns
* source fingerprint
* feature-dataset
* metadata

Feature-datasetet byggs som chunks:

data/processed/analysis/features_*.jsonl

Metadata skrivs till:

data/processed/analysis/features_metadata.json

och Feature QC skrivs till:

data/processed/analysis/features_qc.json

Feature generation ska köras efter att FI-data, prisdata och OMXSPI har
uppdaterats.

Feature-datasetet är upstream för ML/Research. Nya forskningshypoteser ska
inte bygga egna parallella feature-dataset.

⸻

## Source fingerprint

Feature metadata innehåller ett source fingerprint för det underlag som
användes vid feature generation.

Fingerprintet baseras på:

* FI aggregate
* prisfiler

Det gör det möjligt att identifiera vilket upstream-underlag som användes
för ett visst feature-dataset.

Om upstream-data ändras ska feature-datasetet byggas om.

⸻

## Legacy feature-data

Det finns fortfarande äldre featureartefakter i repot.

Framför allt:

data/processed/analysis/fi_price_features.jsonl
data/processed/analysis/fi_price_features_metadata.json

Dessa finns kvar av kompatibilitetsskäl tills den separata
legacy-städningen genomförs.

De ska inte användas som en alternativ väg för ny utveckling.

Den kanoniska feature-vägen är:

data/processed/analysis/features_*.jsonl
    ↓
ml/dataset.py
    ↓
ML / Research

Legacy-städningen är ett separat arbete och är inte en del av
Research Engine-migreringen.

⸻

## Research Engine-migreringen

Research Engine-migreringen är klar.

Den nya arkitekturen är nu den aktiva vägen för ny forskning.

Det innebär:

* YAML-baserad research är aktiv.
* Research Engine är den generiska forskningsmotorn.
* AI Lab kan skapa och köra research specs.
* Feature generation är aktiv igen.
* Feature QC är aktiv igen.
* Äldre legacy-filer kan fortfarande finnas kvar.
* Legacy-filerna ska tas bort i ett separat städarbete.

Det är viktigt att skilja på:

1. migrering av den aktiva arkitekturen
2. borttagning av gammal kod

Migreringen är klar.

Borttagningen av legacy-filer är däremot ännu inte utförd.

En legacy-fil som fortfarande finns i repot ska därför inte betraktas som
en del av den nya arkitekturen bara för att den finns kvar.

⸻

## ML-systemet

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

## Research och Diagnostics

### Research

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

### Diagnostics

Diagnostics används när frågan kräver egen analyslogik.

Exempel:

* komplexa interaktioner
* mekanismanalyser
* specialiserade modeller
* event-sekvenser
* path dependence
* avancerad ekonomisk analys

Principen är:

Använd Research när frågan är generell.

Om Research Engine saknar den generella analysförmåga som behövs ska
Engine utökas innan hypotesen flyttas till Diagnostics.

Använd Diagnostics när frågan kräver verkligt specialiserad logik.

⸻

## Införande av nya forskningshypoteser

När en ny forskningsidé uppstår ska följande ordning användas:

1. Formulera forskningsfrågan.
2. Läs relevant README-dokumentation.
3. Kontrollera befintliga Research Engine-analysis types.
4. Kontrollera befintliga signaler, targets och cache-funktionalitet.
5. Avgör om hypotesen redan kan beskrivas med YAML.
6. Om JA: skapa en YAML research spec.
7. Om NEJ: identifiera vilken generell funktionalitet som saknas.
8. Om den saknade funktionaliteten är generell: implementera den i Research Engine.
9. Uttryck därefter den konkreta hypotesen i YAML.
10. Om analysen inte är generell och kräver verkligt specialiserad logik: använd custom/ eller Diagnostics.
11. Kör SCAN/DEEP.
12. Verifiera OOS-resultat.
13. Ta bort eventuell äldre standalone-/legacy-implementation när den nya vägen är verifierad.

Exempel:

Hypotes:
    momentum
    +
    short-interest change
    +
    interaction

Kontroll:
    Finns analysis type?
        │
        ├── JA
        │    ↓
        │  YAML
        │
        └── NEJ
             ↓
        Är analysformen generell?
             │
             ├── JA
             │    ↓
             │  Engine
             │    ↓
             │  YAML
             │
             └── NEJ
                  ↓
             custom / Diagnostics

⸻

## Walk-forward och OOS

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

## Resultat

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

## Designprinciper

1. Hypotes före implementation.
2. Generisk research före specialkod.
3. Kontrollera befintlig Engine innan ny Python skrivs.
4. Ny generell analysförmåga ska implementeras i Engine.
5. Konkreta hypoteser ska normalt vara YAML.
6. SCAN före dyra analyser.
7. OOS före slutsats.
8. Ingen test leakage.
9. Gemensam logik ska återanvändas.
10. Resultat ska vara maskinläsbara.
11. Specialanalys ska vara explicit.
12. Död och duplicerad kod ska inte ligga kvar.
13. Legacy-implementationer ska tas bort efter verifierad migrering.
14. Migreringen av Research Engine är klar; kvarvarande legacy-filer är
    ett separat städarbete.
15. Feature generation är en aktiv del av datapipelinen och ska inte
    betraktas som legacy.
16. Optimera för:

idé → information

inte för mängden kod.

⸻

## Dokumentation

* ml/README.md – ML-arkitekturen
* ml/research/README.md – Research Engine
* ml/diagnostics/README.md – Diagnostics
* bolagsverket/README.md – Bolagsverket-ingestion
* README.md – övergripande projektarkitektur
