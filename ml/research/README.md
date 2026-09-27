Absolut. Jag har utgått från den faktiska README:n på main och uppdaterat bara det som behöver ändras för den nya Research Engine-strukturen. Jag skulle samtidigt lägga in ett tydligare avsnitt om ansvarsfördelningen mellan spec.py, engine.py och analysmodulerna.

Här är hela README:n:

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
### Ny hypotes
YAML
### Ny generell analysform
Research Engine + YAML
### Unik specialanalys
custom/ eller Diagnostics
Man ska inte skapa standalone-analyser bara för att den första hypotesen av en viss typ kräver ny Engine-funktionalitet.
Om Research Engine exempelvis saknar stöd för en generell analysform ska analysformen implementeras som generell Engine-funktionalitet.
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
## Research Engine
Research Engine är den generiska körmotorn för deklarativ hypotesdriven research.
Ansvarsfördelningen är:
```text
spec.py
    │
    │ deklarativa regler och kontrakt
    ▼
engine.py
    │
    │ dispatch + orchestration
    ▼
analysis modules
    │
    │ konkret analyslogik
    ▼
analysis_utils.py
    │
    │ gemensamma implementation-hjälpare
    ▼
results

spec.py

spec.py definierar Research Specs och deras deklarativa kontrakt.

Det omfattar bland annat:

* analysis types
* signaler
* targets
* windows
* splits
* bootstrap-konfiguration
* krav på antal signaler för olika analysis types

Exempelvis är antalet signaler en egenskap hos analysformen och inte något som ska avgöras av engine.py.

Det betyder att:

spec.py
    ↓
"conditional_regime_comparison kräver minst 2 signaler"

medan den konkreta analysimplementationen fortfarande får kontrollera sina egna runtime-invarianter.

engine.py

engine.py ska främst vara orchestration och dispatch.

Den ansvarar för att:

* läsa en ResearchSpec
* iterera över targets, windows och splits
* kombinera YAML-specens parametrar
* välja rätt analysis module
* samla resultaten

engine.py ska inte innehålla den fullständiga implementationen av varje analysform.

Exempel:

analysis.type
    │
    ├── tail
    ├── interaction
    ├── regime_comparison
    ├── multi_regime_comparison
    ├── nested_regime_comparison
    └── conditional_regime_comparison

Engine väljer rätt analysmodul.

Analysis modules

Varje generell analysform ska ha sin egen implementation när logiken är tillräckligt omfattande.

ml/research/
├── interaction.py
├── conditional.py
├── regime.py
├── multi_regime.py
└── nested_regime.py

Ansvar:

* interaction.py → interaction-analyser
* conditional.py → conditional regime-analyser
* regime.py → tvåsignal regime_comparison
* multi_regime.py → tre eller fler signaler i kombinerad regim
* nested_regime.py → inkrementell signal efter etablerad baseline-regim

Analysmodulerna äger den konkreta runtime-logiken och sina implementation-invarianter.

analysis_utils.py

analysis_utils.py innehåller gemensamma implementation-hjälpare som används av flera analysmoduler.

Exempel:

* stabil seed-generering
* gemensam beräkning av event rate

Detta är inte en egen analysis type.

Syftet är att undvika att samma implementation ligger duplicerad i flera analysis modules eller som privata helpers i engine.py.

Runtime-invarianter

Deklarativa krav och runtime-skydd har olika ansvar.

Exempel:

spec.py
    ↓
conditional_regime_comparison
måste ha minst 2 signaler

och:

conditional.py
    ↓
kontrollerar fortfarande
att runtime-anropet faktiskt
har tillräckligt många signaler

Detta är avsiktligt.

spec.py definierar kontraktet.

Analysis module skyddar sin implementation.

⸻

Research Engine-struktur

Den generiska researchdelen ligger under:

ml/research/
├── spec.py
├── session.py
├── cache.py
├── signals.py
├── engine.py
├── runner.py
├── reporting.py
├── bootstrap.py
├── analysis_utils.py
├── interaction.py
├── conditional.py
├── regime.py
├── multi_regime.py
├── nested_regime.py
├── specs/
└── custom/

Övergripande ansvar:

spec.py
    ↓
deklarativa Research Specs
session.py
    ↓
research-session / körningskontext
cache.py
    ↓
förberäknade masks och data
signals.py
    ↓
signalrelaterad logik
engine.py
    ↓
dispatch och orchestration
analysis modules
    ↓
konkret generell analyslogik
analysis_utils.py
    ↓
gemensamma implementation-hjälpare
bootstrap.py
    ↓
statistisk bootstrap-logik
runner.py
    ↓
körning av research specs
reporting.py
    ↓
resultat och rapportering
specs/
    ↓
konkreta YAML-hypoteser
custom/
    ↓
specialiserade analyser

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
│   ├── analysis_utils.py
│   ├── interaction.py
│   ├── conditional.py
│   ├── regime.py
│   ├── multi_regime.py
│   ├── nested_regime.py
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

Använd Research när frågan är generell.

Om Research Engine saknar den generella analysförmåga som behövs ska Engine utökas innan hypotesen flyttas till Diagnostics.

Använd Diagnostics när frågan kräver verkligt specialiserad logik.

⸻

Införande av nya forskningshypoteser

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
14. Deklarativa kontrakt ska ligga i spec.py.
15. engine.py ska orkestrera, inte äga analysimplementationerna.
16. Gemensamma implementation-hjälpare ska återanvändas via analysis_utils.py.
17. Runtime-invarianter ska skyddas i respektive analysis module.
18. Optimera för:

idé → information

inte för mängden kod.

⸻

Dokumentation

* ml/README.md – ML-arkitekturen
* ml/research/README.md – Research Engine
* ml/diagnostics/README.md – Diagnostics
* bolagsverket/README.md – Bolagsverket-ingestion
* README.md – övergripande projektarkitektur

Det här är den version jag skulle använda. **Ingen forskningslogik ändras** av README-uppdateringen; den dokumenterar den struktur som redan körs framgångsrikt på `main`.
