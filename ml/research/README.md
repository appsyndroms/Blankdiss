Blankdiss Research System

Detta dokument beskriver hur Blankdiss forskningssystem är uppbyggt, hur registrerade experiment körs och hur nya forskningsfrågor ska läggas till.

Målet är att forskningsarbetet ska vara så automatiserat som möjligt:

1. En forskningsfråga definieras.
2. En diagnostic implementerar själva analysen.
3. Experimentet registreras i ml/experiment_registry.json.
4. Den befintliga GitHub Actions-workflowen .github/workflows/ml-research.yml kör forskningen.
5. ml/experiment_registry_runner.py hittar aktiva experiment och kör respektive diagnostic.
6. Resultaten sparas som forskningsartefakter.

Det ska alltså normalt inte krävas manuella ändringar i workflow-filen för varje nytt experiment.

⸻

Översikt

Forskningssystemet består av fyra huvuddelar:

.github/workflows/ml-research.yml
              │
              ▼
ml/experiment_registry_runner.py
              │
              ▼
ml/experiment_registry.json
              │
              ├── active experiment
              │       │
              │       ▼
              │   ml/diagnostics/<diagnostic>.py
              │
              └── planned experiments

Det finns dessutom en separat generell research matrix:

ml/research/runner.py
        │
        ├── ml/research/experiments.py
        ├── ml/research/evaluator.py
        └── ml/research/cache.py

Den generella research matrixen och de registrerade diagnostics är två kompletterande delar av forskningssystemet.

⸻

1. GitHub Actions

Den centrala workflow-filen är:

.github/workflows/ml-research.yml

Den kör hela forskningskedjan från datainsamling till ML-resultat.

Workflowen gör bland annat:

Fetch FI aggregate
        ↓
Fetch prices
        ↓
Build features
        ↓
Feature inspection / verification
        ↓
Feature QC
        ↓
Generic Research Matrix
        ↓
Registered Experiments
        ↓
Verify results
        ↓
Upload research artifacts

Workflowen ska använda:

- name: Run Registered Experiments
  run: |
    python -u -m ml.experiment_registry_runner

Detta steg startar registry-runnern.

Viktigt

Nya experiment ska normalt inte läggas direkt i .github/workflows/ml-research.yml.

I stället ska experimentet registreras i:

ml/experiment_registry.json

Workflowen behöver då inte ändras när ytterligare experiment aktiveras.

⸻

2. Experiment Registry

Registry-filen är:

ml/experiment_registry.json

Den fungerar som forskningssystemets lista över definierade experiment.

Aktuell struktur:

{
  "version": 1,
  "experiments": [
    {
      "id": "volatility_si_level_interaction",
      "status": "active",
      "question": "Är risken för en extrem nedgång särskilt hög när både volatiliteten och blankningsnivån är hög?",
      "module": "ml.diagnostics.volatility_si_level_interaction_diagnostic",
      "priority": 1
    },
    {
      "id": "si_event_risk_interaction",
      "status": "active",
      "question": "Är effekten av förändrad blankning starkare vid extrem event-risk?",
      "module": "ml.diagnostics.fi_short_interest_event_risk_interaction_diagnostic",
      "priority": 2
    },
    {
      "id": "report_proximity",
      "status": "planned",
      "question": "Förstärks sambandet nära rapportdatum?",
      "module": "ml.diagnostics.report_proximity_diagnostic",
      "priority": 3
    },
    {
      "id": "sector_relative_return",
      "status": "planned",
      "question": "Är effekten specifik för aktien relativt sektor och marknad?",
      "module": "ml.diagnostics.sector_relative_diagnostic",
      "priority": 4
    }
  ]
}

Fälten

id

Unikt maskinläsbart namn på experimentet.

Exempel:

"id": "volatility_si_level_interaction"

ID:t ska vara stabilt och ska inte ändras utan anledning.

⸻

status

Styr om experimentet körs.

Tillåtna praktiska värden:

active
planned

active betyder att experimentet körs.

planned betyder att forskningsfrågan är dokumenterad men ännu inte ska köras.

Exempel:

"status": "planned"

När implementationen är klar ändras den till:

"status": "active"

⸻

question

Den mänskligt läsbara forskningsfrågan.

Exempel:

"question": "Är risken för en extrem nedgång särskilt hög när både volatiliteten och blankningsnivån är hög?"

Frågan ska beskriva hypotesen som experimentet försöker undersöka, inte implementationen.

⸻

module

Kopplingen mellan registryt och själva analyskoden.

Exempel:

"module": "ml.diagnostics.volatility_si_level_interaction_diagnostic"

Det betyder att Python importerar:

ml/diagnostics/volatility_si_level_interaction_diagnostic.py

och kör dess:

main()

Detta är den centrala kopplingen mellan research registry och diagnostic implementation.

⸻

priority

Bestämmer körordningen bland aktiva experiment.

Lägre nummer körs först.

Exempel:

priority 1
priority 2
priority 3

Det är främst en ordningsmekanism. Priority är inte ett statistiskt eller vetenskapligt betyg på experimentet.

⸻

3. Registry Runner

Filen:

ml/experiment_registry_runner.py

läser registryt och hittar alla experiment med:

"status": "active"

De sorteras efter priority.

För varje aktivt experiment:

1. Python-modulen importeras.
2. main() hämtas.
3. main() körs.
4. Fel isoleras till experimentet.
5. Alla aktiva experiment får möjlighet att köras.
6. Processen avslutas med exit code 1 om något experiment misslyckades.

Förenklat:

for experiment in active_experiments:
    module = importlib.import_module(experiment["module"])
    module.main()

Det betyder att registry-runnern inte innehåller själva forskningslogiken.

Den fungerar som orkestrator.

⸻

4. Diagnostics

Själva forskningsanalysen ligger under:

ml/diagnostics/

Exempel:

ml/diagnostics/
├── fi_short_interest_event_risk_interaction_diagnostic.py
├── volatility_si_level_interaction_diagnostic.py
└── ...

En diagnostic ska innehålla den specifika analyslogiken för en forskningsfråga.

Den ska normalt exponera:

def main():
    ...

Registryt pekar sedan på modulen:

"module": "ml.diagnostics.volatility_si_level_interaction_diagnostic"

Runnern kör:

module.main()

⸻

5. Exempel: Volatility × SI Level

Det första nya experimentet är:

volatility_si_level_interaction

Forskningsfrågan är:

Är risken för en extrem nedgång särskilt hög när både volatiliteten och blankningsnivån är hög?

Diagnostic:

ml/diagnostics/volatility_si_level_interaction_diagnostic.py

Experimentet använder i huvudsak:

* extrem nedgång som target
* volatilitet som en dimension
* blankningsnivå som en dimension
* OOS event-risk som ytterligare filtrering
* walk-forward evaluation

Den centrala 2×2-strukturen är:

                    LOW SI       HIGH SI
LOW VOL                A             B
HIGH VOL               C             D

Interaktionen beräknas som:

(D - C) - (B - A)

Det är viktigt eftersom frågan inte bara är om cell D har hög risk.

Frågan är om effekten av hög blankning förändras när volatiliteten är hög.

⸻

6. Event-riskmodellen

Det nya volatility × SI-testet ska bygga vidare på samma event-risklogik som det befintliga:

ml/diagnostics/fi_short_interest_event_risk_interaction_diagnostic.py

Den befintliga modellen använder bland annat event-risk baserad på:

10 % absolut förändring på 5 dagar

och event-risk feature sets som bland annat omfattar:

volatility_20d
volatility_60d
volatility_20d_plus_60d
volatility_20d_plus_60d_plus_term_structure

Event-riskmodellen ska väljas med train/validation och därefter refittas på train + validation innan test.

Testperioden ska förbli OOS.

Viktigt om leakage

Trösklar som används för att definiera OOS-grupper ska bestämmas från information som är tillgänglig före testperioden.

Exempel:

Train
   ↓
Validation
   ↓
Bestäm modell
   ↓
Refit train + validation
   ↓
Bestäm pre-test thresholds
   ↓
Test / OOS

Testutfall får inte användas för att definiera grupperna.

⸻

7. Walk-forward

Forskningssystemet använder walk-forward-perioder från:

ml/config.py

Nuvarande struktur är:

Window 1
Train       → 2023-12-31
Validation  → 2024-12-31
Test        → 2025-12-31
Window 2
Train       → 2024-12-31
Validation  → 2025-12-31
Test        → 2026-12-31

Nya diagnostics ska följa samma princip:

TRAIN
  ↓
VALIDATION
  ↓
MODEL SELECTION
  ↓
REFIT
  ↓
OOS TEST

Resultat från flera walk-forward-perioder kan därefter poolas.

⸻

8. Generic Research Matrix

Den generella research-motorn finns i:

ml/research/runner.py

och använder bland annat:

ml/research/experiments.py
ml/research/evaluator.py
ml/research/cache.py

Den genererar en större experimentmatris över:

* signaler
* targets
* tail fractions
* tail direction
* walk-forward windows
* train / validation / test

Den är alltså bredare och mer generell än en specifik diagnostic.

Den generella matrixen ska användas för bred screening.

Diagnostics används när en forskningsfråga kräver mer specifik metodik, till exempel:

* interaktioner
* specialiserade modeller
* event-risk
* rapportdatum
* sektorrelativa jämförelser
* specifika bootstrap-analyser
* andra strukturer som inte passar den generella signal/target-matrisen.

⸻

9. Hur man skapar ett nytt experiment

När en ny forskningsidé uppstår ska följande process användas.

Steg 1 — Definiera forskningsfrågan

Exempel:

Förstärks effekten av förändrad blankning när volatiliteten ökar?

Frågan ska vara tydlig innan implementationen skrivs.

⸻

Steg 2 — Bestäm om det passar generic research eller diagnostic

Använd den generella research matrixen om frågan kan uttryckas som en vanlig:

signal → target

analys.

Skapa en diagnostic om frågan kräver egen analyslogik.

Exempel:

volatility × SI interaction

är en diagnostic eftersom vi vill analysera en explicit interaktion och 2×2-struktur.

⸻

Steg 3 — Skapa diagnostic

Skapa:

ml/diagnostics/<experiment_name>_diagnostic.py

Exempel:

ml/diagnostics/volatility_si_change_interaction_diagnostic.py

Diagnosticen ska ha:

def main():
    ...

så att registry-runnern kan köra den.

⸻

Steg 4 — Implementera OOS korrekt

Diagnosticen ska följa Blankdiss walk-forward-princip:

train
validation
test

Modellval ska inte använda testdata.

Thresholds som definierar testgrupper ska inte optimeras på testutfall.

⸻

Steg 5 — Lägg experimentet i registryt

Lägg till ett objekt i:

ml/experiment_registry.json

Exempel:

{
  "id": "volatility_si_change_interaction",
  "status": "active",
  "question": "Förstärks effekten av förändrad blankning när volatiliteten ökar?",
  "module": "ml.diagnostics.volatility_si_change_interaction_diagnostic",
  "priority": 3
}

⸻

Steg 6 — Kör registry-runnern

Lokalt:

python -u -m ml.experiment_registry_runner

Detta kör alla experiment med:

"status": "active"

i priority-ordning.

⸻

Steg 7 — GitHub Actions

Den normala automatiserade körningen sker via:

.github/workflows/ml-research.yml

Workflowen bygger först data/features och kör sedan:

python -u -m ml.research.runner

och:

python -u -m ml.experiment_registry_runner

Därmed behöver man normalt bara:

1. skapa diagnostic
2. registrera experimentet
3. köra workflowen

⸻

10. Planned vs Active

Det är helt okej att registrera framtida experiment innan implementationen finns.

Exempel:

{
  "id": "report_proximity",
  "status": "planned",
  "question": "Förstärks sambandet nära rapportdatum?",
  "module": "ml.diagnostics.report_proximity_diagnostic",
  "priority": 3
}

När diagnosticen är färdig:

"status": "active"

Runnern börjar då köra den.

Detta gör registryt till en kombination av:

* forskningskatalog
* hypoteslista
* execution configuration

⸻

11. Resultat

Den generella research-runnern skriver resultat under:

data/processed/ml/research/

Senaste körningen finns under:

data/processed/ml/research/latest/

Bland annat:

results.jsonl
pooled.json
metadata.json
report.md

GitHub Actions verifierar att dessa artefakter finns och laddar upp dem som workflow artifacts.

Specifika diagnostics kan dessutom ha egna resultatformatskrav beroende på experimentets karaktär.

När en diagnostic byggs ut bör den helst följa samma princip:

machine-readable result
        +
human-readable summary

så att resultaten senare kan konsumeras automatiskt.

⸻

12. Viktig metodprincip

Blankdiss ska skilja mellan:

Screening

Bred sökning efter möjliga samband.

Exempel:

signal × target × tail

Detta passar:

ml/research/

Hypotesdriven diagnostic

En specifik forskningsfråga där vi vill testa en mekanism eller interaktion.

Exempel:

volatility × SI
SI change × volatility
event risk × SI
report proximity
sector relative return

Detta passar:

ml/diagnostics/

När en screening ger en intressant signal ska nästa steg därför ofta vara en mer explicit diagnostic, inte bara ännu en bred parameterkombination.

⸻

13. Planerade experiment

Registryt innehåller för närvarande bland annat följande forskningsspår:

1. Volatility × SI level

volatility_si_level_interaction

Fråga:

Är risken för en extrem nedgång särskilt hög när både
volatiliteten och blankningsnivån är hög?

2. SI change × event risk

si_event_risk_interaction

Fråga:

Är effekten av förändrad blankning starkare vid extrem event-risk?

3. Report proximity

report_proximity

Fråga:

Förstärks sambandet nära rapportdatum?

4. Sector relative return

sector_relative_return

Fråga:

Är effekten specifik för aktien relativt sektor och marknad?

Dessa ska ses som forskningsfrågor, inte som förutbestämda resultat.

⸻

14. När ett nytt experiment ska läggas till

Använd följande checklista:

[ ] Forskningsfrågan är tydligt definierad
[ ] Det är avgjort om generic research eller diagnostic passar
[ ] Diagnostic ligger under ml/diagnostics/
[ ] Diagnostic har main()
[ ] Train/validation/test är korrekt separerade
[ ] Ingen test leakage
[ ] Thresholds definieras före test
[ ] OOS-resultat rapporteras
[ ] Experimentet finns i experiment_registry.json
[ ] id är unikt
[ ] module pekar på rätt Python-modul
[ ] status är planned eller active
[ ] priority är satt
[ ] ml-research.yml behöver normalt inte ändras

⸻

15. Grundprincip

Det viktigaste arkitekturbeslutet är:

Registry = VAD ska köras?
Runner = HUR hittar och startar vi det?
Diagnostic = HUR genomförs forskningen?
Workflow = NÄR/var körs hela pipelinen?

Det gör att forskningssystemet kan växa utan att själva GitHub Actions-workflowen behöver byggas om för varje ny hypotes.

När en ny forskningsfråga är färdigimplementerad ska den i normalfallet kunna aktiveras genom:

"status": "active"

och sedan köras av samma automatiserade pipeline.
