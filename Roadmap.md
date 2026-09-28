Absolut. Här är hela roadmap.md som vi håller oss efter:

# Blankdiss Roadmap
## Målbild
Blankdiss ska bli en daglig, reproducerbar forskningsmaskin som skiljer tydligt mellan discovery, frysta forskningskandidater och prospektiv evaluation.
Grundprincip:
> Discovery får hitta. Evaluation får inte ändra hypotesen. Walk-forward testar om kandidaten fortsätter att hålla.
Målflödet:
```text
NY DATA
  ↓
DATA QC
  ↓
FEATURE BUILD
  ↓
DISCOVERY
  ↓
CONTROLLED HYPOTHESIS TEST
  ↓
FROZEN RESEARCH CANDIDATE
  ↓
PROSPECTIVE EVALUATION
  ↓
REPEATED WALK-FORWARD
  ↓
VERIFICATION
  ↓
JSON / JSONL RESULTS

⸻

Fas 0 — Baslinje och skyddsräcken

Mål: förstå och bevara det som redan fungerar innan ny arkitektur byggs ovanpå.

* [ ]	Kartlägg nuvarande dataflöde.
* [ ]	Kartlägg befintliga research runners.
* [ ]	Kartlägg feature-build och QC.
* [ ]	Kartlägg befintliga diagnostics.
* [ ]	Kartlägg nuvarande resultatinfrastruktur.
* [ ]	Identifiera vilka tester och filer som är kritiska.
* [ ]	Behåll gamla workflows tills deras ansvar är ersatt och verifierat.

Klart när: vi kan beskriva data → features → research → results → verification och vet vad som ska återanvändas.

⸻

Fas 1 — Ny research-struktur

Ny struktur är en input/specifikationsnivå, inte en parallell resultatinfrastruktur.

research/
├── discovery/
│   └── specs/
├── candidates/
│   └── specs/
└── evaluation/
    └── specs/

* [ ]	Skapa katalogerna.
* [ ]	Definiera YAML-schema för discovery.
* [ ]	Definiera YAML-schema för candidates.
* [ ]	Definiera YAML-schema för evaluation.
* [ ]	Definiera spec-versionering.
* [ ]	Dokumentera semantiken.

Befintliga resultat fortsätter tills vidare under:

data/processed/ml/research/
├── latest/
└── <timestamp>/

YAML = input/specifikation.
JSON/JSONL = maskinläsbara resultat.

⸻

Fas 2 — Kandidatmodell och livscykel

Livscykeln ska vara explicit:

DISCOVERY
    ↓
CONTROLLED HYPOTHESIS TEST
    ↓
FROZEN RESEARCH CANDIDATE
    ↓
PROSPECTIVE EVALUATION
    ↓
REPEATED WALK-FORWARD
    ↓
MODEL CANDIDATE

* [ ]	Definiera candidate ID.
* [ ]	Definiera immutable kandidatmetadata.
* [ ]	Definiera discovery cutoff.
* [ ]	Definiera freeze timestamp.
* [ ]	Definiera candidate-specens version.
* [ ]	Definiera promotion-regler.

Regel: frysta kandidater är immutabla

Om exempelvis:

momentum_60d_quantile: 0.10

ändras till 0.15 är det en ny kandidat, inte en ändring av den gamla.

⸻

Fas 3 — Controlled Hypothesis Test

Discovery-resultat ska inte automatiskt betraktas som validerade signaler.

* [ ]	Gör discovery-resultat till explicita hypoteser.
* [ ]	Definiera förutbestämda testregler.
* [ ]	Dokumentera tränings-/testperioder.
* [ ]	Dokumentera feature- och target-definitioner.
* [ ]	Spara alla parametrar i metadata.
* [ ]	Förbjud resultatdriven parameterändring inom samma kandidat.

Klart när: en kandidat kan frysas utan att definitionen fortfarande beror på testresultatet.

⸻

Fas 4 — Prospective Evaluation

Evaluation-AI får:

* mäta,
* analysera,
* diagnostisera,
* rapportera.

Evaluation-AI får inte:

* ändra kandidatens parametrar,
* optimera kandidaten mot evaluation-perioden,
* välja en ny definition efter resultatet.

Arbete:

* [ ]	Skapa evaluation-runner.
* [ ]	Läs kandidatdefinition från YAML.
* [ ]	Läs evaluation-spec från YAML.
* [ ]	Kontrollera temporal separation.
* [ ]	Kör evaluation.
* [ ]	Spara JSON/JSONL.
* [ ]	Spara exakt data cutoff.
* [ ]	Verifiera att kandidaten inte muterats.

⸻

Fas 5 — Walk-forward

Samma frysta kandidat ska kunna testas över flera framtida perioder.

Frozen Candidate
      │
      ├── Evaluation window 1
      ├── Evaluation window 2
      ├── Evaluation window 3
      ├── Evaluation window 4
      └── ...

* [ ]	Definiera walk-forward-schema i YAML.
* [ ]	Definiera fönster och cutoffs.
* [ ]	Säkerställ att framtida data aldrig används bakåt.
* [ ]	Kör samma kandidat över flera fönster.
* [ ]	Spara varje fönster separat.
* [ ]	Skapa aggregerade JSON-resultat.
* [ ]	Mät stabilitet mellan perioder.
* [ ]	Dokumentera sample size och osäkerhet.

Viktig regel: framtida evaluation-resultat får aldrig användas för att ändra den frysta kandidaten.

⸻

Fas 6 — Blankdiss entrypoint

Hela processen ska kunna startas med ett gemensamt programgränssnitt, exempelvis:

python -m blankdiss

* [ ]	Definiera entrypoint.
* [ ]	Definiera körordning.
* [ ]	Definiera exit codes.
* [ ]	Definiera logging.
* [ ]	Definiera run ID.
* [ ]	Definiera input/output-kontrakt.
* [ ]	Gör samma kärnprocess körbar lokalt.
* [ ]	Lägg tester runt orkestreringen.

Princip: Python innehåller forskningslogiken. GitHub Actions orkestrerar miljö och körning.

⸻

Fas 7 — Nya .github/workflows/blankdiss.yml

blankdiss.yml ska bli Blankdiss dagliga huvudworkflow och får helt nytt innehåll.

Målflöde:

Checkout
   ↓
Setup Python
   ↓
Install dependencies
   ↓
Fetch/update data
   ↓
Build features
   ↓
Discovery
   ↓
Candidate handling
   ↓
Prospective evaluation
   ↓
Walk-forward
   ↓
Verification
   ↓
Persist results

* [ ]	Lägg daglig trigger.
* [ ]	Behåll manuell workflow_dispatch.
* [ ]	Kör samma Blankdiss-entrypoint som lokalt.
* [ ]	Lägg fail-fast på kritiska QC/verifieringar.
* [ ]	Spara artifacts.
* [ ]	Spara run metadata.
* [ ]	Säkerställ reproducerbarhet.

Workflowet ska inte innehålla forskningslogik i form av en lång lista specialkommandon.

Dess roll är:

starta Blankdiss → kör Blankdiss → verifiera Blankdiss → spara resultatet.

⸻

Fas 8 — Resultat och provenance

Varje research-resultat ska kunna spåras tillbaka.

Minsta metadata bör omfatta:

{
  "run_id": "...",
  "candidate_id": "...",
  "spec_version": "...",
  "data_cutoff": "...",
  "created_at": "...",
  "features": [],
  "parameters": {},
  "evaluation_period": {},
  "metrics": {}
}

* [ ]	Definiera gemensamt metadataformat.
* [ ]	Definiera run ID.
* [ ]	Definiera candidate ID.
* [ ]	Definiera spec-version.
* [ ]	Definiera data cutoff.
* [ ]	Definiera source-data-version där möjligt.
* [ ]	Spara JSON/JSONL.
* [ ]	Kontrollera att runs kan identifieras entydigt.

⸻

Fas 9 — Verification och leakage-skydd

Blankdiss ska aktivt kunna upptäcka forskningsfel.

* [ ]	Temporal leakage.
* [ ]	Feature leakage.
* [ ]	Target leakage.
* [ ]	Future-data access.
* [ ]	Candidate mutation.
* [ ]	Evaluation contamination.
* [ ]	Duplicate runs.
* [ ]	Saknade data.
* [ ]	Oväntade distributionsförändringar.
* [ ]	Sample-size-varningar.

Klart när: en run kan stoppas med ett tydligt fel när en central forskningsregel bryts.

⸻

Fas 10 — Integrera befintlig pipeline

Återanvänd fungerande kod där den passar.

Delar att mappa in:

* fi
* prices
* analysis.build_features
* befintlig feature-QC
* befintliga CI-verifieringar
* ml.research.runner
* ml.experiment_registry_runner
* befintliga diagnostics
* befintlig resultatinfrastruktur
* [ ]	Mappa varje runner mot nya livscykeln.
* [ ]	Återanvänd direkt där möjligt.
* [ ]	Bygg adapters där det behövs.
* [ ]	Identifiera verklig legacy.
* [ ]	Ta bort legacy först efter ersättning + verifiering.

⸻

Fas 11 — Discovery-AI

Discovery ska kunna söka efter nya strukturer utan att blandas ihop med valideringen.

Discovery får:

* testa många hypoteser,
* söka feature-interaktioner,
* hitta oväntade samband,
* föreslå kandidater.

Discovery får inte automatiskt:

* kalla något validerat,
* använda framtida evaluation-resultat,
* ändra en redan fryst kandidat.

Output från discovery ska vara strukturerade hypoteser/kandidater som kan gå vidare till controlled hypothesis test.

⸻

Fas 12 — Stabil daglig loop

När ovanstående delar fungerar ska Blankdiss dagliga process vara:

NY DATA
  ↓
DATA QC
  ↓
FEATURE BUILD
  ↓
DISCOVERY
  ↓
HYPOTHESIS TEST
  ↓
FREEZE
  ↓
PROSPECTIVE EVALUATION
  ↓
WALK-FORWARD
  ↓
VERIFICATION
  ↓
JSON / JSONL
  ↓
DAILY RUN

⸻

Regler vi inte bryter mot

1. Ingen kandidatmutering

En fryst kandidat ändras aldrig.

Ändring = ny kandidat.

2. Ingen evaluation-driven optimization

Vi ändrar inte hypotesen för att förbättra evaluation-resultatet.

3. Ingen framtidsinformation

Framtida information får inte påverka beslut som ska representera vad som var känt vid den tidigare tidpunkten.

4. Ingen blind filborttagning

Först:

identifiera
→ förstå
→ ersätta
→ verifiera
→ ta bort

5. YAML in, JSON/JSONL out

YAML används för specifikationer. JSON/JSONL används för maskinläsbara resultat.

6. Workflowet är inte forskningsmotorn

.github/workflows/blankdiss.yml orkestrerar.

Python-koden forskar.

7. Samma process lokalt och i CI

Den centrala Blankdiss-processen ska kunna köras både lokalt och i GitHub Actions.

⸻

Definition of Done

Roadmapens slutmål är uppnått när en daglig körning kan:

1. hämta/uppdatera aktuell data,
2. verifiera datan,
3. bygga aktuella features,
4. köra discovery,
5. skapa strukturerade hypoteser,
6. frysa explicita kandidater,
7. utvärdera tidigare frysta kandidater prospektivt,
8. köra walk-forward,
9. upptäcka och stoppa leakage,
10. spara full provenance,
11. producera JSON/JSONL-resultat,
12. köras både lokalt och i GitHub Actions,
13. reproducera tidigare körningar från sparade specs och metadata.

Slutmålet är inte att Blankdiss varje dag ska hitta en bra signal.

Slutmålet är att Blankdiss ska kunna skilja mellan:

"Det här ser intressant ut"
        ↓
"Det här är en explicit hypotes"
        ↓
"Det här är en fryst kandidat"
        ↓
"Det här fungerade på ny data"
        ↓
"Det här fortsätter att fungera över tid"

på ett reproducerbart sätt.
