Ja. Då börjar vi med första faktiska implementationssteget, och jag skriver ut hela filer — inte fragment.

Jag vill inte aktivera den nya dagliga kedjan ännu. Först bygger vi spec-lagret och kontrakten. Den befintliga ml/research/ får fortsätta vara motorn tills vi har kopplat den mot detta.

1. research/README.md

# Blankdiss Research Specifications
Detta katalogträd innehåller Blankdiss forskningsspecifikationer.
`research/` är ett input-/specifikationslager.
Det är inte en alternativ resultatinfrastruktur.
## Struktur
```text
research/
├── discovery/
│   └── specs/
├── candidates/
│   └── specs/
└── evaluation/
    └── specs/

Ansvar

discovery/

Discovery beskriver hur Blankdiss får söka efter möjliga strukturer och
hypoteser.

Discovery får vara explorativ.

Resultatet från discovery får däremot inte automatiskt betraktas som
validerat.

candidates/

Candidates innehåller frysta forskningskandidater.

En kandidat är en explicit och reproducerbar hypotesdefinition.

När en kandidat har frysts får dess parametrar inte ändras.

En ändring av en kandidatdefinition skapar en ny kandidat.

Exempel:

momentum_60d_quantile: 0.10

ändras till:

momentum_60d_quantile: 0.15

Detta är en ny kandidat.

Den gamla kandidaten ska fortfarande kunna reproduceras.

evaluation/

Evaluation beskriver hur en fryst kandidat ska testas på data som ligger
efter kandidatens freeze-/cutoff-tidpunkt.

Evaluation får mäta, analysera och rapportera.

Evaluation får inte ändra kandidaten.

Dataflöde

DISCOVERY SPEC
      ↓
DISCOVERY RESULT
      ↓
HYPOTHESIS
      ↓
FROZEN CANDIDATE
      ↓
EVALUATION SPEC
      ↓
PROSPECTIVE EVALUATION
      ↓
WALK-FORWARD
      ↓
RESULT

Viktig separation

Specifikationer och resultat ska hållas separerade.

research/
    ↓
YAML
    ↓
Blankdiss
    ↓
data/processed/ml/research/
    ↓
JSON / JSONL

research/ ska därför inte börja innehålla genererade resultatfiler.

Reproducerbarhet

En forskningskörning ska kunna rekonstrueras från:

1. kandidat/specifikation,
2. versionsinformation,
3. data cutoff,
4. feature-version,
5. evaluation-period,
6. körningsmetadata.

Grundregler

1. Discovery får hitta kandidater.
2. En fryst kandidat är immutable.
3. Evaluation får inte optimera kandidaten.
4. Framtida information får inte påverka tidigare beslut.
5. En parameterändring innebär ny kandidat.
6. Resultat skrivs som JSON/JSONL.
7. YAML används för forskningsspecifikationer.
8. research/ är input, inte output.

---
### 2. `research/discovery/specs/README.md`
```markdown
# Discovery Specifications
Discovery-specifikationer beskriver hur Blankdiss får söka efter
potentiella forskningsstrukturer.
Discovery är explorativ.
Ett discovery-resultat är inte en validerad forskningskandidat.
## Syfte
Discovery ska kunna söka över exempelvis:
- signaler,
- signalinteraktioner,
- targets,
- tidsfönster,
- tail-fraktioner,
- kombinationer av befintliga features.
Målet är att hitta strukturer som är tillräckligt intressanta för att
formuleras som explicita hypoteser.
## Exempel
```yaml
id: discovery_momentum_si_001
version: 1
question: >
  Finns det en kombination mellan momentum och förändring i
  short interest som är värd att undersöka vidare?
signals:
  - name: price_return_60d
    directions:
      - upper
      - lower
    fractions:
      - 0.05
      - 0.10
      - 0.20
  - name: short_interest_delta_pp
    directions:
      - upper
      - lower
    fractions:
      - 0.05
      - 0.10
      - 0.20
targets:
  - forward_return_20d
windows:
  - window_1
  - window_2
output:
  propose_candidates: true

Detta är en discovery-specifikation.

Den skapar inte en fryst kandidat.

Discovery får

Discovery får:

* söka över flera parametrar,
* testa många kombinationer,
* identifiera intressanta mönster,
* föreslå hypoteser,
* rangordna discovery-resultat internt för vidare analys.

Discovery får inte

Discovery får inte:

* kalla ett resultat validerat,
* ändra en redan fryst kandidat,
* använda framtida evaluation-resultat,
* skriva över befintliga kandidater,
* göra en evaluation-driven parameterjustering.

Nästa steg

Ett intressant discovery-resultat ska omvandlas till en explicit
controlled hypothesis.

Den hypotesen kan därefter bli en frozen candidate.

Discovery
   ↓
Result
   ↓
Hypothesis
   ↓
Controlled test
   ↓
Frozen candidate
---
### 3. `research/candidates/specs/README.md`
```markdown
# Candidate Specifications
Denna katalog innehåller specifikationer för frysta forskningskandidater.
En candidate-specifikation representerar en explicit hypotes som ska kunna
reproduceras exakt.
## Candidate lifecycle
```text
DISCOVERY
    ↓
CONTROLLED HYPOTHESIS TEST
    ↓
FROZEN CANDIDATE
    ↓
PROSPECTIVE EVALUATION
    ↓
WALK-FORWARD

Immutable

När en kandidat har frysts får dess definition inte ändras.

Exempel:

id: candidate_momentum_si_001
version: 1
signal:
  name: momentum_60d
  quantile: 0.10

Om quantile ändras:

quantile: 0.15

ska det skapas en ny kandidat:

id: candidate_momentum_si_002
version: 1

Den första kandidaten får inte ändras retroaktivt.

Minimum metadata

En candidate-spec ska innehålla:

* id
* version
* question
* created_at
* discovery_cutoff
* freeze_at
* features
* parameters
* target
* training_period
* candidate_status

Exempel

id: candidate_momentum_si_001
version: 1
question: >
  Ger hög 60-dagars momentum kombinerat med positiv förändring
  i short interest en förändrad sannolikhet för framtida prisrörelse?
created_at: "2026-09-28T00:00:00Z"
discovery_cutoff: "2026-06-30"
freeze_at: "2026-07-15"
candidate_status: frozen
features:
  - price_return_60d
  - short_interest_delta_pp
parameters:
  momentum_quantile: 0.10
  short_interest_delta_quantile: 0.10
target:
  name: forward_return_20d
training_period:
  start: "2022-01-01"
  end: "2026-06-30"

Candidate-status

Tillåtna statusar ska vara:

draft
tested
frozen
retired

En kandidat som är frozen får inte ändras.

Om definitionen behöver ändras ska en ny kandidat skapas.

Evaluation

Evaluation ska referera till kandidatens ID och version.

Evaluation ska aldrig innehålla en alternativ kandidatdefinition.

Exempel:

candidate_id: candidate_momentum_si_001
candidate_version: 1

och inte:

candidate_id: candidate_momentum_si_001
parameters:
  momentum_quantile: 0.15

Det senare skulle innebära att evaluation-definitionen skiljer sig från
den frysta kandidaten.

---
### 4. `research/evaluation/specs/README.md`
```markdown
# Evaluation Specifications
Evaluation-specifikationer beskriver hur en fryst forskningskandidat ska
testas på framtida data.
Evaluation ska vara prospektiv.
## Grundregel
Evaluation får läsa:
```text
FROZEN CANDIDATE
       +
FUTURE DATA

Evaluation får inte ändra kandidaten.

Exempel

id: evaluation_candidate_momentum_si_001_2026q3
version: 1
candidate:
  id: candidate_momentum_si_001
  version: 1
evaluation_period:
  start: "2026-07-01"
  end: "2026-09-30"
targets:
  - forward_return_20d
metrics:
  - sample_size
  - event_rate
  - mean_return
  - median_return
  - lift
walk_forward:
  enabled: true

Temporal separation

Evaluation-perioden måste ligga efter kandidatens cutoff.

Exempel:

candidate discovery cutoff
        2026-06-30
             ↓
candidate frozen
        2026-07-15
             ↓
evaluation starts
        2026-07-16

Evaluation får inte använda information från evaluation-perioden för att
ändra:

* features,
* thresholds,
* candidate parameters,
* target definition,
* signaldefinition.

Evaluation-AI

Evaluation-AI får:

* mäta resultat,
* analysera resultat,
* identifiera problem,
* beräkna diagnostik,
* rapportera osäkerhet,
* identifiera stabilitet eller instabilitet.

Evaluation-AI får inte:

* ändra kandidaten,
* optimera kandidatens parametrar,
* välja en ny parameterkombination efter resultatet,
* skapa en bättre kandidat genom att använda evaluation-resultatet.

Om en annan parameterkombination blir intressant ska den bli en ny
candidate.

Walk-forward

En fryst kandidat får testas över flera framtida fönster.

Frozen candidate
      │
      ├── window 1
      ├── window 2
      ├── window 3
      ├── window 4
      └── ...

Samma kandidatdefinition ska användas i samtliga fönster.

Resultat

Evaluation ska producera maskinläsbara resultat.

Exempel:

data/processed/ml/research/
└── <run_id>/
    ├── evaluation.json
    └── manifest.json

Resultaten ska innehålla tillräcklig provenance för att körningen ska
kunna reproduceras.

---
Det här är **första implementationssteget**. Jag skulle inte ändra `blankdiss.yml` ännu. Den nuvarande workflowen har dessutom fortfarande en kommenterad `git commit`/`git push`-sektion — den ska vi ta bort när vi bygger om workflowet. **Den ska inte kunna committa någonting alls.**
Nästa konkreta steg är att implementera **candidate-schema + freeze/verifiering** och därefter koppla den befintliga `ml/research`-motorn till de nya `research/`-specarna.
