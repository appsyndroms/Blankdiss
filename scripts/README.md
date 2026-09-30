Ja. Jag skulle lägga den som scripts/README.md. Den bör beskriva både --force för featurebygget och den mer generella principen: scripts är återanvändbara byggblock som workflows orkestrerar, medan den egentliga domänlogiken ligger i Python-modulerna.

Här är hela README:n:

# Scripts
Katalogen `scripts/` innehåller körbara hjälpskript som används av
Blankdiss workflows.
## Grundprincip
Blankdiss skiljer på tre nivåer:
1. **Python-moduler**
   - Innehåller den egentliga domän- och affärslogiken.
   - Exempelvis featurebygge, research, evaluation och QC.
   - Ska kunna köras och testas utan att vara beroende av GitHub Actions.
2. **Scripts**
   - Är tunna körbara lager ovanpå Python-moduler eller andra verktyg.
   - Hanterar sådant som behöver återanvändas mellan workflows.
   - Ska inte innehålla större mängder domänlogik.
3. **Workflows**
   - Orkestrerar hela arbetsflöden.
   - Bestämmer ordning, triggers, miljö, permissions, artifacts,
     commits och push.
   - Anropar Python-moduler och scripts när respektive steg ska utföras.
Principen är därför:
```text
GitHub Actions workflow
        │
        ├── script
        │      │
        │      └── Python-modul / verktyg
        │
        └── Python-modul direkt

Workflows ska alltså beskriva vad som ska köras och i vilken ordning,
medan scripts och Python-moduler beskriver hur själva operationen
genomförs.

⸻

Scripts ska vara tunna

Ett script bör i första hand:

* anropa en befintlig Python-modul
* sätta upp ett återanvändbart kommando
* hantera shell-specifik orkestrering
* kapsla in återkommande Git-operationer
* returnera korrekt exit code

Ett script bör normalt inte:

* implementera omfattande featurelogik
* duplicera Python-kod
* innehålla analyslogik som egentligen hör hemma i analysis/
* känna till detaljer om GitHub Actions som bara gäller ett specifikt
    workflow

Om logiken börjar bli omfattande är det normalt bättre att flytta den
till en Python-modul och låta scriptet bli ett tunt anropslager.

⸻

Scripts och workflows

Workflows använder scripts för steg som behöver vara konsekventa mellan
olika körningar eller workflows.

Exempel:

- name: Build features
  run: |
    bash scripts/build_features.sh

Workflowet behöver då inte känna till detaljerna för hur featurebygget
startas.

Samtidigt kan samma funktion köras lokalt:

bash scripts/build_features.sh

Det gör att lokal körning och workflow-körning kan använda samma
ingångspunkt.

När det däremot räcker med ett enkelt kommando kan workflowet anropa
Python-modulen direkt:

- name: Build features
  run: |
    python -u -m analysis.build_features

Det finns alltså inget krav på att varje Python-modul måste ha ett
motsvarande shell-script.

⸻

Featurebygget

Featurebygget finns i:

analysis/build_features.py

Det kan köras direkt:

python -u -m analysis.build_features

eller via ett script om workflowet använder ett sådant.

Normal körning

Vid normal körning kontrollerar featurebygget dagens datum i
Europe/Stockholm.

Om dagens snapshot_date redan finns i ett av de kanoniska
feature-chunken:

data/processed/analysis/features_*.jsonl

avslutas bygget utan att datasetet byggs om.

Exempel:

Feature dataset finns redan för 2026-09-30 – hoppar över build.

Det gör den dagliga körningen idempotent när dagens features redan har
producerats.

Full rebuild med --force

Om featuredefinitionen har ändrats, en ny feature har lagts till eller
historiska features av annan anledning behöver räknas om kan bygget
tvingas fram:

python -u -m analysis.build_features --force

--force innebär:

ignorera kontrollen av dagens datum
        ↓
bygg om hela feature-datasetet

Detta är avsiktligt en full rebuild. Befintliga canonical chunks ersätts
av resultatet från den nya körningen.

Exempel på när --force är relevant:

* en ny feature har lagts till
* definitionen av en befintlig feature har ändrats
* en historisk datakälla har korrigerats
* en bug i featurelogiken har rättats
* features behöver räknas om från början

Det innebär att vi inte behöver bygga in komplicerad
versionshantering för features för att kunna göra en full omräkning.

⸻

Varför dagens datum används

Feature-datasetet byggs med snapshot_date som den centrala tidsaxeln.

Den normala dagliga kontrollen är därför enkel:

Finns dagens snapshot_date?
        │
        ├── ja → gör ingenting
        │
        └── nej → bygg datasetet

Om dagens datum saknas görs en full rebuild enligt den befintliga
byggprocessen.

Det innebär också att eventuella historiska luckor normalt korrigeras
automatiskt när ett fullständigt bygge görs.

⸻

--force och workflows

Det normala dagliga workflowet ska inte behöva använda --force.

Exempel:

- name: Build features
  run: |
    python -u -m analysis.build_features

Vid behov kan ett manuellt workflow däremot exponera force-läget.

Exempel:

workflow_dispatch:
  inputs:
    force:
      description: "Force full feature rebuild"
      required: false
      default: false
      type: boolean

Workflowet kan då välja mellan:

force = false
    ↓
normal körning

och:

force = true
    ↓
python -u -m analysis.build_features --force

Det gör att en full rebuild kan triggas manuellt utan att den normala
dagliga körningen förändras.

⸻

Git och commits

Scripts ska inte i normalfallet själva committa eller pusha ändringar.

Git-operationer som är en del av ett workflow ska ligga på workflow-nivå
eller i särskilt avsedda återanvändbara Git-scripts.

Exempel:

Python
  ↓
skapar data
workflow
  ↓
kontrollerar resultat
workflow
  ↓
git add / commit
workflow
  ↓
push

Det gör det tydligt när data faktiskt publiceras till repositoryt.

Ett exempel på ett återanvändbart Git-script är:

scripts/git_push_with_retry.sh

Det kapslar in push/rebase/retry-beteende som annars lätt skulle
dupliceras mellan workflows.

⸻

Exit codes

Scripts ska använda exit codes för att kommunicera resultat till
workflowet.

Grundprincip:

0 = lyckades
!= 0 = fel

Ett workflow ska kunna avbrytas automatiskt när ett script misslyckas.

Exempel:

set -e

bör användas där ett fel ska stoppa det fortsatta workflowet.

⸻

Idempotens

Där det är möjligt bör scripts och de Python-kommandon de startar vara
idempotenta.

Det innebär att samma steg kan köras igen utan att resultatet blir
inkonsekvent.

Featurebygget är ett exempel:

normal körning
    ↓
dagens datum finns
    ↓
ingen rebuild

eller:

normal körning
    ↓
dagens datum saknas
    ↓
full rebuild

och:

--force
    ↓
full rebuild oavsett tidigare resultat

Detta gör workflows säkrare att köra om efter exempelvis ett
nätverksfel, timeout eller misslyckad senare del av pipeline.

⸻

Lokal körning

Ett viktigt mål är att workflowsteg ska kunna reproduceras lokalt.

Om workflowet exempelvis innehåller:

- name: Build features
  run: |
    python -u -m analysis.build_features

ska samma kommando kunna köras från repositoryts rot:

python -u -m analysis.build_features

Och vid en avsiktlig full rebuild:

python -u -m analysis.build_features --force

På så sätt är GitHub Actions framför allt en automatiserad
orkestreringsmiljö, inte den enda plats där Blankdiss kan köras.

⸻

Katalogstruktur

Den övergripande strukturen är tänkt ungefär så här:

.github/
└── workflows/
    ├── blankdiss.yml
    └── feature-build.yml
analysis/
├── build_features.py
├── feature_config.py
├── features_qc.py
└── ...
scripts/
├── README.md
├── build_features.sh
├── git_push_with_retry.sh
└── ...
tests/
└── ...

Exakta scripts kan ändras över tid. Principen är viktigare än ett fast
antal filer.

⸻

Tumregel

När ny funktionalitet ska läggas till:

Är det domänlogik?

Lägg den i Python.

analysis/

Är det ett återanvändbart shell-/CLI-anrop?

Överväg ett script.

scripts/

Är det ordningen mellan flera steg, triggers, permissions,

artifacts eller commits?

Lägg det i workflowet.

.github/workflows/

Behöver samma operation användas av flera workflows?

Försök centralisera den i ett script eller en gemensam Python-modul
i stället för att duplicera implementationen.

⸻

Sammanfattning

Blankdiss ska eftersträva denna ansvarsfördelning:

┌──────────────────────────────┐
│          WORKFLOW             │
│                              │
│  När? I vilken ordning?      │
│  Vilken miljö?               │
│  Commit/push/artifacts?      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│           SCRIPTS            │
│                              │
│  Återanvändbara kommandon    │
│  Shell-orkestrering          │
│  Gemensamma operationer      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│        PYTHON / ANALYSIS     │
│                              │
│  Domänlogik                  │
│  Databearbetning             │
│  Features                    │
│  Research                    │
│  Evaluation                  │
└──────────────────────────────┘

Målet är att varje lager ska ha ett tydligt ansvar och att samma
funktionalitet inte ska behöva implementeras på flera ställen.
