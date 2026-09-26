# AI Lab – Next Generation

## Syfte

`ml/ai-lab/` är Blankdiss autonoma forskningsmotor.

Syftet är att låta systemet genomföra en kontrollerad sekvens av forskningsiterationer där varje iteration:

1. observerar tidigare resultat
2. analyserar vad resultaten faktiskt visar
3. väljer nästa kontrollerade experiment
4. skriver en experiment-specifikation
5. läser tillbaka specifikationen
6. kör experimentet
7. skriver resultatet
8. läser tillbaka resultatet
9. analyserar resultatet
10. bestämmer nästa forskningssteg

AI Lab ska alltså inte vara en generator som försöker hitta positiva resultat.

Det ska vara en forskningsmotor som försöker maximera:

idé → information

# Huvudkomponent

Huvudentrén är:

ml/ai-lab/adaptive_research.py

Denna fil äger den adaptiva forskningsloopen.

CI/CD-workflowet ska endast starta processen och hantera artifacts.

Exempel:

python -u ml/ai-lab/adaptive_research.py

Workflowet ska inte innehålla forskningslogik.

# Forskningsloopen

Den övergripande modellen är:

OBSERVE
   ↓
ANALYZE
   ↓
ADAPT
   ↓
WRITE SPEC
   ↓
READ SPEC
   ↓
EXPERIMENT
   ↓
WRITE RESULT
   ↓
READ RESULT
   ↓
ANALYZE
   ↓
ADAPT

När experimentutrymmet är uttömt:

EXTEND

Det är en verklig forskningsövergång, inte ett tekniskt stopp.

# OBSERVE

OBSERVE läser det persistenta forskningsläget.

Det ska bland annat kunna läsa:

- tidigare experiment
- tidigare specs
- tidigare resultat
- tidigare utfallsklassificeringar
- vilka parametrar som redan testats
- vilket experimentutrymme som återstår
- eventuell tidigare forskningsriktning

Observationen ska bygga på faktiskt sparade artifacts.

Systemet ska inte anta att en iteration lyckades bara för att den genomfördes.

# ANALYZE

ANALYZE tolkar resultaten.

Analysen ska vara neutral.

Exempel:

OUTCOME_POSITIVE

kan innebära att den undersökta hypotesen fått stöd i den aktuella discovery-evalueringen.

Men:

OUTCOME_NEGATIVE

är lika mycket ett forskningsresultat.

Även:

OUTCOME_MIXED

och:

OUTCOME_INCONCLUSIVE

är giltig information.

Analysen ska därför inte vara konstruerad som:

resultat
→ positivt?
→ ja → fortsätt
→ nej → ignorera

utan:

resultat
→ vad lärde vi oss?
→ vad är nu känt?
→ vad återstår att undersöka?

# ADAPT

ADAPT väljer nästa experiment utifrån det aktuella forskningsläget.

Det betyder inte att systemet får hitta på godtyckliga parametrar.

Experimentet måste komma från ett definierat experimentutrymme.

Exempel:

parameters:
  momentum_days:
    values:
      - 3
      - 5
      - 10

  si_threshold:
    values:
      - 20
      - 30
      - 40

AI Lab får välja bland de definierade alternativen.

När ett alternativ redan har testats ska det kunna identifieras som förbrukat.

# Resultatklassificering

Experimentresultat klassificeras i fyra huvudkategorier:

OUTCOME_POSITIVE
OUTCOME_MIXED
OUTCOME_NEGATIVE
OUTCOME_INCONCLUSIVE

Klassificeringen ska baseras på fördefinierade kriterier.

Den ska inte ändras efter att resultatet blivit känt för att göra utfallet mer positivt.

# Ingen result hunting

AI Lab får inte implementeras som en sökmotor efter positiva resultat.

Följande mönster är förbjudet:

kör många kandidater
↓
sortera efter positiv effekt
↓
välj bästa
↓
sök närliggande parametrar
↓
upprepa

Detta riskerar att göra hela processen till ett adaptivt urval baserat på utfallet.

I stället:

definiera experimentutrymme
↓
observera tidigare resultat
↓
välj nästa legitima experiment
↓
spara resultat
↓
analysera även negativa resultat
↓
uppdatera forskningsläge

# Iterationer ska ske i kod

Forskningsiterationer ska ske i:

ml/ai-lab/adaptive_research.py

och relaterade Python-moduler.

Workflowet ska inte implementera exempelvis:

iteration 1
iteration 2
iteration 3

som separata workflow-steg.

Det ska inte heller finnas en godtycklig mekanism som:

MAX_ADAPTIVE_ITERATIONS = 3

som ersättning för faktisk forskningslogik.

Om en begränsning av antal iterationer behövs ska den vara en explicit del av körningens kontroll eller forskningsspecifikationen, inte en ersättning för att förstå när forskningsutrymmet faktiskt är uttömt.

# Diskbaserad iteration

Varje iteration ska ha tydliga persistenta gränser.

Den avsedda sekvensen är:

create spec
    ↓
write spec
    ↓
read spec
    ↓
run experiment
    ↓
write result
    ↓
read result
    ↓
analyze
    ↓
select next step

Detta gör att forskningen kan granskas efteråt.

Det gör också att ett avbrott inte behöver innebära att hela forskningshistoriken försvinner.

# Specifikationer

Adaptive specs skrivs till:

ml/research/specs/adaptive_*.yaml

En spec ska beskriva vad som ska testas.

Den ska inte bara vara en intern Python-datastruktur som aldrig sparas.

När specen har skrivits ska AI Lab läsa tillbaka den från disk innan experimentet körs.

Detta säkerställer att det som körs är samma definition som faktiskt sparades som forskningsartifact.

# Resultat

Experimentresultat sparas under:

data/processed/ml/research/spec_runs/

Resultatet ska därefter läsas tillbaka av AI Lab.

Analysen ska baseras på det persistenta resultatet och inte på ett osynligt internt objekt från föregående funktionsanrop.

# State

AI Lab state finns under:

data/ai_lab/adaptive_research/

med huvudfil:

data/ai_lab/adaptive_research/state.json

State ska göra det möjligt att förstå:

- aktuell forskningsfråga
- tidigare experiment
- testade parameterkombinationer
- kvarvarande experimentutrymme
- tidigare utfall
- aktuell forskningsfas
- eventuell EXTEND-övergång

# Discovery kontra confirmation

AI Lab är primärt en discovery-motor.

Discovery får vara adaptiv.

Confirmation ska däremot vara skyddad och reproducerbar.

Exempel på skyddad confirmation-spec:

ml/research/specs/momentum_si_prospective_confirmation.yaml

AI Lab får inte ändra confirmation-specifikationen baserat på discovery-resultat.

Discovery:

adaptiv
explorativ
hypotesgenererande

Confirmation:

låst
prospektiv
reproducerbar

# Testdata

Testdata får inte användas som adaptiv feedbackloop.

Testdata får inte användas för att välja:

- parametrar
- features
- trösklar
- endpoints
- hypoteser
- experimentfamiljer

Om testresultatet används för att bestämma vad som ska testas härnäst har testdata i praktiken blivit en del av discoveryprocessen.

Det ska undvikas.

# När parametrarna tar slut

Ett centralt krav är att systemet faktiskt ska förstå när det definierade experimentutrymmet är slut.

Exempel:

momentum = 3, 5, 10
SI = 20, 30, 40

ger ett ändligt antal kombinationer.

När alla relevanta kombinationer är genomförda finns det inte längre ett legitimt:

nästa parameter

inom samma experimentfamilj.

Systemet ska då inte hitta på nya värden bara för att fortsätta loopen.

# EXTEND

När experimentutrymmet är uttömt går forskningsprocessen till:

EXTEND

EXTEND betyder att den nuvarande forskningsmodellen inte längre innehåller tillräckligt med information att hämta.

Nästa steg ska därför formuleras som en ny forskningsfråga.

Exempel:

Nuvarande fråga:
Förbättrar momentum signal X?

Experimentutrymmet kan vara:

momentumfönster × SI-tröskel

Om detta utrymme är uttömt och resultaten är negativa eller blandade kan nästa fråga exempelvis vara:

Beror effekten på marknadsregim?

eller:

Är effekten koncentrerad kring rapportdatum?

eller:

Är effekten cross-sectional snarare än time-series-baserad?

Det viktiga är att frågan representerar ny information.

# När ny kod behövs

Om den nya forskningsfrågan inte kan uttryckas med befintlig experimentlogik ska AI Lab inte tvinga in den i en gammal experimentfamilj.

Då är nästa steg:

ny forskningsfråga
    ↓
ny experimentdesign
    ↓
ny experimentmodul

Exempel:

ml/research/discovery/adaptive_*.py

eller annan lämplig modulstruktur.

Ny kod ska alltså vara ett resultat av ett identifierat forskningsbehov.

Inte:

negativt resultat
→ skriv mer kod

utan:

negativt resultat
→ förstå vad som saknas
→ formulera ny fråga
→ avgör om befintlig kod räcker
→ skapa ny kod endast om den behövs

# Research Engine

Generell forskningsfunktionalitet ska ligga i:

ml/research/

Exempel:

spec.py
session.py
cache.py
signals.py
engine.py
runner.py
reporting.py
bootstrap.py

Om en ny funktion behövs av flera typer av forskning bör den normalt läggas i Research Engine.

En konkret hypotes ska däremot normalt uttryckas genom en spec.

# Diagnostics

Specialiserade analyser hör hemma under:

ml/diagnostics/

Exempel:

ml/diagnostics/framework/
ml/diagnostics/experiments/

Diagnostics ska inte bli en parallell generell forskningsmotor.

# Forskningsintegritet

AI Lab ska kunna svara på:

Vad testade vi?

Varför testade vi det?

Vilken data användes?

Vilken spec kördes?

Vad blev resultatet?

Hur klassificerades resultatet?

Vad testade vi därefter?

Varför gick vi vidare till en ny forskningsfråga?

Detta är viktigare än att systemet producerar många experiment.

# Forskningscykel

Den fullständiga cykeln kan beskrivas:

RESEARCH QUESTION
        ↓
OBSERVE
        ↓
ANALYZE
        ↓
ADAPT
        ↓
EXPERIMENT SPEC
        ↓
WRITE
        ↓
READ
        ↓
EXPERIMENT
        ↓
RESULT
        ↓
WRITE
        ↓
READ
        ↓
ANALYZE
        ↓
experiment kvar?
        │
     ┌──┴──┐
    yes    no
     │      │
     ↓      ↓
   ADAPT   EXTEND
              │
              ↓
   NEW RESEARCH QUESTION
              │
              ↓
   NEW EXPERIMENT FAMILY
              │
              ↓
   NEW CODE IF REQUIRED

# Workflowets ansvar

Workflowet ska vara tunt.

Det ska huvudsakligen:

1. installera miljö
2. förbereda data/miljö vid behov
3. starta AI Lab
4. samla artifacts
5. eventuellt commit/push

Det ska inte innehålla:

- kandidatval
- parameteroptimering
- resultat-ranking
- hypotesgenerering
- forskningsloop
- positiv-resultat-filtrering

Den centrala körningen ska vara:

python -u ml/ai-lab/adaptive_research.py

# Designregel

En viktig arkitekturregel är:

idé → information

inte:

idé → mer kod

AI Lab ska därför inte mätas i antal genererade experiment eller antal genererade Python-filer.

Det ska mätas i hur mycket osäkerhet som försvinner genom kontrollerade experiment.

# Relaterad dokumentation

Implementation:

ml/ai-lab/adaptive_research.py

AI Lab README:

ml/ai-lab/README.md

Research Engine:

ml/research/

Research specifications:

ml/research/specs/

Research results:

data/processed/ml/research/spec_runs/

AI Lab state:

data/ai_lab/adaptive_research/
