# AI Lab – Next Generation

AI Lab är Blankdiss autonoma forskningsmotor.

Den ansvarar för den adaptiva forskningsloopen och ska själv hantera iterationer, specs, resultat och nästa forskningssteg.

Workflowet ska endast starta AI Lab och hantera artifacts.

## Körning

Kör:

python -u ml/ai-lab/adaptive_research.py

Forskningsloopen ska ske i Python-koden.

Workflowet ska inte implementera separata forskningsiterationer.

## Kärnprinciper

- Forskningslogiken ligger i Python, inte i workflowet.
- Iterationer ska vara diskbaserade.
- Specs skrivs till disk.
- Specs läses tillbaka innan de körs.
- Resultat skrivs till disk.
- Resultat läses tillbaka innan nästa beslut.
- Negativa resultat är giltiga forskningsresultat.
- Mixed-resultat är giltiga forskningsresultat.
- Inconclusive-resultat är giltiga forskningsresultat.
- AI Lab får inte jaga positiva resultat.
- Testdata får inte styra discovery.
- Confirmation-specifikationer ska vara skyddade.
- När parameterutrymmet är slut ska systemet gå vidare till en ny forskningsfråga.
- Ny kod ska skapas när ett identifierat forskningsbehov inte kan uttryckas med befintlig experimentlogik.

## Forskningsloop

Den normala loopen är:

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
nästa experiment

När det definierade experimentutrymmet är uttömt:

EXTEND

EXTEND innebär att nästa steg normalt är en ny forskningsfråga eller en ny experimentfamilj.

## Resultat

Resultat klassificeras som:

OUTCOME_POSITIVE
OUTCOME_MIXED
OUTCOME_NEGATIVE
OUTCOME_INCONCLUSIVE

Alla fyra är legitima resultat.

AI Lab ska inte filtrera bort negativa eller blandade resultat.

## Ingen result hunting

AI Lab ska inte fungera enligt:

testa kandidater
→ hitta positiv kandidat
→ optimera runt positiv kandidat
→ upprepa

I stället:

definiera experimentutrymme
→ testa kontrollerat
→ spara resultat
→ analysera
→ uppdatera forskningsläge
→ välj nästa legitima experiment

## Diskbaserad iteration

Varje iteration ska följa:

create spec
→ write spec
→ read spec
→ run experiment
→ write result
→ read result
→ analyze
→ next step

Det gör forskningen reproducerbar och möjlig att granska.

## Filer

AI Lab:

ml/ai-lab/
├── README.md
└── adaptive_research.py

Adaptive specs:

ml/research/specs/adaptive_*.yaml

Adaptive experiment code:

ml/research/discovery/adaptive_*.py

AI Lab state:

data/ai_lab/adaptive_research/state.json

Research results:

data/processed/ml/research/spec_runs/

## Research Engine

Generell forskningslogik ska ligga i:

ml/research/

AI Lab ska använda Research Engine där befintlig funktionalitet räcker.

Om en ny forskningsfråga kräver en generell ny funktion ska den normalt läggas i Research Engine.

Om frågan är specialiserad kan ett separat experiment eller diagnostic användas.

## Discovery och confirmation

Discovery är adaptiv.

Confirmation är skyddad.

Exempel:

ml/research/specs/momentum_si_prospective_confirmation.yaml

AI Lab får inte ändra confirmation-specifikationen baserat på discovery-resultat.

## Testdata

Testdata får inte användas för att välja:

- parametrar
- features
- trösklar
- endpoints
- hypoteser
- experimentfamiljer

Testdata ska förbli separat från den adaptiva discovery-loopen.

## När experimentutrymmet tar slut

AI Lab ska känna igen när alla definierade alternativ är förbrukade.

Det ska inte skapa godtyckliga nya parametrar bara för att fortsätta iterationerna.

I stället:

parameterutrymme slut
→ analysera vad resultaten säger
→ identifiera vad som saknas
→ formulera ny forskningsfråga
→ avgör om befintlig kod räcker
→ skapa ny experimentkod om det behövs

## Designprincip

AI Labs viktigaste princip är:

idé → information

inte:

idé → mer kod

Ny kod är därför ett medel, inte målet.

Målet är att minska osäkerheten genom kontrollerade experiment.

## Relaterad dokumentation

Se:

AI_LAB_NEXT_GENERATION.md

för den fullständiga arkitekturbeskrivningen.
