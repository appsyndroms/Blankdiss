AI Lab Results

Den här katalogen innehåller resultat och artefakter från Blankdiss AI Lab.

AI Lab är den del av projektet som står för mer kontrollerad och adaptiv forskningsorkestrering. Det innebär att AI Lab kan planera experiment, hålla state mellan körningar och arbeta iterativt med hypoteser och kandidater.

Rollen i arkitekturen

AI Lab ska skiljas från den generella Research Engine.

AI Lab
  │
  │ planerar och styr forskning
  ▼
Research Engine
  │
  │ kör deklarativa experiment
  ▼
Research results
  │
  ▼
Candidates
  │
  ▼
Prospective evaluation

Research Engine innehåller den generella logiken för att genomföra experiment.

AI Lab ligger på en högre nivå och kan använda Research Engine för att genomföra de experiment som AI Lab planerar.

Resultat

Resultat som hör till AI Labs egna forskningsloopar kan lagras här.

Det kan exempelvis vara:

* experimentresultat
* state-relaterade artefakter
* sammanställningar
* planeringsresultat
* mellanresultat från forskningsloopar
* metadata som behövs för att förstå en AI Lab-körning

Den exakta uppsättningen filer kan förändras när AI Lab utvecklas.

Skillnad mot Research Engine-resultat

Research Engine-resultat hör normalt hemma under:

data/processed/ml/research/

AI Lab-resultat hör hemma här när de representerar artefakter från AI Labs högre nivå av orkestrering.

Det innebär att samma forskningskörning potentiellt kan ha:

data/processed/ml/research/
    └── själva experimentresultatet
data/ai_lab/results/
    └── AI Labs state, planering eller sammanställning

De två lagren ska inte blandas ihop.

Skillnad mot evaluation

Prospektiv kandidat-utvärdering hör hemma under:

data/processed/ml/research/evaluation/

Det är viktigt eftersom ett AI Lab-resultat inte automatiskt innebär att en kandidat har blivit validerad.

En typisk livscykel är:

Hypotes
   │
   ▼
AI Lab planerar experiment
   │
   ▼
Research Engine
   │
   ▼
Historiskt resultat
   │
   ▼
Candidate
   │
   ▼
Prospektiv evaluation
   │
   ▼
Verification

Reproducerbarhet

AI Lab-resultat ska i möjligaste mån kunna kopplas till:

* den forskning som genomfördes
* Research Engine-run
* relevant spec
* kandidat
* datagrund
* tidsperiod
* eventuell evaluation

Det gör det möjligt att skilja mellan ett resultat som endast var intressant under forskning och ett resultat som senare verifierats.

Vad katalogen inte är

data/ai_lab/results/ är inte det generella lagret för alla ML-resultat.

Använd i stället:

data/processed/ml/

för övriga ML-resultat och:

data/processed/ml/research/

för Research Engine-relaterade resultat.

AI Lab-katalogen ska främst innehålla artefakter som hör till AI Labs egen orkestrering och state.

Relaterad kod

Den huvudsakliga AI Lab-koden finns under:

ml/ai-lab/

Research Engine finns under:

ml/research/

Prospektiv evaluation finns under:

ml/research/evaluation/

Detta ger en tydlig separation mellan:

* AI Lab — styrning och adaptiv forskning
* Research Engine — experimentmotor
* Candidates — frysta forskningshypoteser
* Evaluation — prospektiv testning
* Verification — kontroll av resultat
