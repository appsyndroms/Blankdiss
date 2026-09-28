Blankdiss Roadmap

Mål

Blankdiss ska ha en reproducerbar forskningskedja:

DATA
 ↓
FEATURES
 ↓
DISCOVERY
 ↓
CONTROLLED HYPOTHESIS
 ↓
FROZEN CANDIDATE
 ↓
PROSPECTIVE EVALUATION
 ↓
WALK-FORWARD
 ↓
VERIFICATION
 ↓
JSON / JSONL

Kvar att göra

1. Research-arkitekturen

* [ ]	Slutför separationen mellan discovery, candidate och evaluation.
* [ ]	Koppla befintlig ml/research-motor till candidate/evaluation-modellen.
* [ ]	Bestäm vilka gamla research-specs/runners som ska återanvändas respektive bli legacy.
* [ ]	Samla spec-versionering och provenance på ett konsekvent sätt.

2. Frozen candidates

* [ ]	Slutför candidate freeze-flödet.
* [ ]	Beräkna och verifiera fingerprint vid freeze.
* [ ]	Säkerställ att en fryst kandidat inte kan muteras.
* [ ]	Säkerställ att ändrad definition alltid blir en ny kandidat.

3. Prospective evaluation

* [ ]	Slutför evaluation-runnern mot den frysta kandidaten.
* [ ]	Säkerställ temporal separation:
    training_end <= discovery_cutoff < freeze_at < evaluation_start
* [ ]	Evaluation får endast använda kandidatens definition.
* [ ]	Spara komplett provenance i resultatet.

4. Walk-forward

* [ ]	Definiera slutligt YAML-format för walk-forward.
* [ ]	Kör samma frozen candidate över flera framtida fönster.
* [ ]	Säkerställ att varje fönster är tidsmässigt isolerat.
* [ ]	Spara resultat per fönster.
* [ ]	Skapa aggregerade resultat.
* [ ]	Lägg till stabilitets- och sample-size-information.

5. Verification

* [ ]	Samlad verification för:
    * [ ]	temporal leakage
    * [ ]	feature/target leakage
    * [ ]	future-data access
    * [ ]	candidate mutation
    * [ ]	evaluation contamination
    * [ ]	duplicate runs
    * [ ]	saknad data
    * [ ]	distributionsförändringar
* [ ]	Kritiska brott ska stoppa körningen.

6. Gemensam Blankdiss-entrypoint

* [ ]	Skapa gemensam Python-entrypoint.
* [ ]	Samma process ska fungera lokalt och i CI.
* [ ]	Definiera run ID, logging och exit codes.
* [ ]	Entry-pointen ska orkestrera befintliga komponenter, inte innehålla forskningslogik.

7. Daglig GitHub Actions-körning

* [ ]	Bygg nya blankdiss.yml kring entrypointen.
* [ ]	Daglig trigger + workflow_dispatch.
* [ ]	Data → QC → features → discovery → candidate → evaluation → walk-forward → verification.
* [ ]	Spara artifacts och run metadata.
* [ ]	Workflowet ska aldrig innehålla forskningslogik.

8. Legacy och städning

* [ ]	Kartlägg gamla runners mot den nya livscykeln.
* [ ]	Ersätt först, verifiera sedan.
* [ ]	Ta bort verklig legacy först när den inte längre behövs.
* [ ]	Ingen blind filborttagning.

Slutpunkt

En daglig körning ska kunna reproducera:

Vad visste Blankdiss?
        ↓
Vad upptäckte den?
        ↓
Vilken hypotes skapades?
        ↓
Vilken kandidat frystes?
        ↓
Vilken framtida data testades?
        ↓
Hur höll kandidaten över flera perioder?
        ↓
Kan hela resultatet reproduceras?

Grundregler

1. Frysta kandidater ändras aldrig.
2. Evaluation får inte optimera kandidaten.
3. Ingen framtidsinformation får påverka tidigare beslut.
4. YAML = specifikation.
5. JSON/JSONL = resultat.
6. Python = forskningslogik.
7. GitHub Actions = orkestrering.
8. Förändring av kandidatdefinition = ny kandidat.
9. Ingen fil tas bort innan ersättning och verifiering är klar.
