from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
def file_sha256(path: Path) -> str:
    """Beräknar SHA-256 för en fil."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()
def build_source_fingerprint(
    *,
    fi_path: Path,
    price_files: list[Path],
) -> dict[str, object]:
    """
    Skapar ett deterministiskt fingerprint av det underlag
    som används för att bygga feature-datasetet.
    """
    files = [
        fi_path,
        *sorted(price_files),
    ]
    entries = []
    for path in files:
        entries.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    canonical = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    fingerprint = hashlib.sha256(
        canonical
    ).hexdigest()
    return {
        "algorithm": "sha256",
        "fingerprint": fingerprint,
        "files": entries,
    }

Sedan ändrar du write_metadata() så att den tar emot fingerprintet:

def write_metadata(
    *,
    fi: pd.DataFrame,
    prices: pd.DataFrame,
    result: pd.DataFrame,
    stats: dict[str, int],
    price_files: list[Path],
    chunks: list[dict[str, object]],
    source_fingerprint: dict[str, object],
) -> None:

och lägger in detta i metadata:

"source_fingerprint": source_fingerprint,

Slutligen, i main(), direkt efter att price_files har hittats:

source_fingerprint = build_source_fingerprint(
    fi_path=FI_PATH,
    price_files=price_files,
)

och skicka sedan med det till write_metadata():

write_metadata(
    fi=fi,
    prices=prices,
    result=result,
    stats=stats,
    price_files=price_files,
    chunks=chunks,
    source_fingerprint=source_fingerprint,
)

Det gör att metadata exempelvis kommer innehålla:

"source_fingerprint": {
  "algorithm": "sha256",
  "fingerprint": "...",
  "files": [
    {
      "path": "data/processed/fi/aggregate/reconstructed.jsonl",
      "sha256": "...",
      "size_bytes": 12345678
    },
    {
      "path": "data/raw/prices/prices_2022-01-01_latest.jsonl",
      "sha256": "...",
      "size_bytes": 12345678
    }
  ]
}

2. Research-workflowet

Här är den viktiga delen. Efter Fetch FI aggregate och Fetch prices beräknar vi aktuellt fingerprint och jämför med det sparade.

Om samma → ingen feature build.

Om olika → Build features + QC.

:::writing{variant=“document” id=“92754” title=“Blankdiss Research workflow”}

name: Blankdiss Research
on:
  workflow_dispatch:
permissions:
  contents: write
concurrency:
  group: research-manual
  cancel-in-progress: false
jobs:
  research:
    name: Kör Blankdiss Research Matrix
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Clean latest research results
        run: |
          mkdir -p data/processed/ml/research/latest/diagnostic
          find data/processed/ml/research/latest -type f -delete
      - name: Fetch FI aggregate
        run: |
          python -u -m fi
      - name: Fetch prices
        run: |
          python -m prices --start 2022-01-01
      - name: Check whether features need rebuilding
        id: feature_check
        run: |
          python - <<'PY'
          import hashlib
          import json
          from pathlib import Path
          ROOT = Path(".").resolve()
          fi_path = (
              ROOT
              / "data"
              / "processed"
              / "fi"
              / "aggregate"
              / "reconstructed.jsonl"
          )
          price_dir = (
              ROOT
              / "data"
              / "raw"
              / "prices"
          )
          metadata_path = (
              ROOT
              / "data"
              / "processed"
              / "analysis"
              / "features_metadata.json"
          )
          def file_sha256(path):
              digest = hashlib.sha256()
              with path.open("rb") as handle:
                  for chunk in iter(
                      lambda: handle.read(1024 * 1024),
                      b"",
                  ):
                      digest.update(chunk)
              return digest.hexdigest()
          price_files = sorted(
              price_dir.glob("prices_*.jsonl")
          )
          source_files = [
              fi_path,
              *price_files,
          ]
          if not fi_path.exists():
              raise SystemExit(
                  f"Saknar FI-data: {fi_path}"
              )
          entries = []
          for path in source_files:
              if not path.exists():
                  raise SystemExit(
                      f"Saknar datafil: {path}"
                  )
              entries.append(
                  {
                      "path": str(
                          path.relative_to(ROOT)
                      ),
                      "sha256": file_sha256(path),
                      "size_bytes": path.stat().st_size,
                  }
              )
          canonical = json.dumps(
              entries,
              ensure_ascii=False,
              sort_keys=True,
              separators=(",", ":"),
          ).encode("utf-8")
          current_fingerprint = hashlib.sha256(
              canonical
          ).hexdigest()
          old_fingerprint = None
          if metadata_path.exists():
              metadata = json.loads(
                  metadata_path.read_text(
                      encoding="utf-8"
                  )
              )
              old_fingerprint = (
                  metadata
                  .get("source_fingerprint", {})
                  .get("fingerprint")
              )
          changed = (
              old_fingerprint
              != current_fingerprint
          )
          print(
              "Feature source fingerprint:"
          )
          print(
              f"  Previous: {old_fingerprint}"
          )
          print(
              f"  Current:  {current_fingerprint}"
          )
          print(
              f"  Changed:  {changed}"
          )
          with open(
              "feature_check.env",
              "w",
              encoding="utf-8",
          ) as handle:
              handle.write(
                  f"FEATURES_CHANGED={'true' if changed else 'false'}\n"
              )
              handle.write(
                  f"FEATURE_SOURCE_FINGERPRINT={current_fingerprint}\n"
              )
          print(
              "FEATURES_CHANGED="
              + (
                  "true"
                  if changed
                  else "false"
              )
          )
          PY
          cat feature_check.env >> "$GITHUB_ENV"
      - name: Build features
        if: env.FEATURES_CHANGED == 'true'
        run: |
          python -u -m analysis.build_features
      - name: Inspect feature dataset
        if: env.FEATURES_CHANGED == 'true'
        run: |
          python -m analysis.ci inspect-features
      - name: Verify feature dataset
        if: env.FEATURES_CHANGED == 'true'
        run: |
          python -m analysis.ci verify-features
      - name: Run Feature QC
        if: env.FEATURES_CHANGED == 'true'
        run: |
          python -u -m analysis.features_qc
      - name: Verify Feature QC
        if: env.FEATURES_CHANGED == 'true'
        run: |
          python -m analysis.ci verify-feature-qc
      - name: Verify feature dataset exists
        run: |
          test -f data/processed/analysis/features_0001.jsonl
          test -f data/processed/analysis/features_metadata.json
      - name: Run Research Matrix
        run: |
          python -u -m ml.research.runner
      - name: Run Registered Experiments
        run: |
          python -u -m ml.experiment_registry_runner
      - name: Verify research results
        run: |
          test -f data/processed/ml/research/latest/results.jsonl
          test -f data/processed/ml/research/latest/pooled.json
          test -f data/processed/ml/research/latest/metadata.json
          test -f data/processed/ml/research/latest/report.md
          python - <<'PY'
          import json
          from pathlib import Path
          base = Path(
              "data/processed/ml/research/latest"
          )
          results_path = base / "results.jsonl"
          pooled_path = base / "pooled.json"
          metadata_path = base / "metadata.json"
          report_path = base / "report.md"
          result_count = sum(
              1
              for line in results_path.open(
                  encoding="utf-8"
              )
              if line.strip()
          )
          if result_count == 0:
              raise SystemExit(
                  "Research results innehåller inga resultat."
              )
          pooled = json.loads(
              pooled_path.read_text(
                  encoding="utf-8"
              )
          )
          if not isinstance(pooled, list):
              raise SystemExit(
                  "Pooled research results är inte en lista."
              )
          if len(pooled) == 0:
              raise SystemExit(
                  "Pooled research results innehåller inga experiment."
              )
          metadata = json.loads(
              metadata_path.read_text(
                  encoding="utf-8"
              )
          )
          if not isinstance(metadata, dict):
              raise SystemExit(
                  "Research metadata är inte ett objekt."
              )
          report = report_path.read_text(
              encoding="utf-8"
          )
          if not report.strip():
              raise SystemExit(
                  "Research report är tom."
              )
          print(
              "Research results:",
              result_count,
          )
          print(
              "Pooled experiments:",
              len(pooled),
          )
          print(
              "Report:",
              report_path,
          )
          PY
      - name: Verify diagnostic results
        run: |
          test -f data/processed/ml/research/latest/diagnostics.json
          python - <<'PY'
          import json
          from pathlib import Path
          base = Path(
              "data/processed/ml/research/latest"
          )
          manifest_path = (
              base / "diagnostics.json"
          )
          manifest = json.loads(
              manifest_path.read_text(
                  encoding="utf-8"
              )
          )
          if not isinstance(manifest, dict):
              raise SystemExit(
                  "Diagnostic manifest är inte ett objekt."
              )
          results = manifest.get("results")
          if not isinstance(results, list):
              raise SystemExit(
                  "Diagnostic manifest saknar results."
              )
          if not results:
              raise SystemExit(
                  "Inga diagnostic results hittades."
              )
          required_results = {
              "si_level_event_risk_confirmation",
              "si_event_risk_interaction",
              "si_price_dynamics",
          }
          actual_results = {
              result["experiment_id"]
              for result in results
          }
          missing_results = (
              required_results
              - actual_results
          )
          if missing_results:
              raise SystemExit(
                  "Saknar förväntade diagnostic results: "
                  + ", ".join(
                      sorted(missing_results)
                  )
              )
          for result in results:
              experiment_id = result[
                  "experiment_id"
              ]
              result_path = (
                  base
                  / "diagnostic"
                  / f"{experiment_id}.json"
              )
              if not result_path.exists():
                  raise SystemExit(
                      "Saknar diagnostic result: "
                      f"{result_path}"
                  )
              payload = json.loads(
                  result_path.read_text(
                      encoding="utf-8"
                  )
              )
              if payload.get("status") != "completed":
                  raise SystemExit(
                      "Diagnostic failed: "
                      f"{experiment_id}"
                  )
          print(
              "Diagnostic results:",
              len(results),
          )
          print(
              "Completed:",
              manifest.get(
                  "completed",
                  0,
              ),
          )
          print(
              "Failed:",
              manifest.get(
                  "failed",
                  0,
              ),
          )
          print(
              "Manifest:",
              manifest_path,
          )
          PY
      - name: Verify latest diagnostic JSON files
        run: |
          test -f data/processed/ml/research/latest/diagnostic/si_level_event_risk_confirmation.json
          test -f data/processed/ml/research/latest/diagnostic/si_event_risk_interaction.json
          test -f data/processed/ml/research/latest/diagnostic/si_price_dynamics.json
      - name: Upload research artifacts
        uses: actions/upload-artifact@v4
        with:
          name: blankdiss-research
          path: |
            data/processed/ml/research/latest/results.jsonl
            data/processed/ml/research/latest/pooled.json
            data/processed/ml/research/latest/metadata.json
            data/processed/ml/research/latest/report.md
            data/processed/ml/research/latest/diagnostics.json
            data/processed/ml/research/latest/diagnostic/*.json
            data/processed/ml/research/*/results.jsonl
            data/processed/ml/research/*/pooled.json
            data/processed/ml/research/*/metadata.json
            data/processed/ml/research/*/report.md
            data/processed/ml/research/*/diagnostics.json
            data/processed/ml/research/*/diagnostic/*.json
          if-no-files-found: error
      - name: Persist latest diagnostic results
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add \
            data/processed/ml/research/latest/diagnostics.json \
            data/processed/ml/research/latest/diagnostic/*.json \
            data/processed/analysis/features_*.jsonl \
            data/processed/analysis/features_metadata.json
          if git diff --cached --quiet; then
            echo "Inga forsknings- eller featureresultat ändrades."
          else
            git commit -m "Update research results"
            git push
          fi

En viktig detalj

Jag lade medvetet till featurefilerna i sista git add.

Det behövs eftersom om vi får nya priser/FI-data:

fingerprint ändrat
→ Build features
→ nya features
→ nya features_metadata.json
→ dessa måste sparas

Nästa körning läser sedan det nya fingerprintet och kan konstatera:

Previous == Current
→ Build features: SKIP

Så vi får exakt beteendet vi ville ha:

Vanlig ny researchkörning:

fetch → fingerprint oförändrat → ingen feature build → research

Ny data:

fetch → fingerprint ändrat → bygg features → QC → research → spara nya features

Och gamla features_metadata.json saknar fingerprint, så första körningen efter denna ändring bygger om features en gång. Därefter är systemet självgående.
