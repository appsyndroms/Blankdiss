#!/usr/bin/env bash
set -e

ROOT="$(git rev-parse --show-toplevel)"

cd "$ROOT"

FORCE=false

if [[ "${1:-}" == "--force" ]]; then
    FORCE=true
fi

if [[ "${FORCE_SYNC:-false}" == "true" ]]; then
    FORCE=true
fi

echo "=========================================="
echo "BUILD FEATURES"
echo "=========================================="

if [ "$FORCE" = true ]; then
    echo
    echo "FORCE FEATURE BUILD AKTIVERAD."
    echo "Bygger om feature-datasetet."
fi

echo
echo "Bygger feature-dataset..."

if [ "$FORCE" = true ]; then
    python -u -m analysis.build_features --force
else
    python -u -m analysis.build_features
fi

echo
echo "=========================================="
echo "VERIFY FEATURE DATASET"
echo "=========================================="

set -e

test -f data/processed/analysis/features_metadata.json
test -f data/processed/analysis/fi_price_features.jsonl
test -n "$(find data/processed/analysis -maxdepth 1 -name 'features_*.jsonl' -print -quit)"

echo "=========================================="
echo "FEATURE DATASET FINNS"
echo "=========================================="

echo
echo "Feature chunks:"
ls -lh data/processed/analysis/features_*.jsonl

echo
echo "Legacy feature view:"
ls -lh data/processed/analysis/fi_price_features.jsonl

echo
echo "Metadata:"
cat data/processed/analysis/features_metadata.json

echo
echo "=========================================="
echo "KONTROLLERA FEATURE-STORLEK"
echo "=========================================="

python - <<'PY'
from pathlib import Path

paths = sorted(
    Path("data/processed/analysis").glob("features_*.jsonl")
)

if not paths:
    raise SystemExit("Inga feature-chunks hittades.")

limit = 20 * 1024 * 1024

for path in paths:
    size = path.stat().st_size
    mb = size / 1024 / 1024

    print(f"{path.name}: {mb:.2f} MB")

    if size >= limit:
        raise SystemExit(
            f"{path.name} är större än eller lika med 20 MB."
        )
PY

echo
echo "=========================================="
echo "RUN FEATURE QC"
echo "=========================================="

python -u -m analysis.features_qc

echo
echo "=========================================="
echo "KONTROLLERA FEATURE QC-STATUS"
echo "=========================================="

test -f data/processed/analysis/features_qc.json

python - <<'PY'
import json
from pathlib import Path

path = Path(
    "data/processed/analysis/features_qc.json"
)

data = json.loads(
    path.read_text(encoding="utf-8")
)

status = data.get("status")

print(f"Feature QC status: {status}")

if status == "FAIL":
    raise SystemExit(
        "Feature QC failed with status: FAIL"
    )

if status == "WARN":
    print(
        "Feature QC har status WARN. "
        "Varningar tillåts och workflow fortsätter."
    )

elif status == "PASS":
    print(
        "Feature QC har status PASS."
    )

else:
    raise SystemExit(
        f"Feature QC returnerade okänd status: {status}"
    )
PY

echo
echo "=========================================="
echo "FÖRBERED FEATURE-COMMIT"
echo "=========================================="

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

git checkout -- \
  data/processed/analysis/fi_price_features_metadata.json

rm -f \
  data/processed/analysis/fi_price_features.jsonl

git add \
  data/processed/analysis/features_*.jsonl \
  data/processed/analysis/features_metadata.json \
  data/processed/analysis/features_qc.json

git reset -- \
  data/processed/analysis/fi_price_features.jsonl \
  data/processed/analysis/fi_price_features_metadata.json \
  2>/dev/null || true

echo
echo "=========================================="
echo "WORKING TREE FÖRE COMMIT"
echo "=========================================="

git status --short

if git diff --cached --quiet; then
    echo
    echo "Inga feature-förändringar att committa."
    exit 0
fi

echo
echo "Filer som kommer att committas:"
git diff --cached --name-status

echo
echo "Feature-förändringar:"
git diff --cached --stat -- \
  data/processed/analysis/features_*.jsonl \
  data/processed/analysis/features_metadata.json \
  data/processed/analysis/features_qc.json

git commit \
  -m "Update feature dataset"

echo
echo "=========================================="
echo "FEATURE COMMITTAD"
echo "=========================================="

git status --short

echo
echo "Pussar feature-dataset..."
bash scripts/git_push_with_retry.sh

echo
echo "=========================================="
echo "FEATURE BUILD KLAR"
echo "=========================================="

git status --short
