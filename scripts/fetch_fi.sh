#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"

cd "$ROOT"

echo "=========================================="
echo "FETCH FI"
echo "=========================================="

echo
echo "Hämtar aktuell FI-data och återställer"
echo "eventuella historiska luckor..."

python -u -m fi

echo
echo "Uppdaterar FI historiska positioner..."

python -u -m fi.historical_positions

echo
echo "Rekonstruerar FI aggregate..."

python -u -m fi.reconstruct_aggregate

echo
echo "=========================================="
echo "FÖRBERED FI-COMMIT"
echo "=========================================="

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

git add \
  data/raw/fi/aggregate/manifest.json \
  data/raw/fi/aggregate/snapshots

echo
echo "FI-data före commit:"
git status --short

if git diff --cached --quiet; then
    echo
    echo "Inga FI-snapshotförändringar att committa."
    exit 0
fi

echo
echo "Filer som kommer att committas:"
git diff --cached --name-status

git commit \
  -m "Update FI aggregate snapshots"

echo
echo "=========================================="
echo "FI COMMITTAD"
echo "=========================================="

git status --short

echo
echo "Pussar FI-data..."

bash scripts/git_push_with_retry.sh

echo
echo "=========================================="
echo "FI KLAR"
echo "=========================================="

git status --short
