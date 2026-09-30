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
echo "FI KLAR"
echo "=========================================="
