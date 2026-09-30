#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"

cd "$ROOT"

echo "=========================================="
echo "FETCH PRICES"
echo "=========================================="

echo
echo "Hämtar Yahoo-priser från 2022-01-01..."

python -u -m prices --start 2022-01-01

echo
echo "=========================================="
echo "PRICES KLAR"
echo "=========================================="
