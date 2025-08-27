#!/usr/bin/env bash
set -euo pipefail

# Canonical entrypoint for running the CPU-only test suite with the project venv.
# Usage: scripts/test.sh

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

if [ ! -d "venv" ]; then
  echo "Error: venv not found at $ROOT_DIR/venv"
  echo "Create it and install deps, then retry:\n  python -m venv venv && source venv/bin/activate && pip install -r requirements-cpu.txt"
  exit 1
fi

echo "Activating venv and running CPU-only tests …"
source venv/bin/activate
scripts/run_cpu_tests.sh

