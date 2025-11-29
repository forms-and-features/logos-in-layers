#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON_BIN:-"$ROOT_DIR/venv/bin/python"}

cd "$ROOT_DIR"

echo "Running CPU-only tests with $PY"

# Preflight: ensure selected Python exists and can import torch
if [ ! -x "$PY" ]; then
  echo "Error: Python interpreter not executable at: $PY"
  echo "Hint: activate the venv and re-run: source venv/bin/activate && scripts/run_cpu_tests.sh"
  exit 1
fi

echo "Preflight: verifying torch import …"
if ! "$PY" - <<'PY'
try:
    import torch
    print("torch ok:", getattr(torch, "__version__", "unknown"))
except Exception as e:
    print("torch import failed:", e)
    raise SystemExit(1)
PY
then
  echo "Error: torch is not importable with $PY"
  echo "Hint: ensure the project venv is active: source venv/bin/activate && scripts/run_cpu_tests.sh"
  exit 1
fi

echo "Running pytest over 001_layers_baseline/tests …"
"$PY" -m pytest -q 001_layers_baseline/tests

echo "All CPU-only tests completed (pytest)."
