#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON_BIN:-"$ROOT_DIR/venv/bin/python"}

cd "$ROOT_DIR"

echo "Running CPU-only tests with $PY"

${PY} 001_layers_and_logits/tests/test_norm_utils.py
${PY} 001_layers_and_logits/tests/test_numerics.py
${PY} 001_layers_and_logits/tests/test_csv_io.py
${PY} 001_layers_and_logits/tests/test_collapse_rules.py
${PY} 001_layers_and_logits/tests/test_device_policy.py
${PY} 001_layers_and_logits/tests/test_hooks.py
${PY} 001_layers_and_logits/tests/test_run_dir.py

(
  cd 001_layers_and_logits/tests
  "$PY" test_refactored_self_test.py
)

# Normalization test prints status; still CPU-only
${PY} 001_layers_and_logits/tests/test_normalization.py

echo "All CPU-only tests completed."
