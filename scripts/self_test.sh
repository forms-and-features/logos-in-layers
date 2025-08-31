#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to run the KL self-test without remembering env vars.
# Usage: scripts/self_test.sh [MODEL_ID] [DEVICE]
# Defaults: MODEL_ID=mistralai/Mistral-7B-v0.1, DEVICE=cpu

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="${ROOT_DIR}/venv/bin/python"

MODEL_ID="${1:-mistralai/Mistral-7B-v0.1}"
DEVICE="${2:-cpu}"

echo "Running KL self-test for ${MODEL_ID} on ${DEVICE}"
"$PY" "${ROOT_DIR}/001_layers_baseline/run.py" "${MODEL_ID}" --self-test --device "${DEVICE}"

