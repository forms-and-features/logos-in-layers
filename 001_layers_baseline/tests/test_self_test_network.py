#!/usr/bin/env python3
"""
Network-dependent self-test runner for --self-test flag.

Requirements:
- Network access + Hugging Face auth for gated models (if applicable)
- Intended for manual runs only; excluded from CPU-only runner

Usage:
  RUN_NETWORK_TESTS=1 venv/bin/python 001_layers_baseline/tests/test_self_test_network.py
"""

import _pathfix  # noqa: F401

import os
import subprocess
import sys


def main():
    if os.environ.get("RUN_NETWORK_TESTS") != "1":
        print("Skipping network-dependent self-test (set RUN_NETWORK_TESTS=1 to run)")
        return 0

    try:
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "..", "run.py"),
            "mistralai/Mistral-7B-v0.1",
            "--self-test",
            "--device",
            "cpu",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print("STDOUT:\n" + result.stdout)
        if result.stderr:
            print("STDERR:\n" + result.stderr)

        if result.returncode == 0 and ("KL SANITY TEST" in result.stdout and "PASS" in result.stdout):
            print("✅ --self-test flag works correctly")
            return 0
        else:
            print(f"❌ --self-test failed (rc={result.returncode})")
            return 1
    except subprocess.TimeoutExpired:
        print("❌ --self-test timed out")
        return 1
    except Exception as e:
        print(f"❌ --self-test failed with exception: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

