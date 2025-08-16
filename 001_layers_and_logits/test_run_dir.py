#!/usr/bin/env python3
"""Tests for run directory rotation helper (CPU-only, filesystem only)."""

import os
import tempfile
from datetime import datetime

from layers_core.run_dir import setup_run_latest_directory


def fixed_time(ts: str):
    dt = datetime.strptime(ts, "%Y%m%d-%H%M")
    return lambda: dt


def test_initial_creation_and_timestamp():
    with tempfile.TemporaryDirectory() as td:
        run_dir = setup_run_latest_directory(td, now_fn=fixed_time("20250101-1200"))
        assert os.path.basename(run_dir) == "run-latest"
        ts_file = os.path.join(run_dir, "timestamp-20250101-1200")
        assert os.path.exists(ts_file)


def test_rotation_with_existing_timestamp():
    with tempfile.TemporaryDirectory() as td:
        # First create a run-latest with a timestamp
        run_dir = setup_run_latest_directory(td, now_fn=fixed_time("20250101-1200"))
        # Next call should rotate to run-<oldts> and create a new run-latest
        run_dir2 = setup_run_latest_directory(td, now_fn=fixed_time("20250102-1300"))
        rotated_dir = os.path.join(td, "run-20250101-1200")
        assert os.path.exists(rotated_dir)
        assert os.path.exists(os.path.join(run_dir2, "timestamp-20250102-1300"))


def test_rotation_without_timestamp_file():
    with tempfile.TemporaryDirectory() as td:
        # Manually create a run-latest without timestamp
        rl = os.path.join(td, "run-latest")
        os.makedirs(rl, exist_ok=True)
        # Next call should rotate using current ts with -rotated suffix
        run_dir = setup_run_latest_directory(td, now_fn=fixed_time("20250103-1400"))
        rotated_dir = os.path.join(td, "run-20250103-1400-rotated")
        assert os.path.exists(rotated_dir)
        assert os.path.exists(os.path.join(run_dir, "timestamp-20250103-1400"))

