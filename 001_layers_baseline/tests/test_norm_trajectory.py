#!/usr/bin/env python3
"""Tests for residual norm trajectory classification."""

import _pathfix  # noqa: F401

from layers_core.summaries import classify_norm_trajectory


def test_classify_norm_trajectory_monotonic():
    entries = [
        {"layer": 0, "raw_resid_norm": 0.5, "resid_norm_ratio": 2.0, "delta_resid_cos": 0.95},
        {"layer": 1, "raw_resid_norm": 0.7, "resid_norm_ratio": 1.43, "delta_resid_cos": 0.94},
        {"layer": 2, "raw_resid_norm": 1.1, "resid_norm_ratio": 0.91, "delta_resid_cos": 0.93},
    ]
    result = classify_norm_trajectory(entries)
    assert result is not None
    assert result["shape"] in {"monotonic", "plateau"}
    assert result["n_spikes"] == 0
    assert result["sampled_layers"] == 3


def test_classify_norm_trajectory_plateau():
    entries = [
        {"layer": i, "raw_resid_norm": 1.0, "resid_norm_ratio": 1.0, "delta_resid_cos": 0.99}
        for i in range(5)
    ]
    result = classify_norm_trajectory(entries)
    assert result is not None
    assert result["shape"] == "plateau"
    assert result["n_spikes"] == 0


def test_classify_norm_trajectory_spike_detection():
    entries = [
        {"layer": 0, "raw_resid_norm": 1.0, "resid_norm_ratio": 1.0, "delta_resid_cos": 0.99},
        {"layer": 1, "raw_resid_norm": 1.2, "resid_norm_ratio": 1.2, "delta_resid_cos": 0.99},
        {"layer": 2, "raw_resid_norm": 2.0, "resid_norm_ratio": 3.5, "delta_resid_cos": 0.99},
    ]
    result = classify_norm_trajectory(entries, sem_layer=2)
    assert result is not None
    assert result["shape"] == "spike"
    assert result["n_spikes"] >= 1


def test_classify_norm_trajectory_handles_missing_norms():
    entries = [
        {"layer": 0, "resid_norm_ratio": 2.0, "delta_resid_cos": 0.95},
        {"layer": 1, "resid_norm_ratio": 1.0, "delta_resid_cos": 0.95},
    ]
    result = classify_norm_trajectory(entries)
    assert result is not None
    assert result["sampled_layers"] == 2
