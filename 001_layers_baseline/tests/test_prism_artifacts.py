#!/usr/bin/env python3
"""Unit tests for Prism artifact save/load round-trip.
"""

import _pathfix  # noqa: F401

import os
import tempfile
import torch

from layers_core.prism import WhitenStats, save_prism_artifacts, load_prism_artifacts


def test_artifact_round_trip():
    d = 8
    stats = WhitenStats(mean=torch.zeros(d), var=torch.ones(d)*2.0, eps=1e-8)
    Q = torch.eye(d, dtype=torch.float32)
    prov = {"method": "procrustes", "k": 4, "layers": ["embed", 0, 1], "seed": 316}
    with tempfile.TemporaryDirectory() as td:
        w_path, q_path, p_path = save_prism_artifacts(td, stats=stats, Q=Q, provenance=prov)
        assert os.path.exists(w_path)
        assert os.path.exists(q_path)
        assert os.path.exists(p_path)
        stats2, Q2, prov2 = load_prism_artifacts(td)
        assert Q2.shape == Q.shape
        assert torch.allclose(Q2, Q)
        assert stats2.mean.shape == stats.mean.shape
        assert torch.allclose(stats2.mean, stats.mean)
        assert torch.allclose(stats2.var, stats.var)
        assert prov2.get("k") == prov.get("k")

