#!/usr/bin/env python3

import _pathfix  # noqa: F401

import torch
import pytest

from layers_core.tuned_audit import build_tuned_audit_summary, build_provenance_snapshot


def _make_probs(logits):
    tensor = torch.tensor(logits, dtype=torch.float32)
    return torch.softmax(tensor, dim=0)


def test_build_tuned_audit_summary_calibration_only():
    variant_rows = [
        {
            "layer": 1,
            "delta_kl_bits_rot_only": 0.10,
            "delta_kl_bits_temp_only": 1.10,
            "delta_kl_bits_tuned": 1.30,
            "delta_kl_bits_interaction": 0.0,
        },
        {
            "layer": 8,
            "delta_kl_bits_rot_only": 0.15,
            "delta_kl_bits_temp_only": 1.05,
            "delta_kl_bits_tuned": 1.20,
            "delta_kl_bits_interaction": -0.05,
        },
        {
            "layer": 16,
            "delta_kl_bits_rot_only": 0.20,
            "delta_kl_bits_temp_only": 1.00,
            "delta_kl_bits_tuned": 1.25,
            "delta_kl_bits_interaction": 0.05,
        },
    ]
    head_payload = {
        "kl_bits_tuned_final": 0.8,
        "tuned_logits_last": torch.tensor([2.0, 1.0, -1.0]),
        "final_probs": _make_probs([3.0, 1.0, 0.0]),
    }
    audit_data = {
        "variant_rows": variant_rows,
        "positional_rows": [],
        "pos_grid": [],
        "head_mismatch": head_payload,
    }

    summary = build_tuned_audit_summary(audit_data)

    rot_summary = summary["rotation_vs_temperature"]
    assert pytest.approx(rot_summary["delta_kl_rot_p50"], rel=1e-3) == 0.15
    assert summary["tuned_is_calibration_only"] is True
    assert summary["preferred_semantics_lens_hint"] == "tuned_for_calibration_only"

    head = summary["head_mismatch"]
    assert head["kl_bits_tuned_final"] == pytest.approx(0.8)
    assert head["tau_star_modelcal"] is not None
    assert head["kl_bits_tuned_final_after_tau_star"] <= head["kl_bits_tuned_final"]


def test_build_provenance_snapshot_extracts_fields():
    provenance = {
        "translator": {
            "rank": 128,
            "has_preconditioner": True,
            "preconditioner": {"rotation": True},
            "final_identity": True,
            "temperatures": [0.6, 0.7, 0.8],
        },
        "training": {
            "tokens_per_step": 2048,
            "total_steps": 50,
            "layers_sampled_per_step": 12,
            "optimizer": {"type": "Adam", "lr": 1e-3},
            "schedule": "cosine",
            "dataset": {
                "repo_id": "example/corpus",
                "revision": "v1",
                "content_hash": "abc123",
                "position_fraction_range": [0.6, 0.95],
                "positions_per_seq": 8,
            },
        },
    }

    snapshot = build_provenance_snapshot(provenance)
    assert snapshot["dataset_id"] == "example/corpus"
    assert snapshot["dataset_revision"] == "v1"
    assert snapshot["content_hash"] == "abc123"
    assert snapshot["train_pos_window"] == [0.6, 0.95]
    assert snapshot["rank"] == 128
    assert snapshot["preconditioner"] == {"whiten": True, "orthogonal_rotation": True}
    assert snapshot["temperatures_stats"]["min"] == pytest.approx(0.6)
    assert snapshot["fit_total_tokens_est"] == pytest.approx(2048 * 50)
