#!/usr/bin/env python3
"""Unit tests for last-layer consistency helper.

CPU-only; no model/HF required. Uses deterministic tensors.
"""

import _pathfix  # noqa: F401

import math
import torch

from layers_core.consistency import compute_last_layer_consistency


def test_temp_estimation_and_top1_agreement():
    torch.manual_seed(0)
    # Create a base logit vector z and define final_probs = softmax(z)
    z = torch.randn(32, dtype=torch.float32)
    final_probs = torch.softmax(z, dim=0)
    final_top1_id = int(torch.argmax(final_probs).item())

    # Last-layer logits are a rescaled version of z by factor 2
    last_logits = 2.0 * z

    out = compute_last_layer_consistency(
        last_logits=last_logits,
        final_probs=final_probs,
        final_top1_id=final_top1_id,
        first_ans_id=None,
        head_scale_cfg=2.0,
        head_softcap_cfg=1.5,
        topk_cum=5,
    )

    # top1 should match
    assert out["top1_agree"] is True
    # temp_est should be close to the true scale (2.0) within tolerance
    assert out["temp_est"] is not None
    assert math.isfinite(out["temp_est"]) and abs(out["temp_est"] - 2.0) < 0.5

    # sanity checks on keys present and finite
    for k in ("p_top1_lens", "p_top1_model", "kl_to_final_bits"):
        assert k in out and out[k] is not None

    # transforms block present
    kt = out.get("kl_after_transform_bits", {})
    assert set(kt.keys()) == {"scale", "softcap", "scale_then_softcap"}


def test_handles_none_transforms():
    torch.manual_seed(0)
    z = torch.randn(16, dtype=torch.float32)
    final_probs = torch.softmax(z, dim=0)
    final_top1_id = int(torch.argmax(final_probs).item())
    last_logits = z

    out = compute_last_layer_consistency(
        last_logits=last_logits,
        final_probs=final_probs,
        final_top1_id=final_top1_id,
        first_ans_id=None,
        head_scale_cfg=None,
        head_softcap_cfg=None,
        topk_cum=5,
    )

    kt = out.get("kl_after_transform_bits", {})
    assert kt.get("scale") is None and kt.get("softcap") is None and kt.get("scale_then_softcap") is None


def test_warn_gate_triggers_with_threshold():
    torch.manual_seed(0)
    base_logits = torch.tensor([1.5, 0.2, -0.7, -1.0], dtype=torch.float32)
    final_probs = torch.softmax(base_logits, dim=0)
    final_top1_id = int(torch.argmax(final_probs).item())

    # Scale logits to induce calibration mismatch relative to final head
    last_logits = 3.0 * base_logits

    out = compute_last_layer_consistency(
        last_logits=last_logits,
        final_probs=final_probs,
        final_top1_id=final_top1_id,
        first_ans_id=None,
        head_scale_cfg=None,
        head_softcap_cfg=None,
        topk_cum=5,
    )

    assert out["kl_to_final_bits"] is not None
    assert out["kl_to_final_bits"] >= 0.25
    assert out["warn_high_last_layer_kl"] is True
    gates = out.get("gates")
    assert isinstance(gates, dict)
    assert gates.get("warn_high_last_layer_kl") is True
    assert math.isclose(gates.get("threshold_bits"), 0.25, rel_tol=1e-6, abs_tol=1e-6)
    assert "delta_kl_temp_bits" in gates
    assert gates["delta_kl_temp_bits"] >= 0.0


def test_warn_gate_cleared_when_below_threshold():
    torch.manual_seed(1)
    base_logits = torch.tensor([0.8, 0.2, -0.4, -0.6], dtype=torch.float32)
    final_probs = torch.softmax(base_logits, dim=0)
    final_top1_id = int(torch.argmax(final_probs).item())

    # Identical logits should be perfectly calibrated
    last_logits = base_logits.clone()

    out = compute_last_layer_consistency(
        last_logits=last_logits,
        final_probs=final_probs,
        final_top1_id=final_top1_id,
        first_ans_id=None,
        head_scale_cfg=None,
        head_softcap_cfg=None,
        topk_cum=5,
    )

    assert out["kl_to_final_bits"] is not None
    assert out["kl_to_final_bits"] < 0.25
    assert out["warn_high_last_layer_kl"] is False
    gates = out.get("gates")
    assert isinstance(gates, dict)
    assert gates.get("warn_high_last_layer_kl") is False
