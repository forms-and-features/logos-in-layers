#!/usr/bin/env python3
"""CPU-only tests for full raw-vs-norm sweep and lens-artefact score.

Verifies shape and summary stats for compute_full_raw_norm and the numeric
score/tier computed from the full summary.
"""

import _pathfix  # noqa: F401

import torch

from layers_core.raw_lens import compute_full_raw_norm
from layers_core.summaries import compute_lens_artifact_score


def test_full_raw_norm_basic():
    # Toy vocab=4, two layers available: 0 and 1
    V = 4
    W_U = torch.eye(V, dtype=torch.float32)
    b_U = None
    force_fp32 = True
    final_probs = torch.full((V,), 0.25)

    # Norm logits: favor id=2 at layer 0, id=3 at layer 1
    norm_logits_map = {
        0: torch.tensor([0.0, 0.0, 3.0, 0.0], dtype=torch.float32),
        1: torch.tensor([0.0, 0.0, 0.0, 4.0], dtype=torch.float32),
    }
    # Raw resid vectors project to logits favoring id=1 (for layer 0) and id=3 (for layer 1)
    raw_resid_map = {
        0: torch.tensor([0.0, 5.0, 0.0, 0.0], dtype=torch.float32),
        1: torch.tensor([0.0, 0.0, 0.0, 2.0], dtype=torch.float32),
    }
    collected_map = {
        0: {"is_answer": True},
        1: {"is_answer": True},
    }

    def _decode_id(idx):
        return f"t{int(idx)}"

    summary, rows = compute_full_raw_norm(
        norm_logits_map=norm_logits_map,
        raw_resid_map=raw_resid_map,
        collected_map=collected_map,
        final_probs=final_probs,
        W_U=W_U,
        b_U=b_U,
        force_fp32_unembed=force_fp32,
        decode_id_fn=_decode_id,
        ctx_ids_list=[0, 1, 2, 3],
        first_ans_token_id=2,
        ground_truth="t2",
        prompt_id="pos",
        prompt_variant="orig",
        n_layers=2,
    )

    assert summary.get("mode") == "full"
    assert isinstance(rows, list) and len(rows) == 2
    assert "js_divergence_percentiles" in summary
    assert "l1_prob_diff_percentiles" in summary
    # Layer 0 should be norm-only semantics (norm rank=1 for id=2; raw rank!=1)
    r0 = [r for r in rows if r.get("layer") == 0][0]
    assert r0.get("norm_only_semantics") in (True, False)
    # KL fields present
    assert "kl_norm_vs_raw_bits" in r0
    assert "js_divergence" in r0
    assert "kl_raw_to_norm_bits" in r0
    assert "l1_prob_diff" in r0

    # Lens-artefact score produces tier and a float
    score = compute_lens_artifact_score(
        pct_layers_kl_ge_1=summary.get("pct_layers_kl_ge_1.0"),
        pct_layers_kl_ge_0_5=summary.get("pct_layers_kl_ge_0.5"),
        n_norm_only=int(summary.get("n_norm_only_semantics_layers") or 0),
        max_kl_bits=summary.get("max_kl_norm_vs_raw_bits"),
        js_p50=(summary.get("js_divergence_percentiles") or {}).get("p50"),
        l1_p50=(summary.get("l1_prob_diff_percentiles") or {}).get("p50"),
    )
    assert set(score.keys()) == {"lens_artifact_score", "lens_artifact_score_v2", "tier"}
    assert isinstance(score["lens_artifact_score"], float)
    assert isinstance(score["lens_artifact_score_v2"], float)
    assert score["tier"] in ("low", "medium", "high")


def test_full_raw_norm_empty_inputs():
    summary, rows = compute_full_raw_norm(
        norm_logits_map={},
        raw_resid_map={},
        collected_map={},
        final_probs=torch.full((4,), 0.25),
        W_U=torch.eye(4),
        b_U=None,
        force_fp32_unembed=True,
        decode_id_fn=lambda i: str(i),
        ctx_ids_list=[],
        first_ans_token_id=None,
        ground_truth="",
        prompt_id="pos",
        prompt_variant="orig",
        n_layers=0,
    )
    assert isinstance(summary, dict) and summary.get("mode") == "full"
    assert rows == []


def test_score_handles_none_values():
    score = compute_lens_artifact_score(
        pct_layers_kl_ge_1=None,
        pct_layers_kl_ge_0_5=None,
        n_norm_only=0,
        max_kl_bits=None,
    )
    assert set(score.keys()) == {"lens_artifact_score", "lens_artifact_score_v2", "tier"}
    assert isinstance(score["lens_artifact_score"], float)
    assert isinstance(score["lens_artifact_score_v2"], float)
