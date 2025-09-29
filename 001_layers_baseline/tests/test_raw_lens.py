#!/usr/bin/env python3
"""Unit tests for raw-vs-norm dual-lens helpers (CPU-only)."""

import _pathfix  # noqa: F401

import os
import torch

from layers_core.raw_lens import (
    get_raw_lens_mode,
    should_sample_layer,
    init_raw_lens_check,
    record_dual_lens_sample,
    summarize_raw_lens_check,
    compute_windowed_raw_norm,
)


def test_get_raw_lens_mode_defaults():
    # Default when unset → sample
    os.environ.pop("LOGOS_RAW_LENS", None)
    assert get_raw_lens_mode(self_test=False) == "sample"
    # Respect off/sample/full
    os.environ["LOGOS_RAW_LENS"] = "off"
    assert get_raw_lens_mode(self_test=False) == "off"
    os.environ["LOGOS_RAW_LENS"] = "SAMPLE"
    assert get_raw_lens_mode(self_test=False) == "sample"
    os.environ["LOGOS_RAW_LENS"] = "full"
    assert get_raw_lens_mode(self_test=False) == "full"
    # In self-test, upgrade sample→full
    os.environ["LOGOS_RAW_LENS"] = "sample"
    assert get_raw_lens_mode(self_test=True) == "full"


def test_should_sample_layer_policy():
    n = 12
    # off
    assert not should_sample_layer("off", n, 1)
    assert not should_sample_layer("off", n, 6)
    # sample → 25%, 50%, 75% → 3,6,9
    picks = [i for i in range(1, n + 1) if should_sample_layer("sample", n, i)]
    assert picks == [3, 6, 9]
    # full → all layers
    picks_full = [i for i in range(1, n + 1) if should_sample_layer("full", n, i)]
    assert picks_full == list(range(1, n + 1))


class _FakeTok:
    def decode(self, ids):
        # Only used in fallback path; with first_ans_id provided we won't hit this.
        if isinstance(ids, (list, tuple)):
            return f"tok{ids[0]}"
        return f"tok{ids}"


def test_record_and_summarize_dual_lens():
    # Small vocab
    d_vocab = 4
    # Use identity unembedding for simplicity
    W_U = torch.eye(d_vocab, dtype=torch.float32)
    b_U = None
    force_fp32 = True
    tok = _FakeTok()
    # Final reference distribution (uniform)
    final_probs = torch.full((d_vocab,), 1.0 / d_vocab)
    # Ground truth first-token id = 2
    first_ans_id = 2

    # Normalized path peaks on id=2; raw path peaks on id=1
    last_logits_norm = torch.tensor([0.0, 0.0, 5.0, 0.0], dtype=torch.float32)
    resid_raw_last_vec = torch.tensor([0.0, 5.0, 0.0, 0.0], dtype=torch.float32)

    out = init_raw_lens_check("sample")
    record_dual_lens_sample(
        out,
        layer_out_idx=7,
        last_logits_norm=last_logits_norm,
        resid_raw_last_vec=resid_raw_last_vec,
        W_U=W_U,
        b_U=b_U,
        force_fp32_unembed=force_fp32,
        tokenizer=tok,
        final_probs=final_probs,
        first_ans_id=first_ans_id,
        ground_truth="Berlin",
    )
    assert len(out["samples"]) == 1
    s = out["samples"][0]
    # top1 must disagree
    assert s["top1_agree"] is False
    # ranks should reflect answer at id=2: norm is rank 1, raw is not
    assert s["answer_rank_norm"] == 1
    assert s["answer_rank_raw"] != 1
    # summarize
    summ = summarize_raw_lens_check(out["samples"])
    assert summ["first_norm_only_semantic_layer"] == 7
    # KL should be noticeable; risk at least medium
    assert summ["lens_artifact_risk"] in ("medium", "high")


def test_compute_windowed_raw_norm_sets_flags():
    W_U = torch.eye(3, dtype=torch.float32)
    b_U = None
    force_fp32 = True

    norm_logits_map = {1: torch.tensor([0.0, 4.0, 0.0], dtype=torch.float32)}
    raw_resid_map = {1: torch.tensor([3.0, 0.2, 0.1], dtype=torch.float32)}
    collected_map = {1: {"is_answer": True}}
    final_probs = torch.full((3,), 1.0 / 3.0, dtype=torch.float32)

    summary, records = compute_windowed_raw_norm(
        radius=2,
        center_layers=[1],
        norm_logits_map=norm_logits_map,
        raw_resid_map=raw_resid_map,
        collected_map=collected_map,
        final_probs=final_probs,
        W_U=W_U,
        b_U=b_U,
        force_fp32_unembed=force_fp32,
        decode_id_fn=lambda idx: f"tok{int(idx)}",
        ctx_ids_list=[0, 1, 2],
        first_ans_token_id=1,
        ground_truth="tok1",
        prompt_id="pos",
        prompt_variant="orig",
        n_layers=2,
    )

    assert summary["layers_checked"] == [1]
    assert summary["norm_only_semantics_layers"] == [1]
    assert summary["mode"] == "window"
    assert len(records) == 2  # norm + raw rows
    lenses = {rec["lens"] for rec in records}
    assert lenses == {"norm", "raw"}
    # Each record should carry strict-sweep flags
    for rec in records:
        for lab in ("copy_strict@0.7", "copy_strict@0.8", "copy_strict@0.9", "copy_strict@0.95"):
            assert lab in rec
