#!/usr/bin/env python3
"""Equivalence checks for prism sidecar helpers vs prior inline logic.

CPU-only, no model/HF. Uses deterministic tensors to compare outputs.
"""

import _pathfix  # noqa: F401

import torch

from layers_core.prism_sidecar import (
    append_prism_record,
    append_prism_pure_next_token,
)
from layers_core.numerics import bits_entropy_from_logits
from layers_core.metrics import compute_next_token_metrics
from layers_core.collapse_rules import (
    detect_copy_collapse_id_subseq,
    is_pure_whitespace_or_punct,
)
from layers_core.windows import WindowManager


def _decode_id(idx: int) -> str:
    return f"tok{int(idx)}"


def test_prism_record_helper_matches_manual_block():
    torch.manual_seed(0)
    buf = {"records": []}
    layer = 3
    pos = 2
    k = 4
    token_str = "tokX"

    # Deterministic logits for one position (vocab=8)
    logits_pos = torch.randn(8, dtype=torch.float32)

    # Manual inline computation (previous run.py behavior)
    probs = torch.softmax(logits_pos, dim=0)
    pent = bits_entropy_from_logits(logits_pos)
    _, idx = torch.topk(logits_pos, k, largest=True, sorted=True)
    top_probs = probs[idx]
    top_tokens = [_decode_id(i) for i in idx]
    manual = {
        "type": "record",
        "prompt_id": "pos",
        "prompt_variant": "orig",
        "layer": layer,
        "pos": pos,
        "token": token_str,
        "entropy": pent,
        "topk": [[tok, float(p.item())] for tok, p in zip(top_tokens, top_probs)],
    }

    # Helper
    append_prism_record(
        buf,
        prompt_id="pos",
        prompt_variant="orig",
        layer=layer,
        pos=pos,
        token=token_str,
        logits_pos=logits_pos,
        decode_id_fn=_decode_id,
        top_k=k,
    )

    assert buf["records"], "helper did not append a record"
    got = buf["records"][-1]
    if got != manual:
        print("[DEBUG prism-equivalence] record mismatch")
        print(" got:", got)
        print(" exp:", manual)
    assert got == manual


def test_prism_pure_next_token_helper_fields_match_manual_logic():
    torch.manual_seed(0)
    buf = {"pure_next_token_records": []}

    # Setup synthetic tokens/logits (seq_len=5, vocab=9)
    S, V = 5, 9
    logits_all = torch.randn(S, V, dtype=torch.float32)
    tokens_tensor = torch.arange(S)[None, :]
    ctx_ids_list = list(range(S - 1))  # pretend context ids are 0..S-2
    wm = WindowManager(window_k=1)
    final_logits = torch.randn(V, dtype=torch.float32)
    final_probs = torch.softmax(final_logits, dim=0)
    final_dir = final_logits / (torch.norm(final_logits) + 1e-12)
    first_ans_id = 3
    ground_truth = "unused"
    k = 4

    # Manual inline logic for last position
    last_pos = S - 1
    pz = logits_all[last_pos]
    pprobs = torch.softmax(pz, dim=0)
    pent = bits_entropy_from_logits(pz)
    _, p_top_idx = torch.topk(pz, k, largest=True, sorted=True)
    p_top_probs = pprobs[p_top_idx]
    p_top_tokens = [_decode_id(i) for i in p_top_idx]
    p_top1_id = int(p_top_idx[0].item())
    prism_window_ids = wm.append_and_trim("prism", "pos", "orig", int(p_top1_id))
    p_copy = detect_copy_collapse_id_subseq(
        pz, ctx_ids_list, prism_window_ids, copy_threshold=0.95, copy_margin=0.10
    )
    if p_copy and is_pure_whitespace_or_punct(p_top_tokens[0]):
        p_copy = False
    p_metrics = compute_next_token_metrics(pprobs, p_top1_id, final_probs, first_ans_id, topk_cum=5)
    p_is_answer = (
        (p_metrics.get("answer_rank") == 1)
        if p_metrics.get("answer_rank") is not None
        else False
    )
    _pn = torch.norm(pz) + 1e-12
    p_cos = torch.dot((pz / _pn), final_dir).item()
    manual = {
        "type": "pure_next_token_record",
        "prompt_id": "pos",
        "prompt_variant": "orig",
        "layer": 7,  # arbitrary layer id passed below
        "pos": last_pos,
        "token": "⟨NEXT⟩",
        "entropy": pent,
        "topk": [[tok, float(prob.item())] for tok, prob in zip(p_top_tokens, p_top_probs)],
        "copy_collapse": p_copy,
        "entropy_collapse": pent <= 1.0,
        "is_answer": p_is_answer,
        "p_top1": p_metrics.get("p_top1"),
        "p_top5": p_metrics.get("p_top5"),
        "p_answer": p_metrics.get("p_answer"),
        "kl_to_final_bits": p_metrics.get("kl_to_final_bits"),
        "answer_rank": p_metrics.get("answer_rank"),
        "cos_to_final": p_cos,
        "control_margin": None,
    }

    # Helper call should append an identical dict
    append_prism_pure_next_token(
        buf,
        layer_out_idx=7,
        prism_logits_all=logits_all,
        tokens_tensor=tokens_tensor,
        ctx_ids_list=ctx_ids_list,
        window_manager=wm,
        final_probs_tensor=final_probs,
        first_ans_token_id=first_ans_id,
        final_dir_vec=final_dir,
        copy_threshold=0.95,
        copy_margin=0.10,
        entropy_collapse_threshold=1.0,
        decode_id_fn=_decode_id,
        ground_truth=ground_truth,
        top_k_record=k,
        prompt_id="pos",
        prompt_variant="orig",
    )

    assert buf["pure_next_token_records"], "helper did not append a pure next token record"
    got = buf["pure_next_token_records"][-1]
    if got != manual:
        print("[DEBUG prism-equivalence] pure mismatch")
        print(" got:", got)
        print(" exp:", manual)
    assert got == manual


if __name__ == "__main__":
    import traceback
    print("Running prism sidecar equivalence tests…")
    ok = True
    try:
        test_prism_record_helper_matches_manual_block(); print("✅ record equivalence")
        test_prism_pure_next_token_helper_fields_match_manual_logic(); print("✅ pure equivalence")
    except AssertionError as e:
        print("❌ assertion failed:", e); traceback.print_exc(); ok = False
    except Exception as e:
        print("❌ test crashed:", e); traceback.print_exc(); ok = False
    raise SystemExit(0 if ok else 1)
