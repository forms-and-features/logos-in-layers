#!/usr/bin/env python3
"""Equivalence checks for prism sidecar helpers vs prior inline logic.

CPU-only, no model/HF. Uses deterministic tensors to compare outputs.
"""

import _pathfix  # noqa: F401

import math
import torch

from layers_core.prism_sidecar import (
    append_prism_record,
    append_prism_pure_next_token,
)
from layers_core.collapse_rules import format_copy_strict_label, format_copy_soft_label

COPY_THRESH_STRICT = 0.95
COPY_SOFT_THRESHOLD = 0.5
COPY_SOFT_WINDOW_KS = (1, 2, 3)
COPY_STRICT_LABEL = format_copy_strict_label(COPY_THRESH_STRICT)
COPY_SOFT_LABELS = {k: format_copy_soft_label(k, COPY_SOFT_THRESHOLD) for k in COPY_SOFT_WINDOW_KS}
from layers_core.numerics import bits_entropy_from_logits
from layers_core.metrics import compute_next_token_metrics
from layers_core.collapse_rules import (
    detect_copy_collapse_id_subseq,
    is_pure_whitespace_or_punct,
    is_id_subseq,
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
        "entropy_bits": pent,
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
    buf = {"pure_next_token_records": [], "copy_flag_columns": [COPY_STRICT_LABEL, *[COPY_SOFT_LABELS[k] for k in COPY_SOFT_WINDOW_KS]]}

    # Setup synthetic tokens/logits (seq_len=5, vocab=9)
    S, V = 5, 9
    logits_all = torch.randn(S, V, dtype=torch.float32)
    tokens_tensor = torch.arange(S)[None, :]
    ctx_ids_list = list(range(S - 1))  # pretend context ids are 0..S-2
    wm_manual = WindowManager(window_k=1, extra_window_ks=COPY_SOFT_WINDOW_KS)
    wm_helper = WindowManager(window_k=1, extra_window_ks=COPY_SOFT_WINDOW_KS)
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
    prism_window_ids = wm_manual.append_and_trim("prism", "pos", "orig", int(p_top1_id))
    strict_window = wm_manual.get_window("prism", "pos", "orig", wm_manual.window_k)
    p_copy = detect_copy_collapse_id_subseq(
        pz, ctx_ids_list, strict_window, copy_threshold=COPY_THRESH_STRICT, copy_margin=0.10
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
    soft_hits = {}
    for k_soft in COPY_SOFT_WINDOW_KS:
        window_slice = wm_manual.get_window("prism", "pos", "orig", k_soft)
        soft_hits[k_soft] = (
            len(window_slice) >= k_soft
            and is_id_subseq(window_slice, ctx_ids_list)
            and p_metrics.get("p_top1", 0.0) > COPY_SOFT_THRESHOLD
        )

    answer_logit_gap = None
    answer_vs_top1_gap = None
    last_logits = logits_all[last_pos]
    if first_ans_id is not None and 0 <= int(first_ans_id) < last_logits.shape[-1]:
        ans_logit = float(last_logits[int(first_ans_id)].item())
        top_vals, top_idx = torch.topk(last_logits, 2, largest=True, sorted=True)
        top1_logit = float(top_vals[0].item())
        top1_idx = int(top_idx[0].item())
        if p_metrics.get("answer_rank") == 1 and top1_idx == int(first_ans_id):
            if len(top_vals) > 1:
                answer_logit_gap = ans_logit - float(top_vals[1].item())
        elif p_metrics.get("answer_rank") and int(p_metrics.get("answer_rank")) > 1:
            answer_vs_top1_gap = ans_logit - top1_logit

    manual = {
        "type": "pure_next_token_record",
        "prompt_id": "pos",
        "prompt_variant": "orig",
        "layer": 7,  # arbitrary layer id passed below
        "pos": last_pos,
        "token": "⟨NEXT⟩",
        "entropy": pent,
        "entropy_bits": pent,
        "topk": [[tok, float(prob.item())] for tok, prob in zip(p_top_tokens, p_top_probs)],
        "copy_collapse": p_copy,
        COPY_STRICT_LABEL: p_copy,
        "entropy_collapse": pent <= 1.0,
        "is_answer": p_is_answer,
        "top1_token_id": int(p_top1_id),
        "top1_token_str": p_top_tokens[0] if p_top_tokens else None,
        "p_top1": p_metrics.get("p_top1"),
        "p_top5": p_metrics.get("p_top5"),
        "p_answer": p_metrics.get("p_answer"),
        "kl_to_final_bits": p_metrics.get("kl_to_final_bits"),
        "kl_to_final_bits_norm_temp": None,
        "answer_rank": p_metrics.get("answer_rank"),
        "cos_to_final": p_cos,
        # add surface/geom placeholders present in helper output
        "cos_to_answer": None,
        "cos_to_prompt_max": None,
        "geom_crossover": None,
        "echo_mass_prompt": None,
        "answer_mass": None,
        "answer_minus_echo_mass": None,
        "mass_ratio_ans_over_prompt": None,
        "topk_prompt_mass@50": None,
        "teacher_entropy_bits": float(
            -(final_probs * (final_probs + 1e-30).log()).sum().item() / math.log(2)
        ),
        "control_margin": None,
        "resid_norm_ratio": None,
        "delta_resid_cos": None,
        "answer_logit_gap": answer_logit_gap,
        "answer_vs_top1_gap": answer_vs_top1_gap,
    }
    for k_soft, label in COPY_SOFT_LABELS.items():
        manual[label] = soft_hits.get(k_soft, False)

    p_uniform = 1.0 / float(final_probs.shape[0])
    if p_metrics.get("p_answer") is not None:
        manual["answer_minus_uniform"] = float(p_metrics.get("p_answer")) - p_uniform
    else:
        manual["answer_minus_uniform"] = None

    # Helper call should append an identical dict
    append_prism_pure_next_token(
        buf,
        layer_out_idx=7,
        prism_logits_all=logits_all,
        tokens_tensor=tokens_tensor,
        ctx_ids_list=ctx_ids_list,
        window_manager=wm_helper,
        final_probs_tensor=final_probs,
        first_ans_token_id=first_ans_id,
        final_dir_vec=final_dir,
        copy_threshold=COPY_THRESH_STRICT,
        copy_margin=0.10,
        copy_strict_label=COPY_STRICT_LABEL,
        copy_soft_threshold=COPY_SOFT_THRESHOLD,
        copy_soft_window_ks=COPY_SOFT_WINDOW_KS,
        copy_soft_labels=COPY_SOFT_LABELS,
        copy_soft_extra_labels={},
        entropy_collapse_threshold=1.0,
        decode_id_fn=_decode_id,
        ground_truth=ground_truth,
        top_k_record=k,
        prompt_id="pos",
        prompt_variant="orig",
        p_uniform=p_uniform,
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
