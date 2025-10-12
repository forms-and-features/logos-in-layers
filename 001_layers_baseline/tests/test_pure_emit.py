import _pathfix  # noqa: F401
import torch

from layers_core.windows import WindowManager
from layers_core.pure_emit import compute_pure_next_token_info
from layers_core.collapse_rules import format_copy_strict_label, format_copy_soft_label


def test_compute_pure_next_token_info_basic():
    torch.manual_seed(0)
    # Construct logits for a 1-position sequence (pos 0 is last)
    vocab = 6
    seq_len = 3
    d = 4
    logits_all = torch.randn(seq_len, vocab)
    tokens = torch.zeros(1, seq_len, dtype=torch.long)
    ctx_ids = [1, 2, 3, 4]
    wm = WindowManager(window_k=1, extra_window_ks=[1, 2, 3])
    final_logits = torch.randn(vocab)
    final_probs = torch.softmax(final_logits, dim=0)
    final_dir = final_logits / (final_logits.norm() + 1e-12)

    # decode_id just returns f"t{id}"
    decode_id = lambda idx: f"t{int(idx)}"

    view, collected, dual_ctx = compute_pure_next_token_info(
        layer_out_idx=1,
        logits_all=logits_all,
        tokens_tensor=tokens,
        ctx_ids_list=ctx_ids,
        window_manager=wm,
        lens_type="norm",
        final_probs_tensor=final_probs,
        first_ans_token_id=None,
        final_dir_vec=final_dir,
        copy_threshold=0.0,
        copy_margin=0.0,
        copy_strict_label=format_copy_strict_label(0.0),
        copy_soft_threshold=0.33,
        copy_soft_window_ks=(1, 2, 3),
        copy_soft_labels={k: format_copy_soft_label(k, 0.33) for k in (1, 2, 3)},
        copy_soft_extra_labels={},
        entropy_collapse_threshold=10.0,
        decode_id_fn=decode_id,
        ground_truth="X",
        top_k_record=3,
        prompt_id="pos",
        prompt_variant="orig",
    )

    assert isinstance(view["pos"], int)
    assert view["token_str"] == "⟨NEXT⟩"
    assert isinstance(view["entropy_bits"], float)
    assert len(view["top_tokens"]) == 3 and len(view["top_probs"]) == 3
    assert set(collected.keys()) >= {"layer", "copy_collapse", "entropy_collapse", "is_answer", "kl_to_final_bits", "answer_rank", "copy_soft_hits", "top1_token_id"}
    assert isinstance(collected["copy_soft_hits"], dict)
    assert "cos_to_final" in collected
    assert isinstance(collected["cos_to_final"], float)
    assert set(dual_ctx.keys()) >= {"layer", "last_pos", "last_logits_norm", "final_probs", "first_ans_id", "ground_truth"}
    assert "top1_token_id" in view["record_extra"]
    assert "top1_token_str" in view["record_extra"]
    assert "entropy_bits" in view["record_extra"]
    assert "teacher_entropy_bits" in view["record_extra"]
    assert "resid_norm_ratio" in view["record_extra"]
    assert "delta_resid_cos" in view["record_extra"]
    assert "answer_logit_gap" in view["record_extra"]
    assert "answer_vs_top1_gap" in view["record_extra"]
    assert "answer_minus_uniform" in view["record_extra"]
    assert "semantic_margin_ok" in view["record_extra"]
    # Strict-sweep hits are included in collected or record extra
    assert "copy_strict_hits" in collected
    assert "entropy_bits" in collected
    assert "teacher_entropy_bits" in collected
    assert "answer_minus_uniform" in collected
    assert "semantic_margin_ok" in collected
    for lab in ("copy_strict@0.7", "copy_strict@0.8", "copy_strict@0.9", "copy_strict@0.95"):
        assert lab in collected["copy_strict_hits"]


def test_cosine_bias_invariance():
    vocab = 4
    seq_len = 2
    logits_base = torch.tensor(
        [
            [0.1, -0.2, 0.05, 0.0],
            [0.3, 0.0, -0.4, 0.6],
        ],
        dtype=torch.float32,
    )
    bias = torch.full((vocab,), 0.75, dtype=torch.float32)
    logits_shifted = logits_base + bias

    tokens = torch.zeros(1, seq_len, dtype=torch.long)
    ctx_ids = [0, 1]
    ground_truth = "tok0"
    decode_id = lambda idx: f"tok{int(idx)}"
    wm_1 = WindowManager(window_k=1, extra_window_ks=[1, 2])
    wm_2 = WindowManager(window_k=1, extra_window_ks=[1, 2])

    final_logits = logits_base[-1]
    final_probs = torch.softmax(final_logits, dim=0)
    final_dir = final_logits / (final_logits.norm() + 1e-12)

    copy_soft_labels = {1: format_copy_soft_label(1, 0.5)}

    view_base, _, _ = compute_pure_next_token_info(
        layer_out_idx=1,
        logits_all=logits_base,
        tokens_tensor=tokens,
        ctx_ids_list=ctx_ids,
        window_manager=wm_1,
        lens_type="norm",
        final_probs_tensor=final_probs,
        first_ans_token_id=0,
        final_dir_vec=final_dir,
        copy_threshold=0.5,
        copy_margin=0.1,
        copy_strict_label=format_copy_strict_label(0.5),
        copy_soft_threshold=0.5,
        copy_soft_window_ks=(1,),
        copy_soft_labels=copy_soft_labels,
        copy_soft_extra_labels={},
        entropy_collapse_threshold=5.0,
        decode_id_fn=decode_id,
        ground_truth=ground_truth,
        top_k_record=3,
        prompt_id="pos",
        prompt_variant="orig",
    )

    view_bias, _, _ = compute_pure_next_token_info(
        layer_out_idx=1,
        logits_all=logits_shifted,
        tokens_tensor=tokens,
        ctx_ids_list=ctx_ids,
        window_manager=wm_2,
        lens_type="norm",
        final_probs_tensor=final_probs,
        first_ans_token_id=0,
        final_dir_vec=final_dir,
        copy_threshold=0.5,
        copy_margin=0.1,
        copy_strict_label=format_copy_strict_label(0.5),
        copy_soft_threshold=0.5,
        copy_soft_window_ks=(1,),
        copy_soft_labels=copy_soft_labels,
        copy_soft_extra_labels={},
        entropy_collapse_threshold=5.0,
        decode_id_fn=decode_id,
        ground_truth=ground_truth,
        top_k_record=3,
        prompt_id="pos",
        prompt_variant="orig",
        bias_tensor=bias,
    )

    cos_base = view_base["record_extra"]["cos_to_final"]
    cos_bias = view_bias["record_extra"]["cos_to_final"]
    assert abs(cos_base - cos_bias) < 1e-6


def test_control_top2_logit_gap_emitted():
    vocab = 4
    seq_len = 2
    logits_all = torch.zeros(seq_len, vocab)
    logits_all[-1] = torch.tensor([2.0, 3.0, 1.0, -1.0], dtype=torch.float32)
    tokens = torch.zeros(1, seq_len, dtype=torch.long)
    ctx_ids = [0, 1]
    wm = WindowManager(window_k=1, extra_window_ks=[1, 2])
    final_logits = torch.tensor([1.0, 1.5, -0.2, 0.0], dtype=torch.float32)
    final_probs = torch.softmax(final_logits, dim=0)
    final_dir = final_logits / (final_logits.norm() + 1e-12)

    decode_id = lambda idx: f"tok{int(idx)}"
    control_ids = (1, 0)  # Paris id =1, Berlin id =0

    view, collected, _ = compute_pure_next_token_info(
        layer_out_idx=1,
        logits_all=logits_all,
        tokens_tensor=tokens,
        ctx_ids_list=ctx_ids,
        window_manager=wm,
        lens_type="norm",
        final_probs_tensor=final_probs,
        first_ans_token_id=1,
        final_dir_vec=final_dir,
        copy_threshold=0.5,
        copy_margin=0.1,
        copy_strict_label=format_copy_strict_label(0.5),
        copy_soft_threshold=0.5,
        copy_soft_window_ks=(1,),
        copy_soft_labels={1: format_copy_soft_label(1, 0.5)},
        copy_soft_extra_labels={},
        entropy_collapse_threshold=5.0,
        decode_id_fn=decode_id,
        ground_truth="tok1",
        top_k_record=3,
        prompt_id="ctl",
        prompt_variant="orig",
        control_ids=control_ids,
    )

    gap_view = view["record_extra"]["control_top2_logit_gap"]
    gap_collected = collected["control_top2_logit_gap"]
    assert gap_view is not None
    assert gap_collected is not None
    assert abs(gap_view - 1.0) < 1e-6
    assert abs(gap_collected - 1.0) < 1e-6
