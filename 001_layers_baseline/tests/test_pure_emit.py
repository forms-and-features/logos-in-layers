import _pathfix  # noqa: F401
import torch

from layers_core.windows import WindowManager
from layers_core.pure_emit import compute_pure_next_token_info


def test_compute_pure_next_token_info_basic():
    torch.manual_seed(0)
    # Construct logits for a 1-position sequence (pos 0 is last)
    vocab = 6
    seq_len = 3
    d = 4
    logits_all = torch.randn(seq_len, vocab)
    tokens = torch.zeros(1, seq_len, dtype=torch.long)
    ctx_ids = [1, 2, 3, 4]
    wm = WindowManager(window_k=1)
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
    assert set(collected.keys()) >= {"layer", "copy_collapse", "entropy_collapse", "is_answer", "kl_to_final_bits", "answer_rank"}
    assert set(dual_ctx.keys()) >= {"layer", "last_pos", "last_logits_norm", "final_probs", "first_ans_id", "ground_truth"}

