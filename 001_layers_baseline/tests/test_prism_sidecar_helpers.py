import _pathfix  # noqa: F401
import torch

from layers_core.windows import WindowManager
from layers_core.prism_sidecar import append_prism_record, append_prism_pure_next_token
from layers_core.collapse_rules import format_copy_strict_label, format_copy_soft_label

COPY_SOFT_WINDOW_KS = (1, 2, 3)
COPY_SOFT_THRESHOLD = 0.5
COPY_STRICT_LABEL = format_copy_strict_label(0.0)
COPY_SOFT_LABELS = {k: format_copy_soft_label(k, COPY_SOFT_THRESHOLD) for k in COPY_SOFT_WINDOW_KS}


def test_append_prism_record_appends_row():
    buf = {"records": []}
    logits = torch.randn(10)
    append_prism_record(
        buf,
        prompt_id="pos",
        prompt_variant="orig",
        layer=3,
        pos=1,
        token="tok",
        logits_pos=logits,
        decode_id_fn=lambda i: f"t{int(i)}",
        top_k=3,
    )
    assert len(buf["records"]) == 1, f"no record appended; buf={buf}"
    rec = buf["records"][0]
    assert rec["type"] == "record"
    assert rec["prompt_id"] == "pos"
    assert rec["prompt_variant"] == "orig"
    assert rec["layer"] == 3 and rec["pos"] == 1
    assert isinstance(rec["entropy"], float)
    assert len(rec["topk"]) == 3, f"unexpected topk len: {len(rec['topk'])} rec={rec}"


def test_append_prism_pure_next_token_builds_pure_record():
    torch.manual_seed(0)
    buf = {"pure_next_token_records": []}
    seq_len = 4
    vocab = 12
    prism_logits_all = torch.randn(seq_len, vocab)
    tokens = torch.zeros(1, seq_len, dtype=torch.long)
    wm = WindowManager(window_k=1, extra_window_ks=COPY_SOFT_WINDOW_KS)
    final_logits = torch.randn(vocab)
    final_probs = torch.softmax(final_logits, dim=0)
    final_dir = final_logits / (final_logits.norm() + 1e-12)

    append_prism_pure_next_token(
        buf,
        layer_out_idx=2,
        prism_logits_all=prism_logits_all,
        tokens_tensor=tokens,
        ctx_ids_list=[1, 2, 3],
        window_manager=wm,
        final_probs_tensor=final_probs,
        first_ans_token_id=None,
        final_dir_vec=final_dir,
        copy_threshold=0.0,
        copy_margin=0.0,
        copy_strict_label=COPY_STRICT_LABEL,
        copy_soft_threshold=COPY_SOFT_THRESHOLD,
        copy_soft_window_ks=COPY_SOFT_WINDOW_KS,
        copy_soft_labels=COPY_SOFT_LABELS,
        copy_soft_extra_labels={},
        entropy_collapse_threshold=10.0,
        decode_id_fn=lambda i: f"t{int(i)}",
        ground_truth="X",
        top_k_record=5,
        prompt_id="pos",
        prompt_variant="orig",
        p_uniform=1.0 / vocab,
    )
    assert len(buf["pure_next_token_records"]) == 1, f"no pure record appended; buf={buf}"
    rec = buf["pure_next_token_records"][0]
    assert rec["type"] == "pure_next_token_record"
    assert rec["layer"] == 2 and rec["pos"] == seq_len - 1
    assert rec["prompt_id"] == "pos" and rec["prompt_variant"] == "orig"
    # has expected extra metrics
    for k in ("copy_collapse", "entropy_collapse", "p_top1", "kl_to_final_bits", "answer_minus_uniform"):
        assert k in rec, f"missing key {k} in rec={rec}"


if __name__ == "__main__":
    import traceback
    print("Running prism sidecar helper tests…")
    ok = True
    try:
        test_append_prism_record_appends_row(); print("✅ append_prism_record")
        test_append_prism_pure_next_token_builds_pure_record(); print("✅ append_prism_pure_next_token")
    except AssertionError as e:
        print("❌ assertion failed:", e); traceback.print_exc(); ok = False
    except Exception as e:
        print("❌ test crashed:", e); traceback.print_exc(); ok = False
    raise SystemExit(0 if ok else 1)
