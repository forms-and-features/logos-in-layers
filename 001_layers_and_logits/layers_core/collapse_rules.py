import torch


def detect_copy_collapse(
    logits: torch.Tensor,
    prompt_token_ids: set,
    *,
    copy_threshold: float,
    copy_margin: float,
    entropy_bits: float | None = None,
    entropy_fallback_threshold: float = 1.0,
) -> bool:
    """Detect copy-collapse (prompt echo) at the pure next-token position.

    Rule:
    - Top-1 token ∈ prompt_token_ids
    - P(top-1) > copy_threshold
    - P(top-1) − P(top-2) > copy_margin
    Optional fallback: if entropy_bits is provided and < entropy_fallback_threshold, treat as collapse.
    """
    full_probs = torch.softmax(logits, dim=0)
    top2_vals, top2_idx = torch.topk(logits, 2, largest=True, sorted=True)
    top2_probs = full_probs[top2_idx]
    token_id_1, token_id_2 = top2_idx[0].item(), top2_idx[1].item()
    prob_1, prob_2 = top2_probs[0].item(), top2_probs[1].item()

    collapsed = (
        token_id_1 in prompt_token_ids and
        prob_1 > copy_threshold and
        (prob_1 - prob_2) > copy_margin
    )

    if (not collapsed) and (entropy_bits is not None) and (entropy_bits < entropy_fallback_threshold):
        collapsed = True

    return collapsed


def is_semantic_top1(top1_text: str, ground_truth: str) -> bool:
    """Check if the top-1 decoded token matches the ground truth after trimming.

    Matches existing behavior in run.py which uses simple `.strip()` equality.
    """
    return top1_text.strip() == ground_truth

