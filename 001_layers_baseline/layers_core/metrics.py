import torch
from typing import Optional, Dict, Any

from .numerics import kl_bits


def compute_next_token_metrics(
    probs: torch.Tensor,
    top1_id: int,
    final_probs: torch.Tensor,
    first_ans_id: Optional[int],
    *,
    topk_cum: int = 5,
) -> Dict[str, Any]:
    """Compute p_top1, cumulative p_topk, p_answer, answer_rank, and KL-to-final (bits).

    Args:
        probs: 1D probability tensor over vocab (float32 recommended)
        top1_id: index of the top-1 token for this distribution
        final_probs: 1D probability tensor of the final model head distribution
        first_ans_id: index of the gold first answer token, or None if unknown
        topk_cum: K for cumulative probability (default 5)

    Returns a dict with keys: p_top1, p_top5, p_answer, answer_rank, kl_to_final_bits.
    If first_ans_id is None, p_answer and answer_rank are None.
    """
    # p_top1
    p_top1 = probs[top1_id].item()

    # p_topk cumulative
    k = min(int(topk_cum), probs.shape[-1])
    p_topk = probs.topk(k).values.sum().item()

    # p_answer & rank
    if first_ans_id is not None and 0 <= first_ans_id < probs.shape[-1]:
        p_answer = probs[first_ans_id].item()
        answer_rank = int(1 + (probs > p_answer).sum().item())
    else:
        p_answer = None
        answer_rank = None

    # KL to final
    kl_b = kl_bits(probs, final_probs)

    return {
        "p_top1": p_top1,
        "p_top5": p_topk,  # fixed name expected by downstream
        "p_answer": p_answer,
        "answer_rank": answer_rank,
        "kl_to_final_bits": kl_b,
    }

