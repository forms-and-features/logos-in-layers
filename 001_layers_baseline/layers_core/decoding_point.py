from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import torch

from .metrics import compute_next_token_metrics


def _to_cpu_float_vector(tensor: torch.Tensor) -> torch.Tensor:
    """Return a detached, float32, CPU 1-D tensor."""
    if tensor is None:
        raise ValueError("tensor is None")
    vec = tensor.detach()
    if vec.ndim != 1:
        vec = vec.reshape(-1)
    return vec.to(dtype=torch.float32, device="cpu")


def _topk_ids(logits: torch.Tensor, k: int) -> List[int]:
    if logits.numel() == 0 or k <= 0:
        return []
    k_eff = min(int(k), logits.numel())
    values, indices = torch.topk(logits, k_eff, largest=True, sorted=True)
    if values.numel() == 0:
        return []
    return [int(idx.item()) for idx in indices]


def _jaccard(set_a: Iterable[int], set_b: Iterable[int]) -> Optional[float]:
    A = set(int(x) for x in set_a)
    B = set(int(x) for x in set_b)
    if not A and not B:
        return 1.0
    union = A | B
    if not union:
        return None
    intersection = A & B
    return float(len(intersection)) / float(len(union))


def _spearman_from_lists(list_a: List[int], list_b: List[int], *, default_rank: int) -> Optional[float]:
    """Compute Spearman correlation treating missing tokens as default_rank."""
    tokens = set(list_a) | set(list_b)
    if len(tokens) < 2:
        return None
    rank_a: Dict[int, float] = {tok: float(i + 1) for i, tok in enumerate(list_a)}
    rank_b: Dict[int, float] = {tok: float(i + 1) for i, tok in enumerate(list_b)}
    ranks_a: List[float] = []
    ranks_b: List[float] = []
    for tok in tokens:
        ranks_a.append(rank_a.get(tok, float(default_rank)))
        ranks_b.append(rank_b.get(tok, float(default_rank)))
    ta = torch.tensor(ranks_a, dtype=torch.float32)
    tb = torch.tensor(ranks_b, dtype=torch.float32)
    if ta.numel() < 2:
        return None
    ta_mean = torch.mean(ta)
    tb_mean = torch.mean(tb)
    ta_centered = ta - ta_mean
    tb_centered = tb - tb_mean
    denom = torch.norm(ta_centered, p=2) * torch.norm(tb_centered, p=2)
    if denom <= 0 or not torch.isfinite(denom):
        return None
    corr = torch.dot(ta_centered, tb_centered) / denom
    try:
        return float(torch.clamp(corr, -1.0, 1.0).item())
    except Exception:
        return None


def compare_decoding_strategies(
    *,
    logits_same_ln2: torch.Tensor,
    logits_next_ln1: torch.Tensor,
    final_probs: torch.Tensor,
    answer_token_id: Optional[int],
    topk_values: Sequence[int] = (10, 50),
) -> Dict[str, Any]:
    """Compare two decoding strategies applied to the same residual.

    Args:
        logits_same_ln2: Logits from applying the current block's ln2.
        logits_next_ln1: Logits from applying the next block's ln1 (baseline).
        final_probs: Final model distribution for KL/rank comparisons.
        answer_token_id: Optional token id for the gold answer.
        topk_values: Iterable of K values for jaccard comparisons.

    Returns:
        Dictionary with rank/overlap metrics for the two strategies.
    """
    logits_a = _to_cpu_float_vector(logits_same_ln2)
    logits_b = _to_cpu_float_vector(logits_next_ln1)
    probs_final = _to_cpu_float_vector(final_probs)

    max_k = max(1, *(int(k) for k in topk_values if isinstance(k, (int, float))))
    top_ids_a = _topk_ids(logits_a, max_k)
    top_ids_b = _topk_ids(logits_b, max_k)

    if not top_ids_a or not top_ids_b:
        raise ValueError("Unable to compute top-1 tokens for decoding strategies")

    probs_a = torch.softmax(logits_a, dim=0)
    probs_b = torch.softmax(logits_b, dim=0)

    top1_a = int(top_ids_a[0])
    top1_b = int(top_ids_b[0])

    metrics_a = compute_next_token_metrics(probs_a, top1_a, probs_final, answer_token_id)
    metrics_b = compute_next_token_metrics(probs_b, top1_b, probs_final, answer_token_id)

    answer_rank_a = metrics_a.get("answer_rank")
    answer_rank_b = metrics_b.get("answer_rank")

    def _as_int(value: Any) -> Optional[int]:
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    answer_rank_a_int = _as_int(answer_rank_a)
    answer_rank_b_int = _as_int(answer_rank_b)

    if answer_rank_a_int is None or answer_rank_b_int is None:
        rank1_agree: Optional[bool] = None
    else:
        rank1_agree = bool(
            answer_rank_a_int == 1 and
            answer_rank_b_int == 1
        )
    delta_answer_rank: Optional[int] = None
    if answer_rank_a_int is not None and answer_rank_b_int is not None:
        delta_answer_rank = int(answer_rank_a_int - answer_rank_b_int)

    jaccard_results: Dict[int, Optional[float]] = {}
    for k in topk_values:
        try:
            k_int = int(k)
        except (TypeError, ValueError):
            continue
        if k_int <= 0:
            continue
        jaccard_results[k_int] = _jaccard(top_ids_a[:k_int], top_ids_b[:k_int])

    spearman_default_rank = max_k + 1
    spearman_val = _spearman_from_lists(top_ids_a[:max_k], top_ids_b[:max_k], default_rank=spearman_default_rank)

    result: Dict[str, Any] = {
        "rank1_agree": rank1_agree,
        "delta_answer_rank": delta_answer_rank,
        "top1_token_agree": bool(top1_a == top1_b),
        "answer_rank_post_ln2": answer_rank_a_int,
        "answer_rank_next_ln1": answer_rank_b_int,
        "top1_id_post_ln2": top1_a,
        "top1_id_next_ln1": top1_b,
        "p_answer_post_ln2": metrics_a.get("p_answer"),
        "p_answer_next_ln1": metrics_b.get("p_answer"),
        "p_top1_post_ln2": metrics_a.get("p_top1"),
        "p_top1_next_ln1": metrics_b.get("p_top1"),
        "spearman_top50": spearman_val,
    }

    for k, value in jaccard_results.items():
        result[f"jaccard@{k}"] = value

    return result
