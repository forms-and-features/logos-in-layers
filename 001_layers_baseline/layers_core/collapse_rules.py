import torch
import string


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


def is_id_subseq(needle: list[int], haystack: list[int]) -> bool:
    """Return True iff `needle` appears as a contiguous slice of `haystack`.

    Empty needle returns False (no event to match).
    """
    k = len(needle)
    if k == 0 or k > len(haystack):
        return False
    return any(haystack[i : i + k] == needle for i in range(len(haystack) - k + 1))


def detect_copy_collapse_id_subseq(
    logits: torch.Tensor,
    ctx_ids: list[int],
    window_ids: list[int],
    *,
    copy_threshold: float,
    copy_margin: float,
) -> bool:
    """Sub-word-aware copy-collapse detection using ID-level contiguous subsequence.

    Rule:
    - window_ids (k recent top-1 IDs) is a contiguous subsequence of ctx_ids
    - P(top-1) > copy_threshold
    - P(top-1) − P(top-2) > copy_margin

    Note: No entropy fallback here — keep entropy-based collapse as a separate flag.
    """
    if not window_ids:
        return False

    # Probability checks
    full_probs = torch.softmax(logits, dim=0)
    top2_vals, top2_idx = torch.topk(logits, 2, largest=True, sorted=True)
    top2_probs = full_probs[top2_idx]
    prob_1, prob_2 = top2_probs[0].item(), top2_probs[1].item()

    if not (prob_1 > copy_threshold and (prob_1 - prob_2) > copy_margin):
        return False

    # Subsequence check
    return is_id_subseq(window_ids, ctx_ids)


def is_pure_whitespace_or_punct(text: str) -> bool:
    """Heuristic guard: True if token decodes to only whitespace/punctuation.

    Used to avoid counting trivial spacing/punct echoes as copy events.
    """
    if text is None:
        return False
    if text.strip() == "":
        return True
    return all(ch.isspace() or (ch in string.punctuation) for ch in text)
