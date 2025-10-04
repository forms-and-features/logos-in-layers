import torch
import string
from typing import Any, Callable, Dict


def _format_threshold_label(threshold: float) -> str:
    # Format with up to 6 decimals, trimming trailing zeros for uniqueness without noise.
    formatted = f"{threshold:.6f}".rstrip('0').rstrip('.')
    if formatted == "-0":
        formatted = "0"
    return formatted


def format_copy_strict_label(threshold: float) -> str:
    """Return canonical CSV/JSON label for strict copy detector."""
    return f"copy_strict@{_format_threshold_label(threshold)}"


def format_copy_soft_label(window_k: int, threshold: float) -> str:
    """Return canonical label for a soft copy detector at window size k."""
    return f"copy_soft_k{int(window_k)}@{_format_threshold_label(threshold)}"


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
    normalized = text.replace("▁", " ").replace("Ġ", " ").replace("Ċ", " ")
    if normalized.strip() == "":
        return True
    return all(ch.isspace() or (ch in string.punctuation) for ch in normalized)


def build_copy_ignore_mask(
    decode_id_fn: Callable[[int], str],
    vocab_size: int,
    *,
    sample_size: int = 16,
) -> Dict[str, Any]:
    """Return metadata about tokens ignored by copy detectors (whitespace/punct)."""
    ignored_ids: list[int] = []
    sample_tokens: list[str] = []
    for token_id in range(int(vocab_size)):
        try:
            tok = decode_id_fn(token_id)
        except Exception:
            tok = None
        if tok is None:
            continue
        if is_pure_whitespace_or_punct(tok):
            ignored_ids.append(int(token_id))
            if len(sample_tokens) < sample_size:
                sample_tokens.append(tok)
    return {
        "ignored_token_ids": ignored_ids,
        "ignored_token_str_sample": sample_tokens,
        "size": len(ignored_ids),
    }
