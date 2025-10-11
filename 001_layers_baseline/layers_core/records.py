from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def _pack_topk(top_tokens: Iterable[str], top_probs: Iterable[Any]) -> list[list[Any]]:
    # Preserve existing schema: list of [token_str, prob_float]
    packed = []
    for tok, prob in zip(top_tokens, top_probs):
        try:
            p = prob.item()  # torch scalar → Python float
        except Exception:
            p = float(prob)
        packed.append([tok, p])
    return packed


def make_record(
    *,
    prompt_id: str,
    prompt_variant: str,
    layer: int,
    pos: int,
    token: str,
    entropy: float,
    top_tokens: Iterable[str],
    top_probs: Iterable[Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rec = {
        "type": "record",
        "prompt_id": prompt_id,
        "prompt_variant": prompt_variant,
        "layer": layer,
        "pos": pos,
        "token": token,
        "entropy": entropy,
        "entropy_bits": entropy,
        "topk": _pack_topk(top_tokens, top_probs),
    }
    if extra:
        rec.update(extra)
    return rec


def make_pure_record(
    *,
    prompt_id: str,
    prompt_variant: str,
    layer: int,
    pos: int,
    token: str,
    entropy: float,
    top_tokens: Iterable[str],
    top_probs: Iterable[Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rec = {
        "type": "pure_next_token_record",
        "prompt_id": prompt_id,
        "prompt_variant": prompt_variant,
        "layer": layer,
        "pos": pos,
        "token": token,
        "entropy": entropy,
        "topk": _pack_topk(top_tokens, top_probs),
    }
    if extra:
        rec.update(extra)
    return rec
