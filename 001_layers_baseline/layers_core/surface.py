"""Helpers for surface-mass and geometric diagnostics (PROJECT_NOTES §1.13–§1.15)."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import math
import torch

from .collapse_rules import is_pure_whitespace_or_punct


def build_prompt_vocab_ids(ctx_ids: Sequence[int], decode_id_fn) -> List[int]:
    """Return unique prompt token ids, excluding whitespace/punctuation tokens."""
    uniq: List[int] = []
    seen = set()
    for tid in ctx_ids:
        if tid in seen:
            continue
        try:
            tok = decode_id_fn(tid)
        except Exception:
            tok = None
        if tok is not None and is_pure_whitespace_or_punct(tok):
            continue
        uniq.append(int(tid))
        seen.add(tid)
    return uniq


def compute_surface_masses(
    probs: torch.Tensor,
    prompt_vocab_ids: Iterable[int],
    answer_token_id: Optional[int],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (echo_mass, answer_mass, ratio) for a given probability vector."""
    echo_mass = None
    answer_mass = None
    if probs is not None:
        try:
            echo_mass = float(sum(float(probs[int(t)].item()) for t in prompt_vocab_ids))
        except Exception:
            echo_mass = None
        if answer_token_id is not None and 0 <= int(answer_token_id) < probs.shape[-1]:
            try:
                answer_mass = float(probs[int(answer_token_id)].item())
            except Exception:
                answer_mass = None
    ratio = None
    if echo_mass is not None and answer_mass is not None:
        ratio = float(answer_mass / (echo_mass + 1e-9))
    return echo_mass, answer_mass, ratio


def compute_geometric_cosines(
    resid_vec: Optional[torch.Tensor],
    decoder_weight: Optional[torch.Tensor],
    prompt_vocab_ids: Iterable[int],
    answer_token_id: Optional[int],
) -> Tuple[Optional[float], Optional[float]]:
    """Return cos(resid, answer_decoder_col) and max cos to prompt columns."""
    if resid_vec is None or decoder_weight is None:
        return None, None
    r = resid_vec.detach().to(dtype=torch.float32)
    rn = float(torch.norm(r))
    if not math.isfinite(rn) or rn <= 0.0:
        return None, None
    ru = r / (rn + 1e-12)

    def col(ix: int) -> Optional[torch.Tensor]:
        if decoder_weight.dim() < 2:
            return None
        vocab = decoder_weight.shape[1]
        if ix < 0 or ix >= vocab:
            return None
        try:
            c = decoder_weight[:, int(ix)]
        except Exception:
            return None
        return c.to(device=ru.device, dtype=torch.float32)

    cos_ans = None
    if answer_token_id is not None:
        ca = col(int(answer_token_id))
        if ca is not None:
            dn = float(torch.norm(ca))
            if math.isfinite(dn) and dn > 0.0:
                cos_ans = float(torch.dot(ru, ca / (dn + 1e-12)))

    best = None
    for t in prompt_vocab_ids:
        cp = col(int(t))
        if cp is None:
            continue
        dn = float(torch.norm(cp))
        if not math.isfinite(dn) or dn <= 0.0:
            continue
        val = float(torch.dot(ru, cp / (dn + 1e-12)))
        if best is None or val > best:
            best = val
    return cos_ans, best


def compute_topk_prompt_mass(probs: torch.Tensor, prompt_vocab_ids: Iterable[int], k: int) -> Optional[float]:
    """Return sum of prompt-token probabilities among the Top-K by P."""
    if probs is None or k <= 0:
        return None
    k_eff = max(1, min(int(k), int(probs.shape[-1])))
    try:
        vals, idxs = torch.topk(probs, k_eff, largest=True, sorted=True)
    except Exception:
        return None
    prompt_set = set(int(t) for t in prompt_vocab_ids)
    total = 0.0
    for v, i in zip(vals, idxs):
        if int(i.item()) in prompt_set:
            total += float(v.item())
    return float(total)

