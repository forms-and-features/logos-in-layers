from typing import Any, Callable, Dict, List, Optional


def _choose_variant(
    ctx_ids: List[int],
    ctx_ans_ws: Optional[List[int]],
    ctx_ans_ns: Optional[List[int]],
    *,
    pieces_k: int,
    convert_ids_to_tokens: Optional[Callable[[List[int]], List[str]]] = None,
    decode_id: Optional[Callable[[int], str]] = None,
) -> Dict[str, Any]:
    """Common selection logic given precomputed ID sequences.

    Prefers the with-space variant if it strictly extends ctx_ids; otherwise uses
    the no-space variant on the same criterion. If neither extends, returns
    an unresolved result.
    """
    def _mk(ctx_ans: List[int], variant: str) -> Dict[str, Any]:
        start = len(ctx_ids)
        ans_ids = ctx_ans[start:]
        first_id = ans_ids[0] if len(ans_ids) > 0 else None
        # Token pieces (best-effort): use convert_ids_to_tokens if available; else decode single ids
        pieces: List[str] = []
        if len(ans_ids) > 0:
            take = ans_ids[: max(0, int(pieces_k))]
            if convert_ids_to_tokens is not None:
                try:
                    pieces = list(convert_ids_to_tokens(list(take)))
                except Exception:
                    pieces = []
            elif decode_id is not None:
                try:
                    pieces = [decode_id(int(t)) for t in take]
                except Exception:
                    pieces = []
        return {
            "status": "ok",
            "variant": variant,
            "first_id": first_id,
            "pieces": pieces,
            "answer_ids": ans_ids,
            "ctx_ids": ctx_ids,
            "ctx_len": len(ctx_ids),
        }

    # Prefer with-space if it truly extends
    if ctx_ans_ws is not None and len(ctx_ans_ws) > len(ctx_ids):
        return _mk(ctx_ans_ws, "with_space")
    # Fallback to no-space if it extends
    if ctx_ans_ns is not None and len(ctx_ans_ns) > len(ctx_ids):
        return _mk(ctx_ans_ns, "no_space")
    # Unresolved
    return {
        "status": "unresolved",
        "variant": "unknown",
        "first_id": None,
        "pieces": [],
        "answer_ids": [],
        "ctx_ids": ctx_ids,
        "ctx_len": len(ctx_ids),
        "reason": "no_variant_extended_context",
    }


def compute_gold_answer_info(
    tokenizer: Any,
    prompt: str,
    answer_str: str,
    *,
    pieces_k: int = 4,
) -> Dict[str, Any]:
    """Compute gold token alignment using the provided tokenizer.

    Returns a dict with keys: string, status, variant, first_id, pieces,
    answer_ids, ctx_ids, ctx_len. On failure, returns status="unresolved".
    """
    base: Dict[str, Any] = {
        "string": answer_str,
    }
    if tokenizer is None:
        return {
            **base,
            "status": "unresolved",
            "variant": "unknown",
            "first_id": None,
            "pieces": [],
            "answer_ids": [],
            "ctx_ids": [],
            "ctx_len": 0,
            "reason": "no_tokenizer",
        }
    try:
        # HF-style encode without adding special tokens
        ctx_ids = tokenizer.encode(prompt, add_special_tokens=False)
        ctx_ans_ws = tokenizer.encode(prompt + " " + answer_str, add_special_tokens=False)
        ctx_ans_ns = tokenizer.encode(prompt + answer_str, add_special_tokens=False)
        info = _choose_variant(
            ctx_ids,
            ctx_ans_ws,
            ctx_ans_ns,
            pieces_k=pieces_k,
            convert_ids_to_tokens=getattr(tokenizer, "convert_ids_to_tokens", None),
            decode_id=lambda i: tokenizer.decode([i]),
        )
        return {**base, **info}
    except Exception as e:
        return {
            **base,
            "status": "unresolved",
            "variant": "unknown",
            "first_id": None,
            "pieces": [],
            "answer_ids": [],
            "ctx_ids": [],
            "ctx_len": 0,
            "reason": f"tokenizer_error: {type(e).__name__}",
        }


def compute_gold_answer_info_from_sequences(
    ctx_ids: List[int],
    ctx_ans_ws: Optional[List[int]],
    ctx_ans_ns: Optional[List[int]],
    *,
    pieces_k: int = 4,
    convert_ids_to_tokens: Optional[Callable[[List[int]], List[str]]] = None,
    decode_id: Optional[Callable[[int], str]] = None,
    answer_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute gold token alignment from precomputed ID sequences.

    Use this when HF tokenizer isn't fully available; supply either
    convert_ids_to_tokens or a per-id decode function to obtain token strings.
    """
    info = _choose_variant(
        ctx_ids,
        ctx_ans_ws,
        ctx_ans_ns,
        pieces_k=pieces_k,
        convert_ids_to_tokens=convert_ids_to_tokens,
        decode_id=decode_id,
    )
    if answer_str is not None:
        info = {**info, "string": answer_str}
    return info

