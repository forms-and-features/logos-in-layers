from __future__ import annotations

from typing import Any, Callable


def make_decode_id(tokenizer: Any) -> Callable[[Any], str]:
    """Return a `decode_id(idx) -> str` function tolerant to ints or 0-D tensors.

    The returned function mirrors prior inlined behavior in run.py and falls
    back gracefully if `tokenizer.decode` is unavailable.
    """

    def decode_id(idx: Any) -> str:
        try:
            if hasattr(idx, "item"):
                idx = int(idx.item())
            else:
                idx = int(idx)
        except Exception:
            # best-effort cast; string fallback below
            pass
        # Prefer the tokenizer's decode API
        try:
            return tokenizer.decode([idx])
        except Exception:
            # Fallback to convert_ids_to_tokens if available
            conv = getattr(tokenizer, "convert_ids_to_tokens", None)
            if conv is not None:
                try:
                    tok = conv([idx])
                    if isinstance(tok, (list, tuple)) and tok:
                        return str(tok[0])
                except Exception:
                    pass
        return str(idx)

    return decode_id


__all__ = ["make_decode_id"]

