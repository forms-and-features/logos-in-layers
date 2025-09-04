from typing import Optional, Tuple


def _get_num(obj, names: list[str]) -> Optional[float]:
    """Return the first numeric attribute found on obj among names, as float.

    Mirrors the behavior in run.py: ignores missing attributes and non-numerics.
    """
    for n in names:
        try:
            v = getattr(obj, n)
        except Exception:
            continue
        if isinstance(v, (int, float)):
            return float(v)
    return None


def detect_head_transforms(model) -> Tuple[Optional[float], Optional[float]]:
    """Detect simple final-head transforms (scale, softcap) from model/config.

    Probes model.cfg first, then model, for typical attributes:
    - Scale:  ['final_logit_scale', 'logit_scale', 'final_logits_scale'] on cfg,
              then ['final_logit_scale', 'logit_scale'] on model.
    - Softcap:['final_logit_softcap', 'logit_softcap', 'final_logits_softcap', 'softcap'] on cfg,
              then ['final_logit_softcap', 'logit_softcap'] on model.
    Returns (scale, softcap) as floats or None if unavailable.
    Note: preserves original semantics where falsy zero values fall back.
    """
    cfg = getattr(model, 'cfg', None)
    scale = None
    softcap = None
    if cfg is not None:
        scale = _get_num(cfg, ['final_logit_scale', 'logit_scale', 'final_logits_scale']) or scale
        softcap = _get_num(cfg, ['final_logit_softcap', 'logit_softcap', 'final_logits_softcap', 'softcap']) or softcap
    scale = _get_num(model, ['final_logit_scale', 'logit_scale']) or scale
    softcap = _get_num(model, ['final_logit_softcap', 'logit_softcap']) or softcap
    return scale, softcap

