import torch
import torch.nn as nn

# Robust attribute candidates for RMSNorm scale across libraries/implementations
RMS_ATTR_CANDIDATES = ("w", "weight", "scale", "gamma")


def _get_rms_scale(norm_mod):
    """Return the learnable scale tensor for RMSNorm-like modules, or None.

    Tries common attribute names, then falls back to a single non-recursive parameter.
    """
    for attr in RMS_ATTR_CANDIDATES:
        if hasattr(norm_mod, attr):
            return getattr(norm_mod, attr)
    params = list(norm_mod.parameters(recurse=False))
    return params[0] if len(params) == 1 else None


def detect_model_architecture(model):
    """Detect pre-norm vs post-norm based on block child order.

    Returns 'post_norm' if ln2 comes after mlp in the block ordering; else 'pre_norm'.
    Works for GPT-J, NeoX, Falcon vs Llama/Mistral/Gemma families.
    """
    if not getattr(model, "blocks", None):
        return "pre_norm"

    block = model.blocks[0]
    kids = list(block.children())

    if hasattr(block, "ln2") and hasattr(block, "mlp"):
        try:
            ln2_idx = kids.index(block.ln2)
            mlp_idx = kids.index(block.mlp)
            return "post_norm" if ln2_idx > mlp_idx else "pre_norm"
        except ValueError:
            pass

    return "pre_norm"


def get_correct_norm_module(model, layer_idx, probe_after_block=True):
    """Select the correct normalization module based on probe timing and architecture.

    Rules (PROJECT_NOTES.md 1.1):
    - Pre-norm: after block → NEXT block ln1 (or ln_final for last layer); before block → current ln1
    - Post-norm: after block → current block ln2; before block → current ln1
    """
    if layer_idx >= len(model.blocks):
        return getattr(model, "ln_final", None)

    post_norm = detect_model_architecture(model) == "post_norm"

    if probe_after_block:
        if post_norm:
            return getattr(model.blocks[layer_idx], "ln2", None)
        else:
            if layer_idx + 1 < len(model.blocks):
                return getattr(model.blocks[layer_idx + 1], "ln1", None)
            else:
                return getattr(model, "ln_final", None)
    else:
        return getattr(model.blocks[layer_idx], "ln1", None)


def apply_norm_or_skip(residual: torch.Tensor, norm_module):
    """Apply model's own normalization (LayerNorm or RMSNorm) to a residual stream.

    - LayerNorm: faithful mean/var normalization with γ and β applied.
    - RMSNorm: epsilon is inside sqrt (x / sqrt(mean(x^2) + eps)) and learns scale.
    - If `norm_module` is None: return input unchanged (some models expose pre-norm).
    """
    if norm_module is None:
        return residual

    with torch.no_grad():
        if isinstance(norm_module, nn.LayerNorm):
            mean = residual.mean(dim=-1, keepdim=True)
            var = residual.var(dim=-1, unbiased=False, keepdim=True)
            normalized = (residual - mean) / torch.sqrt(var + norm_module.eps)
            weight = norm_module.weight.to(residual.dtype)
            bias = norm_module.bias.to(residual.dtype)
            return normalized * weight + bias
        else:
            rms = torch.sqrt(residual.pow(2).mean(-1, keepdim=True) + norm_module.eps)
            normalized = residual / rms
            scale = _get_rms_scale(norm_module)
            if scale is not None:
                scale = scale.detach().to(residual.device, dtype=residual.dtype)
                return normalized * scale
            return normalized

