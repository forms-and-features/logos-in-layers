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

    if kids and hasattr(block, "ln2"):
        last_child = kids[-1]
        if last_child is getattr(block, "ln2"):
            return "post_norm"

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
    """Apply model's normalization (LayerNorm or RMSNorm) with fp32 math, return original dtype.

    - Computes LN/RMS statistics in float32 to reduce rounding error under bf16/fp16,
      then casts the result back to the residual's dtype.
    - LayerNorm: mean/var + γ/β; RMSNorm: ε is inside sqrt; optional learned scale.
    - If `norm_module` is None: return input unchanged.
    """
    if norm_module is None:
        return residual

    with torch.no_grad():
        in_dtype = residual.dtype
        resid32 = residual.float()

        if isinstance(norm_module, nn.LayerNorm):
            mean32 = resid32.mean(dim=-1, keepdim=True)
            var32 = resid32.var(dim=-1, unbiased=False, keepdim=True)
            normalized32 = (resid32 - mean32) / torch.sqrt(var32 + float(norm_module.eps))
            weight32 = norm_module.weight.detach().to(residual.device, dtype=torch.float32)
            bias32 = norm_module.bias.detach().to(residual.device, dtype=torch.float32)
            out32 = normalized32 * weight32 + bias32
            return out32.to(dtype=in_dtype)
        else:
            # RMSNorm-style module
            rms32 = torch.sqrt(resid32.pow(2).mean(-1, keepdim=True) + float(getattr(norm_module, 'eps', 0.0)))
            normalized32 = resid32 / rms32
            scale = _get_rms_scale(norm_module)
            if scale is not None:
                scale32 = scale.detach().to(residual.device, dtype=torch.float32)
                out32 = normalized32 * scale32
            else:
                out32 = normalized32
            return out32.to(dtype=in_dtype)


def _norm_has_learned_scale(norm_mod):
    if norm_mod is None:
        return False
    if isinstance(norm_mod, nn.LayerNorm):
        return bool(norm_mod.weight is not None)
    return _get_rms_scale(norm_mod) is not None


def describe_norm_origin(model, layer_idx: int, probe_after_block: bool):
    """Return (ln_source, eps_inside_sqrt, scale_gamma_used) for diagnostics."""
    norm_module = get_correct_norm_module(model, layer_idx, probe_after_block=probe_after_block)
    eps_inside = True  # apply_norm_or_skip always keeps eps inside sqrt

    if norm_module is None:
        return ("raw", eps_inside, False)

    if layer_idx >= len(getattr(model, "blocks", [])):
        ln_source = "ln_final"
    else:
        arch = detect_model_architecture(model)
        if probe_after_block:
            if arch == "post_norm":
                ln_source = f"blocks[{layer_idx}].ln2"
            else:
                if layer_idx + 1 < len(model.blocks):
                    ln_source = f"blocks[{layer_idx + 1}].ln1"
                else:
                    ln_source = "ln_final"
        else:
            ln_source = f"blocks[{layer_idx}].ln1"

    scale_used = _norm_has_learned_scale(norm_module)
    return (ln_source, eps_inside, scale_used)
