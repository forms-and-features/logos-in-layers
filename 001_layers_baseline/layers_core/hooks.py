from typing import Tuple, List, Dict, Any


def build_cache_hook(cache_dict: Dict[str, Any]):
    """Return a hook fn that stores detached tensors keyed by hook.name."""
    def cache_residual_hook(tensor, hook):
        cache_dict[hook.name] = tensor.detach()
    return cache_residual_hook


def attach_residual_hooks(model, cache_hook) -> Tuple[List[Any], bool]:
    """Attach residual hooks to embeddings, optional positional emb, and resid_post per layer.

    Returns (handles, has_pos_embed).
    """
    hooks = []
    # Embedding hook
    embed_hook = model.hook_dict['hook_embed'].add_hook(cache_hook)
    hooks.append(embed_hook)
    # Optional positional embedding
    has_pos_embed = False
    if 'hook_pos_embed' in model.hook_dict:
        pos_hook = model.hook_dict['hook_pos_embed'].add_hook(cache_hook)
        hooks.append(pos_hook)
        has_pos_embed = True

    # Per-layer residual post hooks
    n_layers = model.cfg.n_layers
    for layer in range(n_layers):
        resid_hook = model.blocks[layer].hook_resid_post.add_hook(cache_hook)
        hooks.append(resid_hook)

    return hooks, has_pos_embed


def detach_hooks(handles: List[Any]) -> None:
    for h in handles:
        if h is not None:
            h.remove()

