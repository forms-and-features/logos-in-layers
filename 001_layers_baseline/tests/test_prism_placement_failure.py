#!/usr/bin/env python3
"""Simulate Prism Q placement failure and assert behavior.

Replaces PrismLensAdapter in passes with a stub that reports a placement_error
once and disables itself, ensuring no sidecar rows are emitted.
"""

import _pathfix  # noqa: F401

import torch

from layers_core.passes import run_prompt_pass
import layers_core.passes as passes_mod
from layers_core.lenses import NormLensAdapter
from layers_core.windows import WindowManager
from layers_core.prism import WhitenStats
from layers_core.contexts import UnembedContext, PrismContext
from layers_core.collapse_rules import format_copy_strict_label, format_copy_soft_label

COPY_SOFT_WINDOW_KS = (1, 2, 3)
COPY_SOFT_THRESHOLD = 0.5
COPY_STRICT_LABEL = format_copy_strict_label(0.95)
COPY_SOFT_LABELS = {k: format_copy_soft_label(k, COPY_SOFT_THRESHOLD) for k in COPY_SOFT_WINDOW_KS}


class _FakePrismLensAdapter:
    def __init__(self, stats, Q, active):
        self.enabled = bool(active)
        self.diag = {}
        self._fired = False

    def forward(self, *args, **kwargs):
        if not self.enabled:
            return None
        if not self._fired:
            self._fired = True
            # Simulate placement failure on first use
            self.diag["placement_error"] = "prism Q placement failed: Simulated"
            self.enabled = False
        return None


def _decode_id(idx):
    if hasattr(idx, 'item'):
        idx = int(idx.item())
    return f"T{idx}"


class _ModelStub:
    # Minimal stub; reuse from test_pass_runner_minimal via simple inline clone
    def __init__(self, d_model=8, n_layers=2, vocab=11):
        import torch.nn as nn

        class _Hookable:
            def __init__(self, name: str):
                self.name = name
                self._hooks = []
            def add_hook(self, fn):
                self._hooks.append(fn)
                class H:
                    def __init__(self, name):
                        self.name = name
                return type('Hdl', (), {'remove': lambda s: self._hooks.remove(fn) if fn in self._hooks else None})()
            def fire(self, tensor, name: str | None = None):
                for fn in list(self._hooks):
                    class H:
                        def __init__(self, name):
                            self.name = name
                    fn(tensor, H(name or self.name))

        class _Block(nn.Module):
            def __init__(self, d_model: int):
                super().__init__()
                self.ln1 = nn.LayerNorm(d_model)
                self.mlp = nn.Sequential(nn.Linear(d_model, d_model))
                self.hook_resid_post = _Hookable('blocks.X.hook_resid_post')

        import torch.nn as nn
        self.cfg = type('cfg', (), {'n_layers': n_layers, 'd_model': d_model, 'model_name': 'stub'})
        self.blocks = nn.ModuleList([_Block(d_model) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.hook_dict = {
            'hook_embed': _Hookable('hook_embed'),
        }
        self.tokenizer = type('tok', (), {'decode': lambda self, ids: f"T{ids[0]}"})()
        self._vocab = vocab

    def to_tokens(self, s: str):
        return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    def to_str_tokens(self, s: str):
        return ["A", "B", "C", "D"]

    def __call__(self, tokens):
        # Fire embedding and per-layer residual hooks
        import torch
        b, seq = tokens.shape
        d = self.cfg.d_model
        embed = torch.randn(b, seq, d)
        self.hook_dict['hook_embed'].fire(embed, name='hook_embed')
        torch.manual_seed(0)
        for i, blk in enumerate(self.blocks):
            resid = torch.randn(b, seq, d)
            blk.hook_resid_post.fire(resid, name=f'blocks.{i}.hook_resid_post')
        logits = torch.randn(b, seq, self._vocab)
        return logits


def test_prism_placement_failure_path():
    # Monkey-replace PrismLensAdapter with failing stub
    orig_cls = passes_mod.PrismLensAdapter
    passes_mod.PrismLensAdapter = _FakePrismLensAdapter
    try:
        model = _ModelStub()
        norm_lens = NormLensAdapter()
        W_U = torch.randn(model.cfg.d_model, 11, dtype=torch.float32)
        b_U = torch.randn(11, dtype=torch.float32)
        mm_cache = {}
        window_mgr = WindowManager(1, extra_window_ks=COPY_SOFT_WINDOW_KS)
        json_data = {"records": [], "pure_next_token_records": [], "raw_lens_check": {"mode": "off", "samples": [], "summary": None}}
        json_data_prism = {"records": [], "pure_next_token_records": []}
        stats = WhitenStats(mean=torch.zeros(model.cfg.d_model), var=torch.ones(model.cfg.d_model), eps=1e-8)
        Q = torch.eye(model.cfg.d_model, dtype=torch.float32)
        unembed_ctx = UnembedContext(W=W_U, b=b_U, force_fp32=True, cache=mm_cache)
        prism_ctx = PrismContext(stats=stats, Q=Q, active=True)

        summary, last_consistency, arch, diag = run_prompt_pass(
            model=model,
            context_prompt="dummy",
            ground_truth="Berlin",
            prompt_id="pos",
            prompt_variant="orig",
            window_manager=window_mgr,
            norm_lens=norm_lens,
            unembed_ctx=unembed_ctx,
            copy_threshold=0.95,
            copy_margin=0.10,
            entropy_collapse_threshold=1.0,
            top_k_record=5,
            top_k_verbose=20,
            keep_residuals=False,
            out_dir=None,
            RAW_LENS_MODE='off',
            json_data=json_data,
            json_data_prism=json_data_prism,
            prism_ctx=prism_ctx,
            decode_id_fn=_decode_id,
            ctx_ids_list=[1,2,3,4],
            first_ans_token_id=None,
            important_words=["Germany", "Berlin"],
            head_scale_cfg=None,
            head_softcap_cfg=None,
            copy_soft_threshold=COPY_SOFT_THRESHOLD,
            copy_soft_window_ks=COPY_SOFT_WINDOW_KS,
            copy_strict_label=COPY_STRICT_LABEL,
            copy_soft_labels=COPY_SOFT_LABELS,
            copy_soft_extra_labels={},
        )

        # Diagnostic surfaced once, no sidecar rows
        assert isinstance(diag, dict) and "placement_error" in diag
        assert json_data_prism["records"] == []
        assert json_data_prism["pure_next_token_records"] == []
    finally:
        passes_mod.PrismLensAdapter = orig_cls


if __name__ == "__main__":
    test_prism_placement_failure_path()
