#!/usr/bin/env python3
"""Minimal CPU-only test for run_prompt_pass.

Uses a simple model stub that exposes the hooks and shapes expected by the
pass runner and verifies that records and pure-next-token entries are emitted.
"""

import _pathfix  # noqa: F401

import torch
import torch.nn as nn

from layers_core.passes import run_prompt_pass
from layers_core.lenses import NormLensAdapter
from layers_core.windows import WindowManager


class _Hookable:
    def __init__(self, name: str):
        self.name = name
        self._hooks = []

    def add_hook(self, fn):
        self._hooks.append(fn)
        return _Handle(fn, self._hooks)

    def fire(self, tensor):
        for fn in list(self._hooks):
            class H:  # minimal hook object
                def __init__(self, name):
                    self.name = name
            fn(tensor, H(self.name))


class _Handle:
    def __init__(self, fn, container):
        self._fn = fn
        self._container = container

    def remove(self):
        try:
            self._container.remove(self._fn)
        except ValueError:
            pass


class _Block(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model))
        self.hook_resid_post = _Hookable('blocks.X.hook_resid_post')


class _ModelStub(nn.Module):
    def __init__(self, d_model=8, n_layers=2, vocab=11):
        super().__init__()
        self.cfg = type('cfg', (), {'n_layers': n_layers, 'd_model': d_model, 'model_name': 'stub'})
        self.blocks = nn.ModuleList([_Block(d_model) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.hook_dict = {
            'hook_embed': _Hookable('hook_embed'),
            # omit pos embed to simulate rotary; pass runner handles both paths
        }
        self.tokenizer = type('tok', (), {'decode': lambda self, ids: f"T{ids[0]}"})()
        self._vocab = vocab

    def to_tokens(self, s: str):
        # Very simple tokenization: 4 tokens
        return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    def to_str_tokens(self, s: str):
        return ["A", "B", "C", "D"]

    def forward(self, tokens):
        b, seq = tokens.shape
        d = self.cfg.d_model
        # Fire embedding hook
        embed = torch.randn(b, seq, d)
        self.hook_dict['hook_embed'].fire(embed)
        # Fire per-layer resid_post hooks with deterministic randoms
        torch.manual_seed(0)
        for i, blk in enumerate(self.blocks):
            resid = torch.randn(b, seq, d)
            blk.hook_resid_post.fire(resid)
        # Return logits: simple linear projection of last residual
        logits = torch.randn(b, seq, self._vocab)
        return logits


def _decode_id(idx):
    if hasattr(idx, 'item'):
        idx = int(idx.item())
    return f"T{idx}"


def test_run_prompt_pass_minimal():
    model = _ModelStub()
    norm_lens = NormLensAdapter()
    W_U = torch.randn(model.cfg.d_model, 11, dtype=torch.float32)
    b_U = torch.randn(11, dtype=torch.float32)
    mm_cache = {}
    window_mgr = WindowManager(1)
    json_data = {"records": [], "pure_next_token_records": [], "raw_lens_check": {"mode": "off", "samples": [], "summary": None}}
    json_data_prism = {"records": [], "pure_next_token_records": []}

    summary, last_consistency, arch, _diag = run_prompt_pass(
        model=model,
        context_prompt="dummy",
        ground_truth="Berlin",
        prompt_id="pos",
        prompt_variant="orig",
        window_manager=window_mgr,
        norm_lens=norm_lens,
        analysis_W_U=W_U,
        analysis_b_U=b_U,
        force_fp32_unembed=True,
        mm_cache=mm_cache,
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
        prism_active=False,
        prism_stats=None,
        prism_Q=None,
        decode_id_fn=_decode_id,
        ctx_ids_list=[1,2,3,4],
        first_ans_token_id=None,
        important_words=["Germany", "Berlin"],
        head_scale_cfg=None,
        head_softcap_cfg=None,
    )

    # Basic assertions: at least L0 rows exist in both records and pure-next-token
    assert any(rec.get("layer") == 0 for rec in json_data["records"])  # per-position L0 rows
    assert any(rec.get("layer") == 0 for rec in json_data["pure_next_token_records"])  # L0 pure-next-token
    # At least one post-block layer should emit rows (layer>=1)
    assert any(rec.get("layer", -1) >= 1 for rec in json_data["records"])  # per-position
    assert any(rec.get("layer", -1) >= 1 for rec in json_data["pure_next_token_records"])  # pure-next-token
    assert isinstance(summary, dict) and "L_copy" in summary and "L_semantic" in summary
    assert arch in ("pre_norm", "post_norm", "unknown")
    # last consistency may be None in stub; permissive
