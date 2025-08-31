#!/usr/bin/env python3
"""Lightweight orchestrator smoke test (no network, CPU-only)."""

import _pathfix  # noqa: F401

import os
import tempfile
import types
import torch


class Handle:
    def __init__(self, hp):
        self.hp = hp
    def remove(self):
        self.hp._fn = None


class HookPoint:
    def __init__(self, name):
        self.name = name
        self._fn = None
    def add_hook(self, fn):
        self._fn = fn
        return Handle(self)
    def fire(self, tensor):
        if self._fn is None:
            return
        class H:
            pass
        h = H(); h.name = self.name
        self._fn(tensor, h)


class Unembed(torch.nn.Module):
    def __init__(self, d_model=4, d_vocab=8):
        super().__init__()
        self.W_U = torch.nn.Parameter(torch.randn(d_model, d_vocab, dtype=torch.float32))
        self.b_U = torch.nn.Parameter(torch.zeros(d_vocab, dtype=torch.float32))
    def forward(self, resid):
        return resid @ self.W_U + self.b_U


class Block:
    def __init__(self, idx, d_model=4):
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.mlp = f"mlp{idx}"
        self.hook_resid_post = HookPoint(f"blocks.{idx}.hook_resid_post")
    def children(self):
        return [self.ln1, self.ln2, self.mlp]


class Tokenizer:
    def __init__(self):
        self.id_to_tok = {0: "Give", 1: "the", 2: "city", 3: "name", 4: "only", 5: "Berlin", 6: "is", 7:"⟨NEXT⟩"}
    def decode(self, ids):
        return self.id_to_tok.get(ids[0] if isinstance(ids, (list, tuple)) else int(ids), str(ids))


class MockModel(torch.nn.Module):
    def __init__(self, seq_len=6, d_model=4, d_vocab=8, n_layers=2):
        super().__init__()
        class Cfg:
            pass
        self.cfg = Cfg(); self.cfg.n_layers=n_layers; self.cfg.d_model=d_model; self.cfg.n_heads=1; self.cfg.d_vocab=d_vocab; self.cfg.n_ctx=seq_len
        self.blocks = [Block(i, d_model) for i in range(n_layers)]
        self.ln_final = torch.nn.LayerNorm(d_model)
        self.hook_dict = {'hook_embed': HookPoint('hook_embed'), 'hook_pos_embed': HookPoint('hook_pos_embed')}
        self.unembed = Unembed(d_model, d_vocab)
        self.tokenizer = Tokenizer()
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_vocab = d_vocab
        self.berlin_id = 5

    def parameters(self, recurse=True):
        return [self._param]
    def to(self, device):
        return self
    def to_tokens(self, prompt):
        return torch.tensor([[0,1,2,3,4,6]], dtype=torch.long)
    def to_str_tokens(self, prompt):
        return ["Give", "the", "city", "name", "only", "is"]
    def forward(self, tokens):
        b, s = tokens.shape
        resid = torch.randn(1, s, self.d_model)
        self.hook_dict['hook_embed'].fire(resid)
        self.hook_dict['hook_pos_embed'].fire(resid*0.0)
        for i in range(self.cfg.n_layers):
            self.blocks[i].hook_resid_post.fire(resid + (i+1)*0.01)
        logits = torch.randn(1, s, self.d_vocab)
        logits[0, -1, self.berlin_id] = 10.0
        return logits


def test_orchestrator_smoke_writes_outputs():
    import importlib.util
    run_path = os.path.join(os.path.dirname(__file__), '..', 'run.py')
    spec = importlib.util.spec_from_file_location('run_mod', run_path)
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)

    run.HookedTransformer.from_pretrained_no_processing = lambda *a, **k: MockModel()

    from layers_core import ExperimentConfig

    with tempfile.TemporaryDirectory() as td:
        meta = os.path.join(td, 'out.json')
        recs = os.path.join(td, 'out-records.csv')
        pure = os.path.join(td, 'out-pure.csv')
        cfg = ExperimentConfig(device='cpu', fp32_unembed=False, keep_residuals=False,
                               copy_threshold=0.5, copy_margin=0.1, out_dir=td, self_test=False)
        data = run.run_experiment_for_model('mock/Model', (meta, recs, pure), cfg)

        assert os.path.exists(meta)
        assert os.path.exists(recs)
        assert os.path.exists(pure)
        assert 'model_stats' in data and 'diagnostics' in data and 'final_prediction' in data
        assert 'ablation_summary' in data
