#!/usr/bin/env python3
"""Sidecar CSV smoke test with mock model and synthetic Prism artifacts.

This mirrors the structure of test_orchestrator_smoke but verifies that when
compatible Prism artifacts are present and prism=on, the run writes
`-records-prism.csv` and `-pure-next-token-prism.csv` alongside baseline CSVs.
"""

import _pathfix  # noqa: F401

import importlib.util
import os
import sys
import tempfile
import types
import torch

from layers_core.prism import WhitenStats, save_prism_artifacts


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
        class H: pass
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
        class Cfg: pass
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


def test_prism_sidecar_writes_outputs():
    # Provide a minimal stub for transformer_lens to allow importing run.py
    tl = types.SimpleNamespace(HookedTransformer=types.SimpleNamespace(from_pretrained_no_processing=lambda *a, **k: MockModel()))
    sys.modules.setdefault('transformer_lens', tl)

    # Import run.py as a module
    run_path = os.path.join(os.path.dirname(__file__), '..', 'run.py')
    spec = importlib.util.spec_from_file_location('run_mod_prism', run_path)
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)

    # Prepare temp artifacts and outputs
    with tempfile.TemporaryDirectory() as td_out:
        # Create prism artifacts under an absolute temp directory (no repo pollution)
        with tempfile.TemporaryDirectory() as td_prism:
            prism_root = td_prism  # absolute path; run.py join() will respect abs path
            model_clean = 'Model'
            art_dir = os.path.join(prism_root, model_clean)
            os.makedirs(art_dir, exist_ok=True)

            # Save simple identity Q and unit whiten stats for d_model=4
            stats = WhitenStats(mean=torch.zeros(4), var=torch.ones(4), eps=1e-8)
            Q = torch.eye(4, dtype=torch.float32)
            prov = {"method": "procrustes", "k": 4, "layers": ["embed", 0, 1], "seed": 316}
            save_prism_artifacts(art_dir, stats=stats, Q=Q, provenance=prov)

            # Force prism=on and set prism_dir to our temp absolute directory
            old_prism = getattr(run.CLI_ARGS, 'prism', 'auto')
            old_pdir = getattr(run.CLI_ARGS, 'prism_dir', 'prisms')
            run.CLI_ARGS.prism = 'on'
            run.CLI_ARGS.prism_dir = prism_root

            # Run once using the mock model
            meta = os.path.join(td_out, 'out.json')
            recs = os.path.join(td_out, 'out-records.csv')
            pure = os.path.join(td_out, 'out-pure.csv')
            cfg = run.ExperimentConfig(device='cpu', fp32_unembed=False, keep_residuals=False,
                                       copy_threshold=0.5, copy_margin=0.1, out_dir=td_out, self_test=False)
            data = run.run_experiment_for_model('mock/Model', (meta, recs, pure), cfg)

            # Sidecar CSVs should be present
            recs_prism = os.path.join(td_out, 'output-Model-records-prism.csv')
            pure_prism = os.path.join(td_out, 'output-Model-pure-next-token-prism.csv')
            assert os.path.exists(recs_prism), f"missing records sidecar: {recs_prism}"
            assert os.path.exists(pure_prism), f"missing pure sidecar: {pure_prism}"
            # Print file sizes for quick visibility
            print(f"[DEBUG prism-sidecar] records_prism bytes={os.path.getsize(recs_prism)} pure_prism bytes={os.path.getsize(pure_prism)}")

            # Restore CLI args (best-effort) before temp dir cleanup
            run.CLI_ARGS.prism = old_prism
            run.CLI_ARGS.prism_dir = old_pdir


if __name__ == "__main__":
    import traceback
    print("Running prism sidecar smoke test…")
    try:
        test_prism_sidecar_writes_outputs(); print("✅ prism sidecar files exist")
        raise SystemExit(0)
    except AssertionError as e:
        print("❌ assertion failed:", e); traceback.print_exc(); raise SystemExit(1)
    except Exception as e:
        print("❌ test crashed:", e); traceback.print_exc(); raise SystemExit(1)
