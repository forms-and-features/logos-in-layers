#!/usr/bin/env python3
"""Fit identity-plus-low-rank tuned lens translators for a single model.

The defaults (dataset, batching, optimisation schedule) follow the specification
in ``001_layers_baseline/TUNED_LENS_PLAN.md``. The script intentionally avoids a
large CLI surface; callers supply only ``--model-id`` and everything else is
derived from project constants.

Training outline
----------------

1. Stream text from ``HuggingFaceFW/fineweb-edu`` (subset ``CC-MAIN-2025-26``),
   tokenize with the model tokenizer, and pack into contiguous sequences of
   length 512.
2. For each optimiser step (975 total) sample a handful of post-block layers
   (default 8 without replacement). Use the cached post-block residuals at eight
   positions drawn uniformly from the last 60–95% of the context. The model
   forward runs under ``no_grad`` – only the translator (and per-layer
   temperature scalars) receive gradients.
3. Decode the translated residuals with the model's fp32 shadow unembedding and
   minimise the cross-entropy to the model's final distribution. Apply small
   L2 and depth-smoothness regularisation (weights = 1e-4).
4. Save the resulting translator weights, optional preconditioner snapshot, and
   provenance metadata under ``001_layers_baseline/tuned_lenses/<clean_name>/``.

The script keeps bookkeeping minimal yet explicit: it records dataset
parameters, random seeds, the derived training schedule, and rolling averages of
the training loss. Validation hooks are stubbed for now and can be expanded in
a follow-up iteration (see TODO notes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformer_lens import HookedTransformer

from layers_core.device_policy import choose_dtype, should_auto_promote_unembed, select_best_device
from layers_core.hooks import attach_residual_hooks, detach_hooks, build_cache_hook
from layers_core.norm_utils import get_correct_norm_module, apply_norm_or_skip
from layers_core.prism import load_prism_artifacts
from layers_core.tuned_lens import (
    TunedTranslator,
    Preconditioner,
    save_tuned_lens,
    clip_rank,
)
from layers_core.unembed import prepare_unembed_weights
from layers_core.numerics import safe_cast_for_unembed
from layers_core.unembed import unembed_mm
from models import CANDIDATE_MODELS


DATASET_REPO = "HuggingFaceFW/fineweb-edu"
DATASET_NAME = "CC-MAIN-2025-26"
DATASET_SPLIT = "train"
DATASET_REVISION = "v1.4.0"

SEQ_LEN = 512
POSITIONS_PER_SEQ = 16
POSITION_RANGE = (0.60, 0.95)  # relative slice of the context

MICRO_BATCH = 8
GRAD_ACCUM = 4
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM

LAYERS_PER_STEP = 24

TOTAL_TOKENS = 32_000_000
TOKENS_PER_STEP = SEQ_LEN * EFFECTIVE_BATCH
TOTAL_STEPS = math.ceil(TOTAL_TOKENS / TOKENS_PER_STEP)
WARMUP_STEPS = math.ceil(0.1 * TOTAL_STEPS)

LR = 1e-3
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1e-4
REG_L2 = 1e-4
REG_SMOOTH = 1e-4

SEED = 316


def set_deterministic_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingConfig:
    model_id: str
    clean_model_name: str
    device: torch.device
    dtype: torch.dtype
    rank: Optional[int]
    out_dir: Path
    use_prism: bool
    cache_dir: Optional[str]
    prefetch_dataset: bool


class PackedSequenceStreamer:
    """Stream contiguous token sequences of fixed length from the dataset."""

    def __init__(
        self,
        tokenizer,
        *,
        seq_len: int,
        repo: str,
        name: str,
        split: str,
        revision: str,
        cache_dir: Optional[str] = None,
        prefetch: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        if prefetch:
            print("[tuned-lens] Prefetching dataset shards locally (this may take a while)…")
            load_dataset(
                repo,
                name=name,
                split=split,
                revision=revision,
                streaming=False,
                cache_dir=cache_dir,
            )
        self.dataset = load_dataset(
            repo,
            name=name,
            split=split,
            revision=revision,
            streaming=True,
            cache_dir=cache_dir,
        )
        self.buffer: List[int] = []
        self.hasher = hashlib.sha256()
        self.samples_seen = 0

    def __iter__(self) -> Iterator[torch.Tensor]:
        for sample in self.dataset:
            text = sample.get("text", "")
            if not isinstance(text, str):
                continue
            self.hasher.update(text.encode("utf-8", errors="ignore"))
            token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if not token_ids:
                continue
            self.buffer.extend(token_ids)
            self.samples_seen += 1
            while len(self.buffer) >= self.seq_len:
                chunk = self.buffer[: self.seq_len]
                del self.buffer[: self.seq_len]
                yield torch.tensor(chunk, dtype=torch.long)


def clean_model_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def infer_device(model_id: str, device_flag: Optional[str]) -> Tuple[torch.device, torch.dtype, Dict[str, str]]:
    if device_flag and device_flag != "auto":
        dev = torch.device(device_flag)
        dtype = choose_dtype(device_flag, model_id)
        return dev, dtype, {"mode": "forced", "device": str(dev), "dtype": str(dtype)}

    selection = select_best_device(model_id)
    if selection is None:
        # fallback to CPU float32
        dev = torch.device("cpu")
        dtype = choose_dtype("cpu", model_id)
        return dev, dtype, {"mode": "fallback", "device": "cpu", "dtype": str(dtype)}
    device_str, dtype, debug = selection
    return torch.device(device_str), dtype, {"mode": "auto", **{k: str(v) for k, v in debug.items()}}


def maybe_load_preconditioner(model_clean: str) -> Optional[Preconditioner]:
    prism_dir = Path(__file__).with_name("prisms") / model_clean
    if not prism_dir.exists():
        return None
    try:
        stats, Q, _prov = load_prism_artifacts(str(prism_dir))
        if stats is None:
            return None
        mean = stats.mean.float()
        var = stats.var.float()
        eps = float(stats.eps)
        inv_scale = 1.0 / torch.sqrt(var + eps)
        rotation = None if Q is None else Q.float()
        return Preconditioner(mean=mean, inv_scale=inv_scale, rotation=rotation)
    except FileNotFoundError:
        return None


def sample_positions(seq_len: int, count: int, low_frac: float, high_frac: float) -> torch.Tensor:
    low = max(0, int(math.floor(seq_len * low_frac)))
    high = max(low + 1, int(math.ceil(seq_len * high_frac)))
    high = min(seq_len, high)
    if high <= low:
        high = seq_len
    positions = torch.randint(low, high, (count,), dtype=torch.long)
    return positions


def compute_ce(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    teacher_probs = torch.softmax(teacher_logits, dim=-1)
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    ce = -(teacher_probs * student_log_probs).sum(dim=-1)
    return ce.mean()


def regularization_terms(translator: TunedTranslator, layer_idx: int) -> torch.Tensor:
    layer = translator.layers[layer_idx]
    loss = torch.tensor(0.0, device=layer.c.device if layer.c is not None else translator.layers[0].parameters().__next__().device)

    if layer.rank > 0:
        loss = loss + REG_L2 * (layer.U.pow(2).sum() + layer.V.pow(2).sum())
    if layer.bias and layer.c is not None:
        loss = loss + REG_L2 * layer.c.pow(2).sum()

    # Depth smoothness with detached neighbours so only current layer receives gradients
    if layer.rank > 0:
        if layer_idx > 0:
            neighbour = translator.layers[layer_idx - 1]
            if neighbour.rank > 0:
                loss = loss + REG_SMOOTH * (layer.U - neighbour.U.detach()).pow(2).sum()
                loss = loss + REG_SMOOTH * (layer.V - neighbour.V.detach()).pow(2).sum()
        if layer_idx + 1 < translator.num_layers:
            neighbour = translator.layers[layer_idx + 1]
            if neighbour.rank > 0:
                loss = loss + REG_SMOOTH * (layer.U - neighbour.U.detach()).pow(2).sum()
                loss = loss + REG_SMOOTH * (layer.V - neighbour.V.detach()).pow(2).sum()

    return loss


def microbatch_iterator(streamer: PackedSequenceStreamer, batch_size: int) -> Iterator[torch.Tensor]:
    batch: List[torch.Tensor] = []
    for seq in streamer:
        batch.append(seq)
        if len(batch) == batch_size:
            yield torch.stack(batch, dim=0)
            batch.clear()


def build_model(model_id: str, device: torch.device, dtype: torch.dtype) -> HookedTransformer:
    os.environ.setdefault("TRANSFORMERS_BATCH_FIRST", "False")
    model = HookedTransformer.from_pretrained_no_processing(
        model_id,
        device=str(device),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return model


def train_translator(cfg: TrainingConfig) -> Dict[str, float]:
    device = cfg.device
    dtype = cfg.dtype

    # Ensure parallel tokenization stays enabled even if the process forks later
    # (datasets prefetch/generation). This avoids the library's safety auto-disable.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    # Optionally prefetch shards before constructing the tokenizer/model to avoid
    # the "fork after tokenizers" warning and keep tokenizers parallelism on.
    if cfg.prefetch_dataset:
        print("[tuned-lens] Prefetching dataset shards locally (this may take a while)…")
        _ = load_dataset(
            DATASET_REPO,
            name=DATASET_NAME,
            split=DATASET_SPLIT,
            revision=DATASET_REVISION,
            streaming=False,
            cache_dir=cfg.cache_dir,
        )

    model = build_model(cfg.model_id, device, dtype)

    # --- Auto-scale per-step work for large vocabularies --------------------
    # Unembedding cost is O(V); scale down layers/positions when vocab is large
    # using a sqrt rule so 128k≈0.5× of the 32k baseline.
    try:
        d_vocab = int(getattr(model.cfg, "d_vocab", None) or model.unembed.W_U.shape[-1])
    except Exception:
        d_vocab = None
    baseline_vocab = 32_000
    sqrt_scale = 1.0
    if d_vocab and d_vocab > 0:
        sqrt_scale = min(1.0, (baseline_vocab / float(d_vocab)) ** 0.5)
    # Start from defaults and apply scaling with sensible floors
    layers_per_step = max(8, int(round(LAYERS_PER_STEP * sqrt_scale)))
    positions_per_seq = max(12, int(round(POSITIONS_PER_SEQ * sqrt_scale)))
    # Don’t exceed the configured defaults
    layers_per_step = min(layers_per_step, LAYERS_PER_STEP)
    positions_per_seq = min(positions_per_seq, POSITIONS_PER_SEQ)
    print(
        f"[tuned-lens] schedule: d_vocab={d_vocab or 'unknown'} → layers_per_step={layers_per_step}, "
        f"positions_per_seq={positions_per_seq} (baseline {LAYERS_PER_STEP}/{POSITIONS_PER_SEQ})"
    )

    rank = cfg.rank if cfg.rank is not None else clip_rank(model.cfg.d_model)
    translator = TunedTranslator(
        num_layers=model.cfg.n_layers,
        d_model=model.cfg.d_model,
        rank=rank,
        final_identity=True,
        preconditioner=maybe_load_preconditioner(cfg.clean_model_name) if cfg.use_prism else None,
        device=device,
        dtype=torch.float32,
        use_temperature=True,
    )

    auto_fp32_unembed = should_auto_promote_unembed(dtype)
    W_U, b_U = prepare_unembed_weights(
        model.unembed.W_U,
        getattr(model.unembed, "b_U", None),
        force_fp32=True if auto_fp32_unembed else False,
    )
    unembed_ctx = {
        "W": W_U.to(device=device, dtype=torch.float32),
        "b": None if b_U is None else b_U.to(device=device, dtype=torch.float32),
    }

    # Device-aware batching: on CUDA we can use larger micro-batches and reduce
    # grad accumulation to lower per-step Python/launch overhead.
    micro_batch = MICRO_BATCH
    grad_accum = GRAD_ACCUM
    if device.type == "cuda":
        micro_batch = 32
        grad_accum = 1
    effective_batch = micro_batch * grad_accum
    tokens_per_step = SEQ_LEN * effective_batch
    total_steps = math.ceil(TOTAL_TOKENS / tokens_per_step)
    warmup_steps = math.ceil(0.1 * total_steps)

    streamer = PackedSequenceStreamer(
        tokenizer=model.tokenizer,
        seq_len=SEQ_LEN,
        repo=DATASET_REPO,
        name=DATASET_NAME,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
        cache_dir=cfg.cache_dir,
        # We've prefetched already (if requested) before building the model.
        prefetch=False,
    )
    seq_iter = iter(microbatch_iterator(streamer, micro_batch))

    optimizer = torch.optim.AdamW(translator.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.0)

    translator.train()
    meter_loss: List[float] = []
    start_time = time.time()
    LOG_EVERY = 25

    def _format_eta(seconds: float) -> str:
        secs = int(max(0, round(seconds)))
        h = secs // 3600
        m = (secs % 3600) // 60
        s = secs % 60
        if h > 0:
            return f"{h}h {m:02d}m {s:02d}s"
        if m > 0:
            return f"{m}m {s:02d}s"
        return f"{s}s"

    for step_idx in range(total_steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss_values: List[float] = []
        for accum_idx in range(GRAD_ACCUM):
            try:
                batch_tokens = next(seq_iter)
            except StopIteration:
                seq_iter = iter(microbatch_iterator(streamer, micro_batch))
                batch_tokens = next(seq_iter)

            if device.type == "cuda":
                batch_tokens = batch_tokens.pin_memory().to(device=device, non_blocking=True)
            else:
                batch_tokens = batch_tokens.to(device=device)

            # Attach residual hooks
            residual_cache: Dict[str, torch.Tensor] = {}
            cache_hook = build_cache_hook(residual_cache)
            handles, _ = attach_residual_hooks(model, cache_hook)
            with torch.no_grad():
                logits = model(batch_tokens)
            detach_hooks(handles)

            total_layers = translator.num_layers
            trainable_layers = total_layers - 1 if translator.final_identity else total_layers
            if trainable_layers <= 0:
                raise RuntimeError("Translator has no trainable layers")
            candidate_layers = list(range(trainable_layers))
            sample_k = min(layers_per_step, len(candidate_layers))
            sampled_layers = random.sample(candidate_layers, sample_k)

            positions = sample_positions(SEQ_LEN, positions_per_seq, *POSITION_RANGE).to(device=device)
            batch_indices = torch.arange(batch_tokens.shape[0], device=device).unsqueeze(1).repeat(1, positions_per_seq)
            pos_indices = positions.unsqueeze(0).repeat(batch_tokens.shape[0], 1)

            loss_acc = None
            for layer_idx in sampled_layers:
                resid_key = f"blocks.{layer_idx}.hook_resid_post"
                resid = residual_cache.pop(resid_key).to(device=device)
                norm_module = get_correct_norm_module(model, layer_idx, probe_after_block=True)
                resid_norm = apply_norm_or_skip(resid, norm_module)

                resid_selected = resid_norm[batch_indices, pos_indices, :]
                resid_flat = resid_selected.reshape(-1, resid_selected.shape[-1])

                translated = translator(resid_flat, layer_idx)
                logits_student = unembed_mm(
                    safe_cast_for_unembed(translated, unembed_ctx["W"], force_fp32_unembed=True),
                    unembed_ctx["W"],
                    unembed_ctx["b"],
                )
                logits_student = logits_student.reshape(batch_tokens.shape[0], positions_per_seq, -1)
                tau = translator.temperature(layer_idx)
                logits_student = logits_student / tau.to(device=device, dtype=logits_student.dtype)

                teacher = logits[batch_indices, pos_indices, :].reshape(batch_tokens.shape[0], positions_per_seq, -1)
                ce = compute_ce(logits_student.reshape(-1, logits_student.shape[-1]), teacher.reshape(-1, teacher.shape[-1]))

                reg = regularization_terms(translator, layer_idx)
                loss_layer = ce + reg
                loss_acc = loss_layer if loss_acc is None else loss_acc + loss_layer

            residual_cache.clear()

            loss = loss_acc / sample_k
            loss.backward()
            step_loss_values.append(float(loss.detach().cpu()))

        optimizer.step()
        scheduler.step()

        mean_step_loss = float(np.mean(step_loss_values)) if step_loss_values else float('nan')
        meter_loss.append(mean_step_loss)

        recent = meter_loss[-LOG_EVERY:] if len(meter_loss) >= LOG_EVERY else meter_loss
        avg_recent_loss = float(np.mean(recent)) if recent else float('nan')

        if (step_idx % LOG_EVERY == 0) or (step_idx == TOTAL_STEPS - 1):
            elapsed = max(time.time() - start_time, 1e-6)
            steps_done = step_idx + 1
            tokens_done = steps_done * tokens_per_step
            tok_per_sec = tokens_done / elapsed
            remaining_steps = total_steps - steps_done
            eta_seconds = remaining_steps * (elapsed / steps_done)
            print(
                f"[tuned-lens] step {steps_done}/{total_steps} "
                f"loss={avg_recent_loss:.4f} "
                f"tokens={tokens_done/1e6:.2f}M "
                f"throughput={tok_per_sec:,.0f} tok/s "
                f"ETA={_format_eta(eta_seconds)}"
            )

    translator.eval()
    translator_cpu = translator.to("cpu")
    avg_loss = float(np.mean(meter_loss[-min(len(meter_loss), 100):])) if meter_loss else float("nan")

    # Save artifacts
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    tl_version = datetime.utcnow().strftime("%Y%m%dT%H%MZ")

    provenance = {
        "model_id": cfg.model_id,
        "clean_model_name": cfg.clean_model_name,
        "tl_version": tl_version,
        "model_commit": None,
        "tokenizer_commit": None,
        "translator": translator_cpu.metadata(),
        "training": {
            "seed": SEED,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "micro_batch": micro_batch,
            "grad_accum": grad_accum,
            "effective_batch": effective_batch,
            "tokens_per_step": tokens_per_step,
            "layers_sampled_per_step": layers_per_step,
            "use_temperature": translator_cpu.use_temperature,
            "temperature_eps": translator_cpu.temperature_eps if translator_cpu.use_temperature else None,
            "dataset": {
                "repo_id": DATASET_REPO,
                "name": DATASET_NAME,
                "split": DATASET_SPLIT,
                "revision": DATASET_REVISION,
                "seq_len": SEQ_LEN,
                "positions_per_seq": positions_per_seq,
                "position_fraction_range": POSITION_RANGE,
                "content_hash": streamer.hasher.hexdigest(),
                "samples_streamed": streamer.samples_seen,
            },
            "optimizer": {
                "type": "AdamW",
                "lr": LR,
                "betas": BETAS,
                "weight_decay": WEIGHT_DECAY,
            },
            "regularization": {
                "l2": REG_L2,
                "smooth": REG_SMOOTH,
            },
            "device": str(device),
            "dtype": str(dtype),
            "final_loss": avg_loss,
        },
    }

    save_tuned_lens(cfg.out_dir, translator=translator_cpu, provenance=provenance)

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {"train_loss": avg_loss}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tuned lens translators")
    parser.add_argument(
        "--model-id",
        dest="model_ids",
        nargs="+",
        help="One or more model identifiers (Hugging Face repos).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train tuned lenses for every model listed in models.CANDIDATE_MODELS.",
    )
    parser.add_argument("--device", default="auto", help="Device to use (auto|cpu|mps|cuda)")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional override for output directory. Defaults to tuned_lenses/<clean_model_name>",
    )
    parser.add_argument("--disable-prism", action="store_true", help="Disable Prism preconditioning even if artifacts exist")
    parser.add_argument("--cache-dir", default=None, help="Optional datasets cache directory")
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Download dataset shards before streaming (requires disk space)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    set_deterministic_seed(SEED)

    if args.all:
        model_ids = list(CANDIDATE_MODELS)
    elif args.model_ids:
        model_ids = args.model_ids
    else:
        raise SystemExit("❌ Provide --model-id <repo> or --all to train tuned lenses.")

    total = len(model_ids)
    for idx, model_id in enumerate(model_ids, start=1):
        clean_name = clean_model_name(model_id)
        out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).with_name("tuned_lenses") / clean_name
        device, dtype, _ = infer_device(model_id, args.device)

        print(f"=== Tuned lens training [{idx}/{total}] → {model_id} on {device} ===")
        if out_dir.exists() and (out_dir / "weights.pt").exists():
            print(f"[tuned-lens] Skipping {model_id}: existing artifacts found at {out_dir} (delete to retrain)")
            continue
        cfg = TrainingConfig(
            model_id=model_id,
            clean_model_name=clean_name,
            device=device,
            dtype=dtype,
            rank=None,
            out_dir=out_dir,
            use_prism=not args.disable_prism,
            cache_dir=args.cache_dir,
            prefetch_dataset=args.prefetch,
        )

        metrics = train_translator(cfg)
        print(json.dumps({"model": model_id, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise
