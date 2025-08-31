from transformer_lens import HookedTransformer
import torch
import torch.nn as nn
from datetime import datetime
import os
import subprocess
import sys
import json
import argparse
import gc  # For garbage collection

# --- deterministic bootstrap -------------------------------------------------
import random, numpy as np

SEED = 316
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.use_deterministic_algorithms(True)   # PyTorch 2.x+
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # harmless on CPU, required for CUDA
torch.set_num_threads(1)  # optional; comment out if you need full CPU speed
# -----------------------------------------------------------------------------

# Top-k settings for record emission
TOP_K_RECORD = 5    # number of tokens to record for non-verbose slots
TOP_K_VERBOSE = 20  # number of tokens to record for verbose slots and answer position

# Layer-by-layer prediction analysis with LayerNorm lens correction
# Toggle USE_NORM_LENS for raw vs normalized residual stream analysis

from models import CANDIDATE_MODELS

# Backward-compatible name used throughout this script
CONFIRMED_MODELS = CANDIDATE_MODELS

# --- helpers (extracted to norm_utils) --------------------------------------
from layers_core.norm_utils import (
    _get_rms_scale,
    apply_norm_or_skip,
    detect_model_architecture,
    get_correct_norm_module,
)
from layers_core.numerics import (
    bits_entropy_from_logits,
    safe_cast_for_unembed,
    kl_bits,
)
from layers_core.metrics import compute_next_token_metrics
from layers_core.csv_io import write_csv_files
from layers_core.collapse_rules import (
    is_semantic_top1,
    detect_copy_collapse_id_subseq,
    is_pure_whitespace_or_punct,
)
from layers_core.device_policy import (
    choose_dtype,
    should_auto_promote_unembed,
    select_best_device,
)
from layers_core.hooks import build_cache_hook, attach_residual_hooks, detach_hooks
from layers_core.run_dir import setup_run_latest_directory
from layers_core.config import ExperimentConfig
from layers_core.raw_lens import (
    get_raw_lens_mode,
    init_raw_lens_check,
    should_sample_layer,
    record_dual_lens_sample,
    summarize_raw_lens_check,
)
from layers_core.summaries import summarize_pure_records
from layers_core.gold import compute_gold_answer_info, compute_gold_answer_info_from_sequences
from layers_core.prism import load_prism_artifacts, whiten_apply

def clean_model_name(model_id):
    """Extract clean model name for filename"""
    # Remove organization prefix (everything before last '/')
    clean_name = model_id.split('/')[-1]
    return clean_name

 


def run_experiment_for_model(model_id, output_files, config: ExperimentConfig):
    """Run the complete experiment for a single model and write results to files"""
    
    def evaluate_model():
        """The actual experiment code - all prints go to console"""
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL: {model_id}")
        print(f"{'='*60}")
        
        # Variable to store detected architecture
        detected_architecture = None
        
        # ---- device & dtype ---------------------------------------------------
        device = config.device
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available; falling back to CPU.")
            device = "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            print("⚠️  MPS requested but not available; falling back to CPU.")
            device = "cpu"

        dtype = choose_dtype(device, model_id)

        # ---- load model -------------------------------------------------------
        print(f"Loading model on [{device}] ...")
        
        # Clear any existing CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Set batch_first=False for models that need it (TransformerLens expectation)
        os.environ['TRANSFORMERS_BATCH_FIRST'] = 'False'
            
        # Load model
        try:
            print("Loading directly to target device…")
            if device == "cpu":
                # Explicitly keep all weights on CPU – avoids Accelerate placing them on MPS
                model = HookedTransformer.from_pretrained_no_processing(
                    model_id,
                    device="cpu",
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
            else:
                model = HookedTransformer.from_pretrained_no_processing(
                    model_id,
                    device=device,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
        except Exception as e:
            print(f"Direct loading to {device} failed: {e}")
            print("Falling back to CPU loading...")
            # Fallback: load on CPU then move
            model = HookedTransformer.from_pretrained_no_processing(
                model_id,
                device="cpu",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            # Clear cache before moving
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Moving model to {device}...")
            model = model.to(device)
            
        model.eval()  # Hygiene: avoid dropout etc.
        
        # Run KL sanity test if requested
        if config.self_test:
            try:
                from kl_sanity_test import run_kl_sanity_test
                # Prefer the model's own tokenizer when available
                tokenizer = getattr(model, 'tokenizer', None)
                if not hasattr(model, 'lm_head'):
                    print("ℹ️ Skipping KL sanity test: requires HF model interface (lm_head/hidden_states). Use kl_sanity_test.py standalone if needed.")
                else:
                    test_passed = run_kl_sanity_test(model, tokenizer)
                    if not test_passed:
                        print("❌ Self-test failed - normalization scaling is incorrect!")
                        print("❌ ABORTING: Cannot trust analysis results with incorrect scaling!")
                        return {"error": "Self-test failed"}
                    print("✅ Self-test passed - continuing with normal evaluation...\n")
            except ImportError as e:
                print(f"❌ Could not import KL sanity test: {e}")
                return {"error": "Self-test import failed"}
        
        # Clear cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Debug: inspect model parameter devices
        unique_param_devices = {p.device for p in model.parameters()}
        print(f"[DEBUG MODEL] Unique parameter devices: {unique_param_devices}")
        
        # Toggle for using normalized lens (recommended for accurate interpretation)
        USE_NORM_LENS = True
        
        # Toggle for FP32 unembedding (recommended for research-grade precision)
        # Prevents under-resolving logit gaps < 1e-5 with minimal memory overhead.
        USE_FP32_UNEMBED = should_auto_promote_unembed(dtype)

        # Shadow copies for analysis-only unembedding (do NOT mutate model params)
        analysis_W_U = model.unembed.W_U  # default: use model parameter dtype
        analysis_b_U = getattr(model.unembed, 'b_U', None)
        UNEMBED_DTYPE = analysis_W_U.dtype  # initial report

        # If enabled, create FP32 shadow weights for unembedding without touching the model's forward path
        if USE_FP32_UNEMBED and analysis_W_U.dtype != torch.float32:
            print(f"🔬 Using FP32 shadow unembed weights for analysis (was {analysis_W_U.dtype})")
            analysis_W_U = analysis_W_U.float()
            if analysis_b_U is not None:
                analysis_b_U = analysis_b_U.float()
            UNEMBED_DTYPE = torch.float32

        # Apply CLI-based FP32 unembed promotion if requested (analysis-only)
        if config.fp32_unembed and UNEMBED_DTYPE != torch.float32:
            print(f"🔬 CLI: Forcing FP32 shadow unembed weights for analysis (was {UNEMBED_DTYPE})")
            analysis_W_U = analysis_W_U.float()
            if analysis_b_U is not None:
                analysis_b_U = analysis_b_U.float()
            UNEMBED_DTYPE = torch.float32

        # Helper: unembed matmul that matches device of inputs (avoids CPU/MPS mismatch)
        def _unembed_mm(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
            W_use = W
            if hasattr(W_use, 'device') and X.device != W_use.device:
                W_use = W_use.to(X.device)
            out = X @ W_use
            if b is not None:
                b_use = b
                if hasattr(b_use, 'device') and X.device != b_use.device:
                    b_use = b_use.to(X.device)
                out = out + b_use
            return out
        
        context_prompt = "Give the city name only, plain text. The capital of Germany is called simply"
        ground_truth = "Berlin"  # For display/comparison
        # Stylistic filler ablation (PROJECT_NOTES §1.9): drop the adverb
        context_prompt_nf = "Give the city name only, plain text. The capital of Germany is called"
        # Negative control (PROJECT_NOTES §1.8)
        context_prompt_ctl = "Give the city name only, plain text. The capital of France is called simply"
        control_ground_truth = "Paris"
        
        first_block_ln1_type = type(model.blocks[0].ln1).__name__ if hasattr(model, 'blocks') and len(model.blocks) > 0 else None
        final_ln_type = type(model.ln_final).__name__ if hasattr(model, 'ln_final') else None
        
        # Check if LayerNorm bias fix will be applied
        uses_layernorm = (hasattr(model, 'blocks') and len(model.blocks) > 0 and 
                         isinstance(model.blocks[0].ln1, nn.LayerNorm))
        layernorm_bias_fix = "active" if uses_layernorm else "not_needed_rms_model"
        
        # Check normalization alignment for post-block residuals
        has_ln2 = (hasattr(model, 'blocks') and len(model.blocks) > 0 and 
                   hasattr(model.blocks[0], 'ln2'))
        if has_ln2:
            ln2_type = type(model.blocks[0].ln2).__name__
            if isinstance(model.blocks[0].ln2, nn.LayerNorm):
                norm_alignment_fix = "using_ln2_layernorm_for_post_block"
            else:
                norm_alignment_fix = "using_ln2_rmsnorm_for_post_block"
        else:
            norm_alignment_fix = "fallback_to_ln1"
        
        # Check layer-0 normalization approach
        layer0_norm_fix = "using_real_ln1_on_embeddings" if USE_NORM_LENS else "no_normalization"
        
        # Check mixed precision fix
        mixed_precision_fix = "casting_to_fp32_before_unembed"
        
        # Check positional embedding type for layer-0 interpretation
        has_additive_pos_embed = 'hook_pos_embed' in model.hook_dict
        layer0_position_info = "additive_pos_embed_included" if has_additive_pos_embed else "token_only_rotary_model"
        
        # Raw-vs-Norm dual-lens mode (env-controlled; default: sampled checks)
        RAW_LENS_MODE = get_raw_lens_mode(config.self_test)

        # Track a last-layer consistency snapshot (lens vs model final head)
        last_layer_consistency = None

        # Helper: detect simple final-head transforms exposed by the model/config
        def _detect_head_transforms():
            def _get_num(obj, names):
                for n in names:
                    try:
                        v = getattr(obj, n)
                    except Exception:
                        continue
                    if isinstance(v, (int, float)):
                        return float(v)
                return None

            cfg = getattr(model, 'cfg', None)
            scale = None
            softcap = None
            if cfg is not None:
                scale = _get_num(cfg, ['final_logit_scale', 'logit_scale', 'final_logits_scale']) or scale
                softcap = _get_num(cfg, ['final_logit_softcap', 'logit_softcap', 'final_logits_softcap', 'softcap']) or softcap
            scale = _get_num(model, ['final_logit_scale', 'logit_scale']) or scale
            softcap = _get_num(model, ['final_logit_softcap', 'logit_softcap']) or softcap
            return scale, softcap

        head_scale_cfg, head_softcap_cfg = _detect_head_transforms()

        diag = {
            "type": "diagnostics",
            "model": model_id,
            "device": device,
            "use_norm_lens": USE_NORM_LENS,
            "use_fp32_unembed": USE_FP32_UNEMBED,
            "unembed_dtype": str(UNEMBED_DTYPE),
            "first_block_ln1_type": first_block_ln1_type,
            "final_ln_type": final_ln_type,
            "layernorm_bias_fix": layernorm_bias_fix,
            "norm_alignment_fix": norm_alignment_fix,
            "layer0_norm_fix": layer0_norm_fix,
            "mixed_precision_fix": mixed_precision_fix,
            "layer0_position_info": layer0_position_info,
            "context_prompt": context_prompt,
            "target_prediction": "first unseen token (likely 'Berlin')"
        }
        # Prism sidecar setup (auto/on/off) and summary (filled below)
        prism_mode = getattr(CLI_ARGS, "prism", "auto")
        prism_dir_base = getattr(CLI_ARGS, "prism_dir", "prisms")
        prism_clean = clean_model_name(model_id)
        prism_art_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), prism_dir_base, prism_clean)
        prism_stats = None
        prism_Q = None
        prism_prov = None
        prism_active = False
        prism_error = None
        try:
            if prism_mode != "off":
                stats, Q, prov = load_prism_artifacts(prism_art_dir)
                d_model = int(model.cfg.d_model)
                if stats.mean.numel() != d_model or Q.shape != (d_model, d_model):
                    raise RuntimeError(f"prism artifacts incompatible with model dims (d_model={d_model})")
                prism_stats, prism_Q, prism_prov = stats, Q, prov
                prism_active = True
        except Exception as e:
            prism_error = str(e)
            if prism_mode == "on":
                raise RuntimeError(f"Prism mode=on but artifacts unavailable/incompatible: {e}")
        # Collect data for JSON output
        json_data = {
            "prompt": {"type": "prompt", "context_prompt": context_prompt, "ground_truth": ground_truth},
            "records": [],
            "pure_next_token_records": [],
            "test_prompts": [],
            "temperature_exploration": [],
            "diagnostics": None,
            "final_prediction": None,
            "model_stats": None,
            # Raw-vs-Norm sanity block (PROJECT_NOTES §1.4)
            "raw_lens_check": init_raw_lens_check(RAW_LENS_MODE),
        }
        # Initialize prism summary on diagnostics
        diag["prism_summary"] = {
            "mode": prism_mode,
            "artifact_path": prism_art_dir,
            "present": bool(prism_prov is not None),
            "compatible": bool(prism_active),
            "k": (None if prism_prov is None else prism_prov.get("k")),
            "layers": (None if prism_prov is None else prism_prov.get("layers")),
            "error": prism_error,
        }
        # Sidecar record buffers for Prism (identical schemas to baseline CSVs)
        json_data_prism = {"records": [], "pure_next_token_records": []}
        IMPORTANT_WORDS = ["Germany", "Berlin", "capital", "Answer", "word", "simply"]

        def is_verbose_position(pos, token_str, seq_len):
            # Final position (predicting next token) always verbose
            if pos == seq_len - 1:
                return True
            # Any prompt word in token string
            for w in IMPORTANT_WORDS:
                if w.lower() in token_str.lower().strip(".,!?;:"):
                    return True
            return False

        # prompt_id and prompt_variant for tagging records across passes
        current_prompt_id = "pos"
        current_prompt_variant = "orig"

        def print_summary(layer_idx, pos, token_str, entropy_bits, top_tokens, top_probs, is_pure_next_token=False, extra=None):
            # Collect record data for JSON output
            record = {
                "type": "pure_next_token_record" if is_pure_next_token else "record",
                "prompt_id": current_prompt_id,
                "prompt_variant": current_prompt_variant,
                "layer": layer_idx,
                "pos": pos,
                "token": token_str,
                "entropy": entropy_bits,
                "topk": [[tok, prob.item()] for tok, prob in zip(top_tokens, top_probs)]
            }
            if extra:
                record.update(extra)
            
            # Add to appropriate collection
            if is_pure_next_token:
                json_data["pure_next_token_records"].append(record)
            else:
                json_data["records"].append(record)

        # Helper: robust single-id decode (tensor or int)
        def decode_id(idx):
            return model.tokenizer.decode([idx.item() if hasattr(idx, 'item') else int(idx)])

        # Helper: emit pure next-token metrics and record (reduces duplication)
        def emit_pure_next_token_record(
            layer_out_idx: int,
            logits_all: torch.Tensor,
            tokens_tensor: torch.Tensor,
            ctx_ids_list,
            window_ids_list,
            final_probs_tensor: torch.Tensor,
            first_ans_token_id,
            final_dir_vec: torch.Tensor,
            collected_records: list,
            do_raw_lens_sample: bool,
            resid_raw_tensor: torch.Tensor | None,
            *,
            control_ids: tuple[int | None, int | None] | None = None,
        ):
            last_pos = tokens_tensor.shape[1] - 1
            last_logits = logits_all[last_pos]
            last_entropy_bits = bits_entropy_from_logits(last_logits)
            last_full_probs = torch.softmax(last_logits, dim=0)
            last_token_str = "⟨NEXT⟩"
            _, last_top_indices = torch.topk(last_logits, TOP_K_RECORD, largest=True, sorted=True)
            last_top_probs = last_full_probs[last_top_indices]
            last_top_tokens = [decode_id(idx) for idx in last_top_indices]

            # Update rolling window
            top1_id = last_top_indices[0].item()
            window_ids_list.append(top1_id)
            if len(window_ids_list) > config.copy_window_k:
                window_ids_list.pop(0)

            # Copy / semantic flags and metrics
            copy_collapse = detect_copy_collapse_id_subseq(
                last_logits,
                ctx_ids_list,
                window_ids_list,
                copy_threshold=config.copy_threshold,
                copy_margin=config.copy_margin,
            )
            if copy_collapse and is_pure_whitespace_or_punct(last_top_tokens[0]):
                copy_collapse = False
            entropy_collapse = last_entropy_bits <= 1.0
            # Defer final is_answer decision until rank is known; keep string fallback
            is_answer_fallback = is_semantic_top1(last_top_tokens[0], ground_truth)

            metrics = compute_next_token_metrics(
                last_full_probs, top1_id, final_probs_tensor, first_ans_token_id, topk_cum=5
            )

            # Prefer rank-based ID check when available; fallback to string match
            is_answer = (metrics.get("answer_rank") == 1) if metrics.get("answer_rank") is not None else is_answer_fallback

            # Cosine to final direction (PROJECT_NOTES §1.5)
            _curr_norm = torch.norm(last_logits) + 1e-12
            cos_to_final = torch.dot((last_logits / _curr_norm), final_dir_vec).item()

            # Control margin (PROJECT_NOTES §1.8): only meaningful for control prompt rows
            control_margin = None
            if control_ids is not None and all(x is not None for x in control_ids):
                paris_id, berlin_id = control_ids  # type: ignore
                try:
                    control_margin = float(last_full_probs[int(paris_id)]) - float(last_full_probs[int(berlin_id)])
                except Exception:
                    control_margin = None

            record_extra = {
                "copy_collapse": copy_collapse,
                "entropy_collapse": entropy_collapse,
                "is_answer": is_answer,
                **metrics,
                "cos_to_final": cos_to_final,
                "control_margin": control_margin,
            }

            # Collect for L_copy/L_semantic computation
            collected_records.append({
                "layer": layer_out_idx,
                "copy_collapse": copy_collapse,
                "entropy_collapse": entropy_collapse,
                "is_answer": is_answer,
                "kl_to_final_bits": metrics["kl_to_final_bits"],
                "answer_rank": metrics["answer_rank"],
            })

            print_summary(layer_out_idx, last_pos, last_token_str, last_entropy_bits, last_top_tokens, last_top_probs, is_pure_next_token=True, extra=record_extra)

            # Dual-lens sanity sample (PROJECT_NOTES §1.4)
            if RAW_LENS_MODE != "off" and do_raw_lens_sample and resid_raw_tensor is not None:
                record_dual_lens_sample(
                    json_data["raw_lens_check"],
                    layer_out_idx=layer_out_idx,
                    last_logits_norm=last_logits,
                    resid_raw_last_vec=resid_raw_tensor[0, last_pos, :],
                    W_U=analysis_W_U,
                    b_U=analysis_b_U,
                    force_fp32_unembed=(config.fp32_unembed or USE_FP32_UNEMBED),
                    tokenizer=model.tokenizer,
                    final_probs=final_probs_tensor,
                    first_ans_id=first_ans_token_id,
                    ground_truth=ground_truth,
                )
        
        # Tokenize the context prompt (without "Answer:" to avoid teacher-forcing)
        tokens = model.to_tokens(context_prompt)      # let Accelerate move it

        # Gold-token alignment (PROJECT_NOTES §1.7): prefer tokenizer path
        gold_info = compute_gold_answer_info(getattr(model, 'tokenizer', None), context_prompt, ground_truth, pieces_k=4)

        if gold_info.get("status") != "ok":
            # Fallback: construct sequences explicitly
            try:
                ctx_ids_try = model.tokenizer(context_prompt, add_special_tokens=False)["input_ids"]
                ctx_ans_ws_try = model.tokenizer(context_prompt + " " + ground_truth, add_special_tokens=False)["input_ids"]
                ctx_ans_ns_try = model.tokenizer(context_prompt + ground_truth, add_special_tokens=False)["input_ids"]
                convert = getattr(model.tokenizer, 'convert_ids_to_tokens', None)
                gold_info = compute_gold_answer_info_from_sequences(
                    ctx_ids_try, ctx_ans_ws_try, ctx_ans_ns_try,
                    pieces_k=4,
                    convert_ids_to_tokens=convert,
                    decode_id=(lambda i: model.tokenizer.decode([i])),
                    answer_str=ground_truth,
                )
            except Exception:
                try:
                    ctx_ids_try = model.to_tokens(context_prompt)[0].tolist()
                    ctx_ans_ws_try = model.to_tokens(context_prompt + " " + ground_truth)[0].tolist()
                    ctx_ans_ns_try = model.to_tokens(context_prompt + ground_truth)[0].tolist()
                    dec = (lambda i: model.tokenizer.decode([i])) if hasattr(model, 'tokenizer') else None
                    gold_info = compute_gold_answer_info_from_sequences(
                        ctx_ids_try, ctx_ans_ws_try, ctx_ans_ns_try,
                        pieces_k=4,
                        convert_ids_to_tokens=getattr(getattr(model, 'tokenizer', None), 'convert_ids_to_tokens', None),
                        decode_id=dec,
                        answer_str=ground_truth,
                    )
                except Exception:
                    gold_info = {
                        "string": ground_truth,
                        "status": "unresolved",
                        "variant": "unknown",
                        "first_id": None,
                        "pieces": [],
                        "answer_ids": [],
                        "ctx_ids": [],
                        "ctx_len": 0,
                    }

        # Provide ctx ids for copy detector and first answer id for metrics
        ctx_ids = gold_info.get("ctx_ids", [])
        first_ans_id = gold_info.get("first_id", None)

        # Control gold alignment (Paris)
        gold_info_ctl = compute_gold_answer_info(getattr(model, 'tokenizer', None), context_prompt_ctl, control_ground_truth, pieces_k=4)
        if gold_info_ctl.get("status") != "ok":
            try:
                c_ctx = model.tokenizer(context_prompt_ctl, add_special_tokens=False)["input_ids"]
                c_ws = model.tokenizer(context_prompt_ctl + " " + control_ground_truth, add_special_tokens=False)["input_ids"]
                c_ns = model.tokenizer(context_prompt_ctl + control_ground_truth, add_special_tokens=False)["input_ids"]
                gold_info_ctl = compute_gold_answer_info_from_sequences(
                    c_ctx, c_ws, c_ns,
                    pieces_k=4,
                    convert_ids_to_tokens=getattr(model.tokenizer, 'convert_ids_to_tokens', None),
                    decode_id=(lambda i: model.tokenizer.decode([i])),
                    answer_str=control_ground_truth,
                )
            except Exception:
                gold_info_ctl = {
                    "string": control_ground_truth,
                    "status": "unresolved",
                    "variant": "unknown",
                    "first_id": None,
                    "pieces": [],
                    "answer_ids": [],
                    "ctx_ids": [],
                    "ctx_len": 0,
                }
        ctx_ids_ctl = gold_info_ctl.get("ctx_ids", [])
        first_ans_id_ctl = gold_info_ctl.get("first_id", None)
        # Rolling window of the last k top-1 IDs
        window_ids: list[int] = []
        
        # Storage to collect pure_next_token_records for L_copy/L_semantic computation
        collected_pure_records = []
        
        # Begin capturing residual streams
        with torch.no_grad():
            # MEMORY EFFICIENT: Use targeted caching instead of run_with_cache
            # (removed human-readable progress print)
            
            # Storage for only the residual streams we need
            residual_cache = {}
            # Build hook closure
            cache_hook = build_cache_hook(residual_cache)
            # Attach hooks and record handles
            hooks, has_pos_embed = attach_residual_hooks(model, cache_hook)
            
            try:
                # Run forward pass with targeted hooks
                logits = model(tokens)

                # Cache final head reference distribution once (for KL-to-final)
                final_logits = logits[0, -1, :].float()
                final_probs = torch.softmax(final_logits, dim=0)
                # Representation-drift baseline direction (PROJECT_NOTES §1.5)
                _final_norm = torch.norm(final_logits) + 1e-12
                final_dir = (final_logits / _final_norm)
                final_top1_id = int(torch.argmax(final_probs).item())
                
                # Debug: print device placements of cached activations and tokens
                for name, t in residual_cache.items():
                    print(f"[DEBUG CACHE] {name} device: {t.device}")
                print(f"[DEBUG TOKENS] tokens device: {tokens.device}")
                
                # Show top predictions at different layers
                print(f"\nLayer-by-layer analysis of context: '{context_prompt}'")
                print("→ Predicting the first unseen token (what comes after the context)")
                if USE_NORM_LENS:
                    # Check if we'll actually be applying norms
                    if hasattr(model, 'blocks') and len(model.blocks) > 0:
                        first_norm = model.blocks[0].ln1
                        norm_type = type(first_norm).__name__
                        
                        if isinstance(first_norm, nn.LayerNorm):
                            print("Using NORMALIZED residual stream (LayerNorm applied - more accurate)")
                        elif 'RMS' in norm_type:
                            if _get_rms_scale(first_norm) is None:
                                print("Using NORMALIZED residual stream (RMS, no learnable scale)")
                            else:
                                print("Using NORMALIZED residual stream (RMS + learned scale)")
                        else:
                            print("Using RAW residual stream (unsupported normalization, skipping to avoid distortion)")
                    else:
                        print("Using RAW residual stream (no normalization layers found)")
                else:
                    print("Using RAW residual stream (normalization disabled)")
                print("Note: Shown probabilities are from full softmax (calibrated and comparable)")
                print(f"copy-collapse: top-1 ID-window in prompt & p>{config.copy_threshold}")
                if config.keep_residuals:
                    print("💾 Saving residual tensors to disk (--keep-residuals enabled)")
                print("-" * 60)
                
                # Get string representations of tokens for labeling output
                str_tokens = model.to_str_tokens(context_prompt)

                # Layer 0: embeddings (+ positional embeddings if available)
                print("Layer  0 (embeddings):")
                if has_pos_embed:
                    resid = (residual_cache['hook_embed'] +
                             residual_cache['hook_pos_embed'])
                else:
                    resid = residual_cache['hook_embed']
                    print("[diagnostic] No separate positional embedding hook found (as expected for rotary models).")
                    print("[diagnostic] Layer 0 contains TOKEN information only; positional info is injected inside attention layers.")
                # Keep a raw copy before normalization for dual-lens sanity
                resid_raw_L0 = resid
                # FIXED: Apply first real normalizer to embeddings if using norm-lens
                # This gives us the normalized embeddings that the model actually sees
                if USE_NORM_LENS:
                    # Use the actual first normalization layer instead of synthetic γ=1
                    print("[diagnostic] Applying real ln1 normalization to embeddings (not synthetic γ=1)")
                    if 'detected_architecture' not in locals():
                        detected_architecture = detect_model_architecture(model)
                    norm_module = get_correct_norm_module(model, 0, probe_after_block=False)
                    resid = apply_norm_or_skip(resid, norm_module)
                
                # Vectorized unembedding for all positions using analysis shadow weights
                resid_cast = safe_cast_for_unembed(resid[0], analysis_W_U, force_fp32_unembed=(config.fp32_unembed or USE_FP32_UNEMBED))
                logits_all = _unembed_mm(resid_cast, analysis_W_U, analysis_b_U)
                logits_all = logits_all.float()  # downstream numerics in fp32
                # Prism sidecar logits for all positions (if active)
                if prism_active:
                    Xw_L0 = whiten_apply(resid[0], prism_stats)
                    Xp_L0 = Xw_L0 @ prism_Q.to(Xw_L0.device)
                    Wp = analysis_W_U.float() if analysis_W_U.dtype != torch.float32 else analysis_W_U
                    bp = (analysis_b_U.float() if (analysis_b_U is not None and analysis_b_U.dtype != torch.float32) else analysis_b_U)
                    prism_logits_all_L0 = _unembed_mm(Xp_L0, Wp, bp)
                    prism_logits_all_L0 = prism_logits_all_L0.float()
                
                # Save residuals if requested
                if config.keep_residuals:
                    clean_name = clean_model_name(model_id)
                    resid_filename = f"{clean_name}_00_resid.pt"
                    # Use configured output directory; meta_filepath is not available here
                    resid_path = os.path.join(config.out_dir or os.getcwd(), resid_filename)
                    resid_cpu = resid.to(dtype=model.cfg.dtype if hasattr(model.cfg, 'dtype') else torch.float32).cpu()
                    torch.save(resid_cpu, resid_path)
                    del resid_cpu
                
                for pos in range(tokens.shape[1]):
                    layer_logits = logits_all[pos]
                    # Compute entropy in bits via centralized helper
                    entropy_bits = bits_entropy_from_logits(layer_logits)  # Prevent negative zero

                    token_str = str_tokens[pos]
                    # Decide verbosity for this position
                    verbose = is_verbose_position(pos, token_str, tokens.shape[1])
                    # Choose k based on verbosity
                    k = TOP_K_VERBOSE if verbose else TOP_K_RECORD
                    # Get top-k indices from raw logits
                    _, top_indices_k = torch.topk(layer_logits, k, largest=True, sorted=True)
                    full_probs  = torch.softmax(layer_logits, dim=0)
                    top_probs_k = full_probs[top_indices_k]
                    top_tokens_k = [decode_id(idx) for idx in top_indices_k]
                    print_summary(0, pos, token_str, entropy_bits, top_tokens_k, top_probs_k)
                    # Verbose console output removed to reduce noise - data still captured in files
                
                # Pure next-token record via helper (layer 0)
                emit_pure_next_token_record(
                    layer_out_idx=0,
                    logits_all=logits_all,
                    tokens_tensor=tokens,
                    ctx_ids_list=ctx_ids,
                    window_ids_list=window_ids,
                    final_probs_tensor=final_probs,
                    first_ans_token_id=first_ans_id,
                    final_dir_vec=final_dir,
                    collected_records=collected_pure_records,
                    do_raw_lens_sample=True,
                    resid_raw_tensor=resid_raw_L0,
                )
                # Prism sidecar: pure next-token (layer 0)
                if prism_active:
                    last_pos = tokens.shape[1] - 1
                    pz = prism_logits_all_L0[last_pos]
                    pprobs = torch.softmax(pz, dim=0)
                    pent = bits_entropy_from_logits(pz)
                    _, p_top_idx = torch.topk(pz, TOP_K_RECORD, largest=True, sorted=True)
                    p_top_probs = pprobs[p_top_idx]
                    p_top_tokens = [decode_id(idx) for idx in p_top_idx]
                    p_top1_id = p_top_idx[0].item()
                    # Update rolling window for copy detector (shares window_ids)
                    window_ids.append(p_top1_id)
                    if len(window_ids) > getattr(config, "copy_window_k", 1):
                        window_ids.pop(0)
                    p_copy = detect_copy_collapse_id_subseq(pz, ctx_ids, window_ids, copy_threshold=config.copy_threshold, copy_margin=config.copy_margin)
                    if p_copy and is_pure_whitespace_or_punct(p_top_tokens[0]):
                        p_copy = False
                    p_metrics = compute_next_token_metrics(pprobs, p_top1_id, final_probs, first_ans_id, topk_cum=5)
                    p_is_answer = (p_metrics.get("answer_rank") == 1) if p_metrics.get("answer_rank") is not None else is_semantic_top1(p_top_tokens[0], ground_truth)
                    _pn = torch.norm(pz) + 1e-12
                    p_cos = torch.dot((pz / _pn), final_dir).item()
                    json_data_prism["pure_next_token_records"].append({
                        "prompt_id": current_prompt_id,
                        "prompt_variant": current_prompt_variant,
                        "layer": 0,
                        "pos": last_pos,
                        "token": "⟨NEXT⟩",
                        "entropy": pent,
                        "topk": [[tok, prob.item()] for tok, prob in zip(p_top_tokens, p_top_probs)],
                        "copy_collapse": p_copy,
                        "entropy_collapse": pent <= 1.0,
                        "is_answer": p_is_answer,
                        "p_top1": p_metrics.get("p_top1"),
                        "p_top5": p_metrics.get("p_top5"),
                        "p_answer": p_metrics.get("p_answer"),
                        "kl_to_final_bits": p_metrics.get("kl_to_final_bits"),
                        "answer_rank": p_metrics.get("answer_rank"),
                        "cos_to_final": p_cos,
                        "control_margin": None,
                    })
                
                # --- free Layer-0 residual to keep host RAM flat ---------------------
                del resid
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Layers 1 to n_layers: after each transformer block
                # Detect architecture once for efficiency
                detected_architecture = detect_model_architecture(model)
                print(f"Detected architecture: {detected_architecture}")
                
                # Determine number of layers for iteration (no longer set by hook attachment)
                n_layers = model.cfg.n_layers
                # Decide which layers to sample for raw-vs-norm (1-indexed for post-block layers)
                # Build a closure to test sampling quickly
                def _should_sample(one_indexed: int) -> bool:
                    return should_sample_layer(RAW_LENS_MODE, n_layers, one_indexed)

                for layer in range(n_layers):
                    print(f"Layer {layer + 1:2d} (after transformer block {layer}):")
                    # Get residual stream after this layer's block
                    resid = residual_cache[f'blocks.{layer}.hook_resid_post']
                    resid_raw = resid  # keep pre-norm for raw lens
                    
                    # Apply normalization if requested
                    if USE_NORM_LENS:
                        # Use the correct normalization module based on probe timing and architecture
                        norm_module = get_correct_norm_module(model, layer, probe_after_block=True)
                        resid = apply_norm_or_skip(resid, norm_module)
                    
                    # Vectorized unembedding for all positions using analysis shadow weights
                    resid_cast = safe_cast_for_unembed(resid[0], analysis_W_U, force_fp32_unembed=(config.fp32_unembed or USE_FP32_UNEMBED))
                    logits_all = _unembed_mm(resid_cast, analysis_W_U, analysis_b_U)
                    logits_all = logits_all.float()  # downstream numerics in fp32
                    # Prism sidecar logits (post-block) for all positions
                    if prism_active:
                        Xw = whiten_apply(resid[0], prism_stats)
                        Xp = Xw @ prism_Q.to(Xw.device)
                        Wp = analysis_W_U.float() if analysis_W_U.dtype != torch.float32 else analysis_W_U
                        bp = (analysis_b_U.float() if (analysis_b_U is not None and analysis_b_U.dtype != torch.float32) else analysis_b_U)
                        prism_logits_all = _unembed_mm(Xp, Wp, bp)
                        prism_logits_all = prism_logits_all.float()
                    
                    # Save residuals if requested
                    if config.keep_residuals:
                        clean_name = clean_model_name(model_id)
                        resid_filename = f"{clean_name}_{layer+1:02d}_resid.pt"
                        # Use configured output directory; meta_filepath is not available here
                        resid_path = os.path.join(config.out_dir or os.getcwd(), resid_filename)
                        resid_cpu = resid.to(dtype=model.cfg.dtype if hasattr(model.cfg, 'dtype') else torch.float32).cpu()
                        torch.save(resid_cpu, resid_path)
                        del resid_cpu
                    
                    for pos in range(tokens.shape[1]):
                        layer_logits = logits_all[pos]
                        # fresh per-token probabilities for THIS layer
                        full_probs  = torch.softmax(layer_logits, dim=0)
                        entropy_bits = bits_entropy_from_logits(layer_logits)

                        token_str = str_tokens[pos]
                        # Decide verbosity for this position
                        verbose = is_verbose_position(pos, token_str, tokens.shape[1])
                        # Choose k based on verbosity
                        k = TOP_K_VERBOSE if verbose else TOP_K_RECORD
                        # Get top-k indices from raw logits
                        _, top_indices_k = torch.topk(layer_logits, k, largest=True, sorted=True)
                        top_probs_k = full_probs[top_indices_k]
                        top_tokens_k = [decode_id(idx) for idx in top_indices_k]
                    print_summary(layer + 1, pos, token_str, entropy_bits, top_tokens_k, top_probs_k)
                    # Verbose console output removed to reduce noise - data still captured in files
                    # Prism sidecar record (post-block layer)
                    if prism_active:
                        pz = prism_logits_all[pos]
                        pprobs = torch.softmax(pz, dim=0)
                        pent = bits_entropy_from_logits(pz)
                        _, p_top_idx = torch.topk(pz, k, largest=True, sorted=True)
                        p_top_probs = pprobs[p_top_idx]
                        p_top_tokens = [decode_id(idx) for idx in p_top_idx]
                        json_data_prism["records"].append({
                            "type": "record",
                            "prompt_id": current_prompt_id,
                            "prompt_variant": current_prompt_variant,
                            "layer": layer + 1,
                            "pos": pos,
                            "token": token_str,
                            "entropy": pent,
                            "topk": [[tok, prob.item()] for tok, prob in zip(p_top_tokens, p_top_probs)],
                        })
                    
                    # Pure next-token record via helper (post-block layer)
                    emit_pure_next_token_record(
                        layer_out_idx=layer + 1,
                        logits_all=logits_all,
                        tokens_tensor=tokens,
                        ctx_ids_list=ctx_ids,
                        window_ids_list=window_ids,
                        final_probs_tensor=final_probs,
                        first_ans_token_id=first_ans_id,
                        final_dir_vec=final_dir,
                        collected_records=collected_pure_records,
                        do_raw_lens_sample=_should_sample(layer + 1),
                        resid_raw_tensor=resid_raw,
                    )
                    # Prism sidecar pure next-token record (post-block layer)
                    if prism_active:
                        last_pos = tokens.shape[1] - 1
                        pz = prism_logits_all[last_pos]
                        pprobs = torch.softmax(pz, dim=0)
                        pent = bits_entropy_from_logits(pz)
                        _, p_top_idx = torch.topk(pz, TOP_K_RECORD, largest=True, sorted=True)
                        p_top_probs = pprobs[p_top_idx]
                        p_top_tokens = [decode_id(idx) for idx in p_top_idx]
                        p_top1_id = p_top_idx[0].item()
                        window_ids.append(p_top1_id)
                        if len(window_ids) > getattr(config, "copy_window_k", 1):
                            window_ids.pop(0)
                        p_copy = detect_copy_collapse_id_subseq(pz, ctx_ids, window_ids, copy_threshold=config.copy_threshold, copy_margin=config.copy_margin)
                        if p_copy and is_pure_whitespace_or_punct(p_top_tokens[0]):
                            p_copy = False
                        p_metrics = compute_next_token_metrics(pprobs, p_top1_id, final_probs, first_ans_id, topk_cum=5)
                        p_is_answer = (p_metrics.get("answer_rank") == 1) if p_metrics.get("answer_rank") is not None else is_semantic_top1(p_top_tokens[0], ground_truth)
                        _pn = torch.norm(pz) + 1e-12
                        p_cos = torch.dot((pz / _pn), final_dir).item()
                        json_data_prism["pure_next_token_records"].append({
                            "prompt_id": current_prompt_id,
                            "prompt_variant": current_prompt_variant,
                            "layer": layer + 1,
                            "pos": last_pos,
                            "token": "⟨NEXT⟩",
                            "entropy": pent,
                            "topk": [[tok, prob.item()] for tok, prob in zip(p_top_tokens, p_top_probs)],
                            "copy_collapse": p_copy,
                            "entropy_collapse": pent <= 1.0,
                            "is_answer": p_is_answer,
                            "p_top1": p_metrics.get("p_top1"),
                            "p_top5": p_metrics.get("p_top5"),
                            "p_answer": p_metrics.get("p_answer"),
                            "kl_to_final_bits": p_metrics.get("kl_to_final_bits"),
                            "answer_rank": p_metrics.get("answer_rank"),
                            "cos_to_final": p_cos,
                            "control_margin": None,
                        })

                    # Record last-layer lens vs final-head consistency snapshot
                    if layer == (n_layers - 1):
                        # Use the exact values we just computed for the last position
                        last_pos = tokens.shape[1] - 1
                        last_logits = logits_all[last_pos]
                        last_full_probs = torch.softmax(last_logits, dim=0)
                        _, last_top_indices = torch.topk(last_logits, TOP_K_RECORD, largest=True, sorted=True)
                        lens_top1_id = int(last_top_indices[0].item())
                        # metrics relative to final head
                        m = compute_next_token_metrics(last_full_probs, lens_top1_id, final_probs, first_ans_id, topk_cum=5)
                        # Estimate a scalar temperature s that best aligns lens to final
                        try:
                            zs = last_logits.float()
                            # Search s ∈ [0.1, 10] (log-space) for minimal KL(P(z/s)||final)
                            s_values = torch.logspace(-1, 1, steps=25, dtype=torch.float32).tolist()
                            best_s = float('nan')
                            best_kl = float('inf')
                            for s in s_values:
                                s_f = float(s)
                                P = torch.softmax(zs / s_f, dim=0)
                                kl = float(kl_bits(P, final_probs))
                                if kl < best_kl:
                                    best_kl = kl
                                    best_s = s_f
                        except Exception:
                            best_s = None
                            best_kl = None

                        # KL after simple family-specific transforms (last layer only)
                        kl_after_scale = None
                        kl_after_softcap = None
                        kl_after_scale_then_softcap = None
                        cfg_transform = {"scale": head_scale_cfg, "softcap": head_softcap_cfg}

                        try:
                            zs = last_logits.float()
                            if head_scale_cfg is not None and head_scale_cfg > 0:
                                P = torch.softmax(zs / float(head_scale_cfg), dim=0)
                                kl = float(kl_bits(P, final_probs))
                                kl_after_scale = kl
                            if head_softcap_cfg is not None and head_softcap_cfg > 0:
                                c = float(head_softcap_cfg)
                                zs_c = torch.tanh(zs / c) * c
                                P = torch.softmax(zs_c, dim=0)
                                kl_after_softcap = float(kl_bits(P, final_probs))
                            if (head_scale_cfg is not None and head_scale_cfg > 0) and (head_softcap_cfg is not None and head_softcap_cfg > 0):
                                c = float(head_softcap_cfg)
                                s = float(head_scale_cfg)
                                zs_sc = torch.tanh((zs / s) / c) * c
                                P = torch.softmax(zs_sc, dim=0)
                                kl_after_scale_then_softcap = float(kl_bits(P, final_probs))
                        except Exception:
                            pass

                        last_layer_consistency = {
                            "kl_to_final_bits": m.get("kl_to_final_bits"),
                            "top1_agree": bool(lens_top1_id == final_top1_id),
                            "p_top1_lens": m.get("p_top1"),
                            "p_top1_model": float(final_probs[final_top1_id].item()),
                            "p_answer_lens": m.get("p_answer"),
                            "answer_rank_lens": m.get("answer_rank"),
                            # Temperature probe: best scalar s and KL after rescale
                            "temp_est": best_s,
                            "kl_after_temp_bits": best_kl,
                            # Config-reported head transforms and KL after applying them
                            "cfg_transform": cfg_transform,
                            "kl_after_transform_bits": {
                                "scale": kl_after_scale,
                                "softcap": kl_after_softcap,
                                "scale_then_softcap": kl_after_scale_then_softcap,
                            },
                            # Advisory warning for family-agnostic visibility
                            "warn_high_last_layer_kl": bool(m.get("kl_to_final_bits") is not None and m.get("kl_to_final_bits") > 0.5),
                        }
                    
                    # --- free residual for this layer -----------------------------------
                    del resid
                    del residual_cache[f'blocks.{layer}.hook_resid_post']
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Summarize collapse and thresholds via helper
                diag.update(
                    summarize_pure_records(
                        collected_pure_records,
                        copy_threshold=config.copy_threshold,
                        copy_window_k=getattr(config, "copy_window_k", 1),
                        copy_match_level="id_subsequence",
                    )
                )
                # Record gold-alignment status (PROJECT_NOTES §1.7)
                diag["gold_alignment"] = "ok" if gold_info.get("status") == "ok" else "unresolved"
                # Capture orig variant summary fields for ablation
                L_copy_orig = diag.get("L_copy")
                L_sem_orig = diag.get("L_semantic")
                json_data["diagnostics"] = diag
                if last_layer_consistency is not None:
                    json_data["diagnostics"]["last_layer_consistency"] = last_layer_consistency

                # Summarize raw-vs-norm sanity samples (PROJECT_NOTES §1.4)
                try:
                    json_data["raw_lens_check"]["summary"] = summarize_raw_lens_check(
                        json_data["raw_lens_check"]["samples"]
                    )
                except Exception:
                    pass
                
            finally:
                # Clean up hooks and cache
                detach_hooks(hooks)
                
                # Aggressively free memory
                residual_cache.clear()  # free the dict but keep the name alive
                hooks.clear()  # Keep the variable, just empty it
                gc.collect()  # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Ensure all CUDA operations are complete
            
            # === stop capturing large activations – probes don't need them =======


            # Let's also see what the actual model would predict (final layer)
            print("=" * 60)
            print("ACTUAL MODEL PREDICTION (for comparison):")

            # final_logits already computed above; ensure float
            final_logits = final_logits.float()
            _, final_top_indices = torch.topk(final_logits, 20, largest=True, sorted=True)
            final_full_probs = torch.softmax(final_logits, dim=0)
            final_top_probs = final_full_probs[final_top_indices]
            
            # Calculate final entropy via zero-safe helper
            final_entropy_bits = bits_entropy_from_logits(final_logits)
            
            # Collect final prediction data
            final_record = {
                "type": "final_prediction",
                "entropy": final_entropy_bits,
                "topk": [[decode_id(idx), prob.item()] for prob, idx in zip(final_top_probs, final_top_indices)]
            }
            json_data["final_prediction"] = final_record
            
            # Emit additional probing records for test prompts
            # Process test prompts in smaller batches to reduce memory usage
            test_prompts = [
                "Berlin is the capital of",
                "Germany's capital city is called simply",
                "The capital city of Germany is named simply",
                "Germany has its capital at the city called simply",
                "In Germany the capital city is simply",            
                "Germany's capital city is called",
                "The capital city of Germany is named",
                "Germany has its capital at",
                "In Germany the capital city is known as",
                "Give the country name only, plain text. Berlin is the capital of",
                "Give the city name only, plain text. Germany's capital city is called",
                "Give the city name only, plain text. The capital city of Germany is named simply",
                "Give the city name only, plain text. Germany has its capital at",
                "Give the city name only, plain text. In Germany, the capital city is known as",
            ]
            
            # Process test prompts one at a time to minimize memory usage
            for test_prompt in test_prompts:
                test_tokens = model.to_tokens(test_prompt)      # let Accelerate move it
                
                test_logits = model(test_tokens)
                last_slice = test_logits[0, -1, :]
                _, test_top_indices = torch.topk(last_slice, 10, largest=True, sorted=True)
                test_full_probs = torch.softmax(last_slice, dim=0)
                test_top_probs = test_full_probs[test_top_indices]
                test_entropy_bits = bits_entropy_from_logits(last_slice)
                # Collect test prompt data
                probe_record = {
                    "type": "test_prompt",
                    "prompt": test_prompt,
                    "entropy": test_entropy_bits,
                    "topk": [[decode_id(idx), prob.item()] for prob, idx in zip(test_top_probs, test_top_indices)]
                }
                json_data["test_prompts"].append(probe_record)
                
                # Clean up tensors immediately
                del test_tokens, test_logits, test_top_indices, test_full_probs, test_top_probs
            
            # Emit temperature exploration records
            # Note: Using consistent prompt for temperature exploration to maintain comparability
            temp_test_prompt = "Give the city name only, plain text. The capital of Germany is called simply"
            temp_tokens = model.to_tokens(temp_test_prompt)      # let Accelerate move it
            
            # Single forward pass - then rescale for different temperatures
            base_logits = model(temp_tokens)[0, -1, :]
            
            temperatures = [0.1, 2.0]
            
            for temp in temperatures:
                # Compute for temperature and emit JSONL record
                
                # Rescale existing logits instead of new forward pass (cast to float32 for numerical stability)
                scaled_logits = (base_logits / temp).float()
                _, temp_top_indices = torch.topk(scaled_logits, 15, largest=True, sorted=True)
                temp_full_probs = torch.softmax(scaled_logits, dim=0)
                temp_top_probs = temp_full_probs[temp_top_indices]
                temp_entropy_bits = bits_entropy_from_logits(scaled_logits)
                # Collect temperature exploration data
                temp_record = {
                    "type": "temperature_exploration",
                    "temperature": temp,
                    "entropy": temp_entropy_bits,
                    "topk": [[decode_id(idx), prob.item()] for prob, idx in zip(temp_top_probs, temp_top_indices)]
                }
                json_data["temperature_exploration"].append(temp_record)
            
            # Clean up temperature exploration tensors
            del temp_tokens, base_logits
        
        print("=== END OF INSPECTING ==============\n")

        # ---------------- Ablation pass: no-filler (PROJECT_NOTES §1.9) --------
        # Run a separate forward pass on the positive prompt without the stylistic filler
        current_prompt_id = "pos"
        current_prompt_variant = "no_filler"
        # Gold alignment for the ablated variant
        gold_info_nf = compute_gold_answer_info(getattr(model, 'tokenizer', None), context_prompt_nf, ground_truth, pieces_k=4)
        if gold_info_nf.get("status") != "ok":
            try:
                n_ctx = model.tokenizer(context_prompt_nf, add_special_tokens=False)["input_ids"]
                n_ws = model.tokenizer(context_prompt_nf + " " + ground_truth, add_special_tokens=False)["input_ids"]
                n_ns = model.tokenizer(context_prompt_nf + ground_truth, add_special_tokens=False)["input_ids"]
                gold_info_nf = compute_gold_answer_info_from_sequences(
                    n_ctx, n_ws, n_ns,
                    pieces_k=4,
                    convert_ids_to_tokens=getattr(model.tokenizer, 'convert_ids_to_tokens', None),
                    decode_id=(lambda i: model.tokenizer.decode([i])),
                    answer_str=ground_truth,
                )
            except Exception:
                gold_info_nf = {
                    "string": ground_truth,
                    "status": "unresolved",
                    "variant": "unknown",
                    "first_id": None,
                    "pieces": [],
                    "answer_ids": [],
                    "ctx_ids": [],
                    "ctx_len": 0,
                }
        ctx_ids_nf = gold_info_nf.get("ctx_ids", [])
        first_ans_id_nf = gold_info_nf.get("first_id", None)

        # Per-variant rolling window and record collection
        window_ids_nf: list[int] = []
        collected_pure_records_nf = []

        with torch.no_grad():
            residual_cache = {}
            cache_hook = build_cache_hook(residual_cache)
            hooks, _ = attach_residual_hooks(model, cache_hook)
            try:
                tokens_nf = model.to_tokens(context_prompt_nf)
                logits_nf = model(tokens_nf)
                final_logits_nf = logits_nf[0, -1, :].float()
                final_probs_nf = torch.softmax(final_logits_nf, dim=0)
                _final_norm_nf = torch.norm(final_logits_nf) + 1e-12
                final_dir_nf = (final_logits_nf / _final_norm_nf)

                # Layer 0
                resid0 = residual_cache['hook_embed']
                if 'hook_pos_embed' in residual_cache:
                    resid0 = resid0 + residual_cache['hook_pos_embed']
                resid = resid0
                resid_raw_tensor = resid.detach().clone() if RAW_LENS_MODE != "off" else None
                if USE_NORM_LENS:
                    norm_module = get_correct_norm_module(model, 0, probe_after_block=False)
                    resid = apply_norm_or_skip(resid, norm_module)
                casted = safe_cast_for_unembed(resid[0, :, :], analysis_W_U, force_fp32_unembed=(config.fp32_unembed or USE_FP32_UNEMBED))
                layer_logits = _unembed_mm(casted, analysis_W_U, analysis_b_U).float()
                if prism_active:
                    Xw_nf0 = whiten_apply(resid[0, :, :], prism_stats)
                    Xp_nf0 = Xw_nf0 @ prism_Q.to(Xw_nf0.device)
                    Wp = analysis_W_U.float() if analysis_W_U.dtype != torch.float32 else analysis_W_U
                    bp = (analysis_b_U.float() if (analysis_b_U is not None and analysis_b_U.dtype != torch.float32) else analysis_b_U)
                    prism_logits_all_nf0 = _unembed_mm(Xp_nf0, Wp, bp).float()
                emit_pure_next_token_record(
                    0,
                    layer_logits,
                    tokens_nf,
                    ctx_ids_nf,
                    window_ids_nf,
                    final_probs_nf,
                    first_ans_id_nf,
                    final_dir_nf,
                    collected_records=collected_pure_records_nf,
                    do_raw_lens_sample=False,
                    resid_raw_tensor=resid_raw_tensor,
                )
                if prism_active:
                    last_pos = tokens_nf.shape[1] - 1
                    pz = prism_logits_all_nf0[last_pos]
                    pprobs = torch.softmax(pz, dim=0)
                    pent = bits_entropy_from_logits(pz)
                    _, p_top_idx = torch.topk(pz, TOP_K_RECORD, largest=True, sorted=True)
                    p_top_probs = pprobs[p_top_idx]
                    p_top_tokens = [decode_id(idx) for idx in p_top_idx]
                    p_top1_id = p_top_idx[0].item()
                    window_ids_nf.append(p_top1_id)
                    if len(window_ids_nf) > getattr(config, "copy_window_k", 1):
                        window_ids_nf.pop(0)
                    p_copy = detect_copy_collapse_id_subseq(pz, ctx_ids_nf, window_ids_nf, copy_threshold=config.copy_threshold, copy_margin=config.copy_margin)
                    if p_copy and is_pure_whitespace_or_punct(p_top_tokens[0]):
                        p_copy = False
                    p_metrics = compute_next_token_metrics(pprobs, p_top1_id, final_probs_nf, first_ans_id_nf, topk_cum=5)
                    p_is_answer = (p_metrics.get("answer_rank") == 1) if p_metrics.get("answer_rank") is not None else is_semantic_top1(p_top_tokens[0], ground_truth)
                    _pn = torch.norm(pz) + 1e-12
                    p_cos = torch.dot((pz / _pn), final_dir_nf).item()
                    json_data_prism["pure_next_token_records"].append({
                        "prompt_id": current_prompt_id,
                        "prompt_variant": current_prompt_variant,
                        "layer": 0,
                        "pos": last_pos,
                        "token": "⟨NEXT⟩",
                        "entropy": pent,
                        "topk": [[tok, prob.item()] for tok, prob in zip(p_top_tokens, p_top_probs)],
                        "copy_collapse": p_copy,
                        "entropy_collapse": pent <= 1.0,
                        "is_answer": p_is_answer,
                        "p_top1": p_metrics.get("p_top1"),
                        "p_top5": p_metrics.get("p_top5"),
                        "p_answer": p_metrics.get("p_answer"),
                        "kl_to_final_bits": p_metrics.get("kl_to_final_bits"),
                        "answer_rank": p_metrics.get("answer_rank"),
                        "cos_to_final": p_cos,
                        "control_margin": None,
                    })

                # Post-block layers
                n_layers = model.cfg.n_layers
                for layer in range(n_layers):
                    resid = residual_cache[f'blocks.{layer}.hook_resid_post']
                    resid_raw_tensor = resid.detach().clone() if RAW_LENS_MODE != "off" else None
                    if USE_NORM_LENS:
                        norm_module = get_correct_norm_module(model, layer, probe_after_block=True)
                        resid = apply_norm_or_skip(resid, norm_module)
                    casted = safe_cast_for_unembed(resid[0, :, :], analysis_W_U, force_fp32_unembed=(config.fp32_unembed or USE_FP32_UNEMBED))
                    layer_logits = _unembed_mm(casted, analysis_W_U, analysis_b_U).float()
                    if prism_active:
                        Xw_nfl = whiten_apply(resid[0, :, :], prism_stats)
                        Xp_nfl = Xw_nfl @ prism_Q.to(Xw_nfl.device)
                        Wp = analysis_W_U.float() if analysis_W_U.dtype != torch.float32 else analysis_W_U
                        bp = (analysis_b_U.float() if (analysis_b_U is not None and analysis_b_U.dtype != torch.float32) else analysis_b_U)
                        prism_logits_all_nfl = _unembed_mm(Xp_nfl, Wp, bp).float()
                    emit_pure_next_token_record(
                        layer + 1,
                        layer_logits,
                        tokens_nf,
                        ctx_ids_nf,
                        window_ids_nf,
                        final_probs_nf,
                        first_ans_id_nf,
                        final_dir_nf,
                        collected_records=collected_pure_records_nf,
                        do_raw_lens_sample=False,
                        resid_raw_tensor=resid_raw_tensor,
                    )
                    if prism_active:
                        last_pos = tokens_nf.shape[1] - 1
                        pz = prism_logits_all_nfl[last_pos]
                        pprobs = torch.softmax(pz, dim=0)
                        pent = bits_entropy_from_logits(pz)
                        _, p_top_idx = torch.topk(pz, TOP_K_RECORD, largest=True, sorted=True)
                        p_top_probs = pprobs[p_top_idx]
                        p_top_tokens = [decode_id(idx) for idx in p_top_idx]
                        p_top1_id = p_top_idx[0].item()
                        window_ids_nf.append(p_top1_id)
                        if len(window_ids_nf) > getattr(config, "copy_window_k", 1):
                            window_ids_nf.pop(0)
                        p_copy = detect_copy_collapse_id_subseq(pz, ctx_ids_nf, window_ids_nf, copy_threshold=config.copy_threshold, copy_margin=config.copy_margin)
                        if p_copy and is_pure_whitespace_or_punct(p_top_tokens[0]):
                            p_copy = False
                        p_metrics = compute_next_token_metrics(pprobs, p_top1_id, final_probs_nf, first_ans_id_nf, topk_cum=5)
                        p_is_answer = (p_metrics.get("answer_rank") == 1) if p_metrics.get("answer_rank") is not None else is_semantic_top1(p_top_tokens[0], ground_truth)
                        _pn = torch.norm(pz) + 1e-12
                        p_cos = torch.dot((pz / _pn), final_dir_nf).item()
                        json_data_prism["pure_next_token_records"].append({
                            "prompt_id": current_prompt_id,
                            "prompt_variant": current_prompt_variant,
                            "layer": layer + 1,
                            "pos": last_pos,
                            "token": "⟨NEXT⟩",
                            "entropy": pent,
                            "topk": [[tok, prob.item()] for tok, prob in zip(p_top_tokens, p_top_probs)],
                            "copy_collapse": p_copy,
                            "entropy_collapse": pent <= 1.0,
                            "is_answer": p_is_answer,
                            "p_top1": p_metrics.get("p_top1"),
                            "p_top5": p_metrics.get("p_top5"),
                            "p_answer": p_metrics.get("p_answer"),
                            "kl_to_final_bits": p_metrics.get("kl_to_final_bits"),
                            "answer_rank": p_metrics.get("answer_rank"),
                            "cos_to_final": p_cos,
                            "control_margin": None,
                        })

                # Summarize ablation variant
                diag_nf = summarize_pure_records(
                    collected_pure_records_nf,
                    copy_threshold=config.copy_threshold,
                    copy_window_k=getattr(config, "copy_window_k", 1),
                    copy_match_level="id_subsequence",
                )
                L_copy_nf = diag_nf.get("L_copy")
                L_sem_nf = diag_nf.get("L_semantic")
                json_data["ablation_summary"] = {
                    "L_copy_orig": L_copy_orig,
                    "L_sem_orig": L_sem_orig,
                    "L_copy_nf": L_copy_nf,
                    "L_sem_nf": L_sem_nf,
                    "delta_L_copy": (None if (L_copy_orig is None or L_copy_nf is None) else (L_copy_nf - L_copy_orig)),
                    "delta_L_sem": (None if (L_sem_orig is None or L_sem_nf is None) else (L_sem_nf - L_sem_orig)),
                }
            finally:
                detach_hooks(hooks)
                residual_cache.clear()
                hooks.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ---------------- Control pass (PROJECT_NOTES §1.8) --------------------
        # Run a separate forward pass on the control prompt (France → Paris)
        current_prompt_id = "ctl"
        current_prompt_variant = "orig"
        with torch.no_grad():
            residual_cache = {}
            cache_hook = build_cache_hook(residual_cache)
            hooks, _ = attach_residual_hooks(model, cache_hook)
            try:
                tokens_ctl = model.to_tokens(context_prompt_ctl)
                logits_ctl = model(tokens_ctl)

                final_logits_ctl = logits_ctl[0, -1, :].float()
                final_probs_ctl = torch.softmax(final_logits_ctl, dim=0)
                _final_norm_ctl = torch.norm(final_logits_ctl) + 1e-12
                final_dir_ctl = (final_logits_ctl / _final_norm_ctl)

                # Layer 0 (embedding-only)
                resid0 = residual_cache['hook_embed']
                if 'hook_pos_embed' in residual_cache:
                    resid0 = resid0 + residual_cache['hook_pos_embed']
                resid = resid0
                resid_raw_tensor = resid.detach().clone() if RAW_LENS_MODE != "off" else None
                if USE_NORM_LENS:
                    norm_module = get_correct_norm_module(model, 0, probe_after_block=False)
                    resid = apply_norm_or_skip(resid, norm_module)
                casted = safe_cast_for_unembed(resid[0, :, :], analysis_W_U, force_fp32_unembed=(config.fp32_unembed or USE_FP32_UNEMBED))
                layer_logits = _unembed_mm(casted, analysis_W_U, analysis_b_U).float()
                # Prepare Prism logits for control L0 if active
                if prism_active:
                    Xw_ctl0 = whiten_apply(resid[0, :, :], prism_stats)
                    Xp_ctl0 = Xw_ctl0 @ prism_Q.to(Xw_ctl0.device)
                    Wp = analysis_W_U.float() if analysis_W_U.dtype != torch.float32 else analysis_W_U
                    bp = (analysis_b_U.float() if (analysis_b_U is not None and analysis_b_U.dtype != torch.float32) else analysis_b_U)
                    prism_logits_all_ctl0 = _unembed_mm(Xp_ctl0, Wp, bp).float()
                emit_pure_next_token_record(
                    0,
                    layer_logits,
                    tokens_ctl,
                    ctx_ids_ctl,
                    window_ids,
                    final_probs_ctl,
                    first_ans_id_ctl,
                    final_dir_ctl,
                    collected_records=[],
                    do_raw_lens_sample=False,
                    resid_raw_tensor=resid_raw_tensor,
                    control_ids=(first_ans_id_ctl, first_ans_id),
                )
                if prism_active:
                    last_pos = tokens_ctl.shape[1] - 1
                    pz = prism_logits_all_ctl0[last_pos]
                    pprobs = torch.softmax(pz, dim=0)
                    pent = bits_entropy_from_logits(pz)
                    _, p_top_idx = torch.topk(pz, TOP_K_RECORD, largest=True, sorted=True)
                    p_top_probs = pprobs[p_top_idx]
                    p_top_tokens = [decode_id(idx) for idx in p_top_idx]
                    p_top1_id = p_top_idx[0].item()
                    window_ids.append(p_top1_id)
                    if len(window_ids) > getattr(config, "copy_window_k", 1):
                        window_ids.pop(0)
                    p_copy = detect_copy_collapse_id_subseq(pz, ctx_ids_ctl, window_ids, copy_threshold=config.copy_threshold, copy_margin=config.copy_margin)
                    if p_copy and is_pure_whitespace_or_punct(p_top_tokens[0]):
                        p_copy = False
                    p_metrics = compute_next_token_metrics(pprobs, p_top1_id, final_probs_ctl, first_ans_id_ctl, topk_cum=5)
                    p_is_answer = (p_metrics.get("answer_rank") == 1) if p_metrics.get("answer_rank") is not None else is_semantic_top1(p_top_tokens[0], control_ground_truth)
                    _pn = torch.norm(pz) + 1e-12
                    p_cos = torch.dot((pz / _pn), final_dir_ctl).item()
                    control_margin = None
                    try:
                        if first_ans_id_ctl is not None and first_ans_id is not None:
                            control_margin = float(pprobs[int(first_ans_id_ctl)]) - float(pprobs[int(first_ans_id)])
                    except Exception:
                        control_margin = None
                    json_data_prism["pure_next_token_records"].append({
                        "prompt_id": current_prompt_id,
                        "prompt_variant": current_prompt_variant,
                        "layer": 0,
                        "pos": last_pos,
                        "token": "⟨NEXT⟩",
                        "entropy": pent,
                        "topk": [[tok, prob.item()] for tok, prob in zip(p_top_tokens, p_top_probs)],
                        "copy_collapse": p_copy,
                        "entropy_collapse": pent <= 1.0,
                        "is_answer": p_is_answer,
                        "p_top1": p_metrics.get("p_top1"),
                        "p_top5": p_metrics.get("p_top5"),
                        "p_answer": p_metrics.get("p_answer"),
                        "kl_to_final_bits": p_metrics.get("kl_to_final_bits"),
                        "answer_rank": p_metrics.get("answer_rank"),
                        "cos_to_final": p_cos,
                        "control_margin": control_margin,
                    })

                # Each block's post-residual layers
                n_layers = model.cfg.n_layers
                for layer in range(n_layers):
                    resid = residual_cache[f'blocks.{layer}.hook_resid_post']
                    resid_raw_tensor = resid.detach().clone() if RAW_LENS_MODE != "off" else None
                    if USE_NORM_LENS:
                        norm_module = get_correct_norm_module(model, layer, probe_after_block=True)
                        resid = apply_norm_or_skip(resid, norm_module)
                casted = safe_cast_for_unembed(resid[0, :, :], analysis_W_U, force_fp32_unembed=(config.fp32_unembed or USE_FP32_UNEMBED))
                layer_logits = _unembed_mm(casted, analysis_W_U, analysis_b_U).float()
                if prism_active:
                    Xw_ctll = whiten_apply(resid[0, :, :], prism_stats)
                    Xp_ctll = Xw_ctll @ prism_Q.to(Xw_ctll.device)
                    Wp = analysis_W_U.float() if analysis_W_U.dtype != torch.float32 else analysis_W_U
                    bp = (analysis_b_U.float() if (analysis_b_U is not None and analysis_b_U.dtype != torch.float32) else analysis_b_U)
                    prism_logits_all_ctll = _unembed_mm(Xp_ctll, Wp, bp).float()
                emit_pure_next_token_record(
                    layer + 1,
                    layer_logits,
                    tokens_ctl,
                    ctx_ids_ctl,
                    window_ids,
                    final_probs_ctl,
                    first_ans_id_ctl,
                    final_dir_ctl,
                    collected_records=[],
                    do_raw_lens_sample=False,
                    resid_raw_tensor=resid_raw_tensor,
                    control_ids=(first_ans_id_ctl, first_ans_id),
                )
                if prism_active:
                    last_pos = tokens_ctl.shape[1] - 1
                    pz = prism_logits_all_ctll[last_pos]
                    pprobs = torch.softmax(pz, dim=0)
                    pent = bits_entropy_from_logits(pz)
                    _, p_top_idx = torch.topk(pz, TOP_K_RECORD, largest=True, sorted=True)
                    p_top_probs = pprobs[p_top_idx]
                    p_top_tokens = [decode_id(idx) for idx in p_top_idx]
                    p_top1_id = p_top_idx[0].item()
                    window_ids.append(p_top1_id)
                    if len(window_ids) > getattr(config, "copy_window_k", 1):
                        window_ids.pop(0)
                    p_copy = detect_copy_collapse_id_subseq(pz, ctx_ids_ctl, window_ids, copy_threshold=config.copy_threshold, copy_margin=config.copy_margin)
                    if p_copy and is_pure_whitespace_or_punct(p_top_tokens[0]):
                        p_copy = False
                    p_metrics = compute_next_token_metrics(pprobs, p_top1_id, final_probs_ctl, first_ans_id_ctl, topk_cum=5)
                    p_is_answer = (p_metrics.get("answer_rank") == 1) if p_metrics.get("answer_rank") is not None else is_semantic_top1(p_top_tokens[0], control_ground_truth)
                    _pn = torch.norm(pz) + 1e-12
                    p_cos = torch.dot((pz / _pn), final_dir_ctl).item()
                    control_margin = None
                    try:
                        if first_ans_id_ctl is not None and first_ans_id is not None:
                            control_margin = float(pprobs[int(first_ans_id_ctl)]) - float(pprobs[int(first_ans_id)])
                    except Exception:
                        control_margin = None
                    json_data_prism["pure_next_token_records"].append({
                        "prompt_id": current_prompt_id,
                        "prompt_variant": current_prompt_variant,
                        "layer": layer + 1,
                        "pos": last_pos,
                        "token": "⟨NEXT⟩",
                        "entropy": pent,
                        "topk": [[tok, prob.item()] for tok, prob in zip(p_top_tokens, p_top_probs)],
                        "copy_collapse": p_copy,
                        "entropy_collapse": pent <= 1.0,
                        "is_answer": p_is_answer,
                        "p_top1": p_metrics.get("p_top1"),
                        "p_top5": p_metrics.get("p_top5"),
                        "p_answer": p_metrics.get("p_answer"),
                        "kl_to_final_bits": p_metrics.get("kl_to_final_bits"),
                        "answer_rank": p_metrics.get("answer_rank"),
                        "cos_to_final": p_cos,
                        "control_margin": control_margin,
                    })
            finally:
                detach_hooks(hooks)
                residual_cache.clear()
                hooks.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Compute and persist control prompt info and summary
        first_pos = None
        max_margin = None
        for rec in json_data.get("pure_next_token_records", []):
            if rec.get("prompt_id") != "ctl":
                continue
            cm = rec.get("control_margin")
            if cm is None:
                continue
            if max_margin is None or cm > max_margin:
                max_margin = cm
            if cm > 0 and first_pos is None:
                first_pos = rec.get("layer")
        json_data["control_prompt"] = {
            "context_prompt": context_prompt_ctl,
            "gold_answer": {
                "string": gold_info_ctl.get("string"),
                "pieces": gold_info_ctl.get("pieces", []),
                "first_id": gold_info_ctl.get("first_id"),
                "answer_ids": gold_info_ctl.get("answer_ids", []),
                "variant": gold_info_ctl.get("variant", "unknown"),
            },
            "gold_alignment": "ok" if gold_info_ctl.get("status") == "ok" else "unresolved",
        }
        json_data["control_summary"] = {
            "first_control_margin_pos": first_pos,
            "max_control_margin": max_margin,
        }

        # Restore prompt_id for any subsequent records
        current_prompt_id = "pos"

        # Collect model stats data (architecture already detected above)
        stats_record = {
            "type": "model_stats",
            "num_layers": model.cfg.n_layers,
            "d_model": model.cfg.d_model,
            "n_heads": model.cfg.n_heads,
            "d_vocab": model.cfg.d_vocab,
            "n_ctx": model.cfg.n_ctx,
            "architecture": detected_architecture or 'unknown'
        }
        json_data["model_stats"] = stats_record

        # Persist gold-answer transparency (PROJECT_NOTES §1.7)
        json_data["gold_answer"] = {
            "string": gold_info.get("string"),
            "pieces": gold_info.get("pieces", []),
            "first_id": gold_info.get("first_id"),
            "answer_ids": gold_info.get("answer_ids", []),
            "variant": gold_info.get("variant", "unknown"),
        }
        
        # Clean up model to free memory (though process will end anyway)
        del model
        gc.collect()  # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Attach Prism buffers for outer writer (sidecar CSVs)
        json_data["prism_sidecar"] = json_data_prism
        return json_data

    try:
        # Run the experiment and let output print normally
        json_data = evaluate_model()

        # In self-test mode, do not write JSON/CSV artifacts
        if config.self_test:
            return json_data

        # Extract file paths
        meta_filepath, csv_filepath, pure_csv_filepath = output_files

        # Write CSV files FIRST (they need the full record lists)
        write_csv_files(json_data, csv_filepath, pure_csv_filepath, TOP_K_VERBOSE)
        # Prism sidecar CSVs (auto/on): only if artifacts were loaded and records exist
        try:
            prism_buf = json_data.get("prism_sidecar")
            prism_summary = (json_data.get("diagnostics") or {}).get("prism_summary") or {}
            if prism_summary.get("compatible") and prism_buf and (prism_buf["records"] or prism_buf["pure_next_token_records"]):
                out_dir = os.path.dirname(csv_filepath)
                clean_name = clean_model_name(model_id)
                csv_prism = os.path.join(out_dir, f"output-{clean_name}-records-prism.csv")
                pure_prism = os.path.join(out_dir, f"output-{clean_name}-pure-next-token-prism.csv")
                write_csv_files(prism_buf, csv_prism, pure_prism, TOP_K_VERBOSE)
                print(f"✅ Prism Records CSV saved to: {csv_prism}")
                print(f"✅ Prism Pure next-token CSV saved to: {pure_prism}")
            elif getattr(CLI_ARGS, "prism", "auto") != "off":
                # Single-line notice in auto mode when missing/incompatible
                err = prism_summary.get("error")
                if err:
                    print(f"ℹ️ Prism sidecar disabled: {err}")
                else:
                    print("ℹ️ Prism sidecar not written (no artifacts or empty buffers)")
        except Exception as e:
            print(f"⚠️ Failed to write Prism sidecar CSVs: {e}")

        # Strip bulky per-token records from JSON to keep it compact
        json_data_compact = {k: v for k, v in json_data.items()
                             if k not in ("records", "pure_next_token_records", "prism_sidecar")}

        # Write compact JSON metadata
        with open(meta_filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data_compact, f, ensure_ascii=False, indent=2)

        return json_data_compact
        
    except Exception as e:
        error_msg = f"ERROR evaluating {model_id}: {str(e)}"
        print(error_msg)
        raise

## csv writing helpers moved to layers_core.csv_io

def run_single_model(model_id):
    """Run experiment for a single model - used when called as subprocess"""
    print(f"\n{'='*80}")
    print(f"🚀 Starting subprocess for model: {model_id}")
    print(f"{'='*80}")
    
    # Set memory limits BEFORE any CUDA operations
    if torch.cuda.is_available():
        try:
            # Use 85% of GPU memory (increased from 80% since we're managing memory better)
            # torch.cuda.set_per_process_memory_fraction(0.85)
            # Also set environment variable for better memory management
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        except AttributeError:
            pass
    
    # Generate filename components
    clean_name = clean_model_name(model_id)
    meta_filename = f"output-{clean_name}.json"
    csv_filename  = f"output-{clean_name}-records.csv"
    pure_csv_filename = f"output-{clean_name}-pure-next-token.csv"

    # Determine output directory (parent may pass --out_dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if CLI_ARGS.out_dir:
        out_dir = CLI_ARGS.out_dir
    else:
        # For self-test, avoid rotating/creating run-latest; write nowhere
        if CLI_ARGS.self_test:
            out_dir = script_dir
        else:
            # Stand-alone invocation: create its own run-latest directory
            out_dir = setup_run_latest_directory(script_dir)
    # Ensure directory exists
    os.makedirs(out_dir, exist_ok=True)

    meta_filepath = os.path.join(out_dir, meta_filename)
    csv_filepath  = os.path.join(out_dir, csv_filename)
    pure_csv_filepath = os.path.join(out_dir, pure_csv_filename)
    
    try:
        # Resolve device: auto-pick if requested
        chosen_device = CLI_ARGS.device
        debug_info = None
        if CLI_ARGS.device == "auto":
            sel = select_best_device(model_id)
            if sel is None:
                print(f"⛔ No suitable device fits the model: {model_id}")
                # Write a minimal meta file noting skip
                with open(meta_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"SKIPPED (no device fit) FOR {model_id}\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y%m%d%H%M%S')}\n")
                return False
            dev, dtype, debug_info = sel
            chosen_device = dev
            print("📐 Device decision:")
            print(f"   device={dev} dtype={dtype} est_peak={debug_info.get('est_peak')}B available={debug_info.get('available')}B")

        # Run the experiment
        cfg = ExperimentConfig(
            device=chosen_device,
            fp32_unembed=CLI_ARGS.fp32_unembed,
            keep_residuals=CLI_ARGS.keep_residuals,
            copy_threshold=CLI_ARGS.copy_threshold,
            copy_margin=CLI_ARGS.copy_margin,
            out_dir=out_dir,
            self_test=CLI_ARGS.self_test,
        )
        data = run_experiment_for_model(model_id, (meta_filepath, csv_filepath, pure_csv_filepath), cfg)

        if CLI_ARGS.self_test:
            print("✅ Self-test complete. No artifacts written (by design).")
        else:
            print(f"✅ Experiment complete. JSON metadata saved to: {meta_filepath}")
            print(f"✅ Records CSV saved to: {csv_filepath}")
            print(f"✅ Pure next-token CSV saved to: {pure_csv_filepath}")
        return True
        
    except Exception as e:
        error_msg = f"❌ Failed to evaluate {model_id}: {str(e)}"
        print(error_msg)
        
        # Still save error output
        with open(meta_filepath, 'w', encoding='utf-8') as f:
            f.write(f"EXPERIMENT FAILED FOR {model_id}\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y%m%d%H%M%S')}\n")
        return False

# CLI -----------------------------------------------------------------------
def parse_cli():
    p = argparse.ArgumentParser(description="Layer-by-layer logit-lens sweep")
    p.add_argument("--device",
                   default="auto",
                   choices=["auto", "cuda", "mps", "cpu"],
                   help="compute device to run on (default: auto picks best fit)")
    p.add_argument("--fp32-unembed",
                   action="store_true",
                   help="Use FP32 shadow unembedding for analysis-only decoding (do not mutate model params)")
    p.add_argument("--keep-residuals",
                   action="store_true",
                   help="Dump full residual tensors; if absent, keep only per-layer logits")
    p.add_argument("--copy-threshold", type=float, default=0.95,
                   help="Minimum P(top-1) for copy collapse")
    p.add_argument("--copy-margin", type=float, default=0.10,
                   help="Require P(top-1) − P(top-2) > margin for copy collapse")
    p.add_argument("model_id", nargs="?", default=None,
                   help="Model ID for single-run (when invoking as subprocess)")
    p.add_argument("--out_dir",
                   default=None,
                   help="Output directory to save CSV & JSON results (default: current script directory or value forwarded by parent launcher)")
    p.add_argument("--self-test",
                   action="store_true",
                   help="Run KL sanity test to validate normalization scaling (PROJECT_NOTES.md section 1.1). Can also run standalone: python kl_sanity_test.py MODEL_ID")
    # Prism sidecar (shared decoder) controls
    p.add_argument("--prism",
                   default=os.environ.get("LOGOS_PRISM", "auto"),
                   choices=["auto", "on", "off"],
                   help="Prism sidecar mode: auto (default), on (require artifacts), off (disable)")
    p.add_argument("--prism-dir",
                   default="prisms",
                   help="Prism artifacts root directory (default: prisms under this script directory)")
    return p.parse_args()


# Parse once and promote to module-global so run_single_model and main can see it
CLI_ARGS = parse_cli()

def main():
    """Main function to launch separate processes for each model"""
    # If a model_id was provided, run in single-model mode
    if CLI_ARGS.model_id:
        success = run_single_model(CLI_ARGS.model_id)
        sys.exit(0 if success else 1)
    
    # Main process - launch subprocess for each model
    print(f"🎯 Starting experiment launcher for {len(CONFIRMED_MODELS)} models...")
    print("Each model will run in a separate process for clean memory isolation.")

    script_path = os.path.abspath(__file__)

    # Set up run-latest directory with automatic rotation
    run_dir = setup_run_latest_directory(os.path.dirname(script_path))

    # Create empty markdown files for evaluation reports
    print(f"📝 Creating empty evaluation markdown files...")
    for model_id in CONFIRMED_MODELS:
        clean_name = clean_model_name(model_id)
        eval_md_path = os.path.join(run_dir, f"evaluation-{clean_name}.md")
        with open(eval_md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Evaluation Report: {model_id}\n\n")
            f.write(f"*Run executed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

        print(f"   📄 Created: evaluation-{clean_name}.md")

    results = []
    launched_models = []
    
    for i, model_id in enumerate(CONFIRMED_MODELS, 1):
        print(f"\n{'='*80}")
        print(f"📋 Launching process {i}/{len(CONFIRMED_MODELS)}: {model_id}")
        print(f"{'='*80}")
        
        try:
            # Decide device per model (auto by default)
            if CLI_ARGS.device == "auto":
                sel = select_best_device(model_id)
                if sel is None:
                    print(f"⛔ Skipping {model_id}: no device fits (estimates)")
                    results.append((model_id, "SKIPPED_NO_FIT"))
                    continue
                dev, dtype, debug = sel
                print(f"📐 Decision for {model_id}: device={dev} dtype={dtype} est_peak={debug.get('est_peak')} avail={debug.get('available')}")
                chosen_device = dev
            else:
                chosen_device = CLI_ARGS.device

            # Launch subprocess
            cmd = [
                sys.executable,
                script_path,
                "--device", chosen_device,   # forward the per-model device
                "--out_dir", run_dir,        # ensure all subprocesses share same dir
            ]
            if CLI_ARGS.fp32_unembed:
                cmd.append("--fp32-unembed")   # forward the fp32-unembed flag
            if CLI_ARGS.keep_residuals:
                cmd.append("--keep-residuals") # forward the keep-residuals flag
            # Forward copy-collapse parameters
            cmd.extend(["--copy-threshold", str(CLI_ARGS.copy_threshold)])
            cmd.extend(["--copy-margin", str(CLI_ARGS.copy_margin)])
            cmd.append(model_id)
            
            result = subprocess.run(cmd, capture_output=False, text=True, check=False)

            if result.returncode == 0:
                print(f"✅ Process {i} completed successfully")
                results.append((model_id, "SUCCESS"))
                launched_models.append(model_id)
            else:
                print(f"❌ Process {i} failed with return code {result.returncode}")
                results.append((model_id, "FAILED"))
                
        except Exception as e:
            error_msg = f"Failed to launch subprocess for {model_id}: {str(e)}"
            print(f"❌ {error_msg}")
            results.append((model_id, f"LAUNCH_FAILED: {str(e)}"))
    
    # Summary
    print(f"\n{'='*80}")
    print("🎉 All model processes completed!")
    print(f"📁 Output files saved in: {run_dir}")
    
    print("\n📊 Results Summary:")
    for model_id, status in results:
        clean_name = clean_model_name(model_id)
        status_emoji = "✅" if status == "SUCCESS" else "❌"
        print(f"   {status_emoji} {clean_name}: {status}")
    
    print(f"\n📄 Expected output files:")
    for model_id in (launched_models if len(launched_models) > 0 else CONFIRMED_MODELS):
        clean_name = clean_model_name(model_id)
        print(f"   {os.path.join(run_dir, f'output-{clean_name}.json')}")
        print(f"   {os.path.join(run_dir, f'output-{clean_name}-records.csv')}")
        print(f"   {os.path.join(run_dir, f'output-{clean_name}-pure-next-token.csv')} ")
        print(f"   {os.path.join(run_dir, f'evaluation-{clean_name}.md')}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
