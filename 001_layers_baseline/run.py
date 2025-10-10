from transformer_lens import HookedTransformer
import torch
import torch.nn as nn
from datetime import datetime
import os
import sys
import json
import gc  # For garbage collection
import platform
from typing import Any, Dict, List, Optional

# --- deterministic bootstrap -------------------------------------------------
import random, numpy as np

SEED = 316
random.seed(SEED)
np.random.seed(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # harmless on CPU, required for CUDA; set before any CUDA checks
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.use_deterministic_algorithms(True)   # PyTorch 2.x+
torch.set_num_threads(1)  # optional; comment out if you need full CPU speed
try:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
except Exception:
    pass
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
    # Re-exported for backward compatibility with tests/tools expecting these on run.py
    detect_model_architecture,
    get_correct_norm_module,
    apply_norm_or_skip,
)
from layers_core.numerics import (
    bits_entropy_from_logits,
)
from layers_core.csv_io import (
    write_artifact_audit_csv,
    write_csv_files,
    write_milestones_csv,
    write_raw_lens_full_csv,
    write_raw_lens_window_csv,
    write_tuned_positions_csv,
    write_tuned_variants_csv,
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
    summarize_raw_lens_check,
)
from layers_core.windows import WindowManager
from layers_core.gold import (
    compute_gold_answer_info,
    compute_gold_answer_info_from_sequences,
    build_gold_alignment_entry,
    compute_gold_alignment_rate,
)
from layers_core.prism import load_prism_artifacts
from layers_core.head_transforms import detect_head_transforms
from layers_core.unembed import prepare_unembed_weights
from layers_core.lenses import NormLensAdapter, TunedLensAdapter
from layers_core.tuned_lens import load_tuned_lens
from layers_core.tuned_audit import build_tuned_audit_summary, build_provenance_snapshot
from layers_core.eval_pack import build_evaluation_pack
from layers_core.passes import run_prompt_pass
from layers_core.contexts import UnembedContext, PrismContext
from layers_core.probes import emit_test_prompts, emit_temperature_exploration
from layers_core.token_utils import make_decode_id
from layers_core.collapse_rules import (
    format_copy_strict_label,
    format_copy_soft_label,
    build_copy_ignore_mask,
)
from layers_core.temperature import fit_norm_temperatures
from layers_core.skip_sanity import evaluate_skip_layers
from layers_core.summaries import (
    build_unified_lens_metrics,
    tuned_rotation_vs_temp_attribution,
    compute_confirmed_semantics,
    compute_lens_artifact_score,
    classify_norm_trajectory,
)

def clean_model_name(model_id):
    """Extract clean model name for filename"""
    # Remove organization prefix (everything before last '/')
    clean_name = model_id.split('/')[-1]
    return clean_name

def _vprint(*args, **kwargs):
    """Verbose print controlled by CLI_ARGS.quiet (defaults to verbose)."""
    cli = globals().get('CLI_ARGS', None)
    try:
        if not (hasattr(cli, 'quiet') and getattr(cli, 'quiet')):
            print(*args, **kwargs)
    except Exception:
        print(*args, **kwargs)



def run_experiment_for_model(model_id, output_files, config: ExperimentConfig):
    """Run the complete experiment for a single model and write results to files"""

    json_data_tuned_outer = None
    tuned_provenance_outer = None
    tuned_diag_info_outer = None
    tuned_audit_data_outer = None
    evaluation_pack: Optional[Dict[str, Any]] = None
    evaluation_pack_milestones: List[Dict[str, Any]] = []
    evaluation_pack_artifacts: List[Dict[str, Any]] = []

    def evaluate_model():
        """The actual experiment code - all prints go to console"""
        nonlocal json_data_tuned_outer, tuned_provenance_outer, tuned_diag_info_outer
        nonlocal tuned_audit_data_outer
        nonlocal evaluation_pack, evaluation_pack_milestones, evaluation_pack_artifacts
        _vprint(f"\n{'='*60}")
        _vprint(f"EVALUATING MODEL: {model_id}")
        _vprint(f"{'='*60}")
        
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
        _vprint(f"Loading model on [{device}] ...")
        
        # Clear any existing CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Set batch_first=False for models that need it (TransformerLens expectation)
        os.environ['TRANSFORMERS_BATCH_FIRST'] = 'False'
            
        # Load model
        try:
            _vprint("Loading directly to target device…")
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
            # Fallback: load on CPU
            model = HookedTransformer.from_pretrained_no_processing(
                model_id,
                device="cpu",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            # Try move to requested device; if it fails, stay on CPU
            if device != "cpu":
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _vprint(f"Moving model to {device}...")
                    model = model.to(device)
                except Exception as move_e:
                    print(f"Move to {device} failed: {move_e}. Staying on CPU for this run.")
                    device = "cpu"
            
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
                    _vprint("✅ Self-test passed - continuing with normal evaluation...\n")
            except ImportError as e:
                print(f"❌ Could not import KL sanity test: {e}")
                return {"error": "Self-test import failed"}
        
        # Clear cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Debug: inspect model parameter devices
        unique_param_devices = {p.device for p in model.parameters()}
        _vprint(f"[DEBUG MODEL] Unique parameter devices: {unique_param_devices}")
        
        # Toggle for using normalized lens (recommended for accurate interpretation)
        USE_NORM_LENS = True
        
        # Toggle for FP32 unembedding (recommended for research-grade precision)
        # Prevents under-resolving logit gaps < 1e-5 with minimal memory overhead.
        AUTO_FP32_UNEMBED = should_auto_promote_unembed(dtype)

        # Prepare analysis-only unembedding weights (no mutation of model params)
        orig_unembed_dtype = model.unembed.W_U.dtype
        if config.fp32_unembed and orig_unembed_dtype != torch.float32:
            _vprint(f"🔬 CLI: Forcing FP32 shadow unembed weights for analysis (was {orig_unembed_dtype})")
        elif AUTO_FP32_UNEMBED and orig_unembed_dtype != torch.float32:
            _vprint(f"🔬 Using FP32 shadow unembed weights for analysis (was {orig_unembed_dtype})")
        analysis_W_U, analysis_b_U = prepare_unembed_weights(
            model.unembed.W_U,
            getattr(model.unembed, 'b_U', None),
            force_fp32=(config.fp32_unembed or AUTO_FP32_UNEMBED),
        )
        UNEMBED_DTYPE = analysis_W_U.dtype

        # Per-device cache for unembedding matmul
        _unembed_cache = {"device": None, "W": None, "b": None}
        unembed_ctx = UnembedContext(
            W=analysis_W_U,
            b=analysis_b_U,
            force_fp32=(config.fp32_unembed or AUTO_FP32_UNEMBED),
            cache=_unembed_cache,
        )
        bias_diag = {"present": False, "l2_norm": 0.0, "max_abs": 0.0}
        if analysis_b_U is not None:
            try:
                bias_tensor = analysis_b_U.detach().to(dtype=torch.float32)
                l2_norm = float(torch.norm(bias_tensor).item())
                max_abs = float(torch.max(torch.abs(bias_tensor)).item())
                bias_diag["present"] = bool(l2_norm > 0.0)
                bias_diag["l2_norm"] = l2_norm
                bias_diag["max_abs"] = max_abs
            except Exception as exc:
                bias_diag["error"] = str(exc)

        context_prompt = "Give the city name only, plain text. The capital of Germany is called simply"
        ground_truth = "Berlin"  # For display/comparison
        # Stylistic filler ablation (001_LAYERS_BASELINE_PLAN §1.9): drop the adverb
        context_prompt_nf = "Give the city name only, plain text. The capital of Germany is called"
        # Negative control (001_LAYERS_BASELINE_PLAN §1.8)
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
            _ln2_type = type(model.blocks[0].ln2).__name__
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
        clean_name = clean_model_name(model_id)

        # Track a last-layer consistency snapshot (lens vs model final head)
        last_layer_consistency = None

        # Detect simple final-head transforms exposed by the model/config (factorized)
        head_scale_cfg, head_softcap_cfg = detect_head_transforms(model)

        diag = {
            "type": "diagnostics",
            "model": model_id,
            "device": device,
            "use_norm_lens": USE_NORM_LENS,
            "use_fp32_unembed": AUTO_FP32_UNEMBED,
            "unembed_dtype": str(UNEMBED_DTYPE),
            "first_block_ln1_type": first_block_ln1_type,
            "final_ln_type": final_ln_type,
            "layernorm_bias_fix": layernorm_bias_fix,
            "norm_alignment_fix": norm_alignment_fix,
            "layer0_norm_fix": layer0_norm_fix,
            "mixed_precision_fix": mixed_precision_fix,
            "layer0_position_info": layer0_position_info,
            "context_prompt": context_prompt,
            "target_prediction": "first unseen token (likely 'Berlin')",
            "surface_diagnostics_config": {
                "delta": 0.05,
                "gamma": 0.02,
                "K": 50,
                "tau": 0.33,
                "semantic_margin": {
                    "delta_abs": 0.002,
                },
            },
            "answer_margin_unit": "logit",
        }
        diag["unembed_bias"] = bias_diag
        diag_flags = diag.setdefault("flags", {})
        gold_alignment_entries: List[Dict[str, Any]] = []
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
            # Raw-vs-Norm sanity block (001_LAYERS_BASELINE_PLAN §1.4)
            "raw_lens_check": init_raw_lens_check(RAW_LENS_MODE),
        }
        try:
            cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        except Exception:
            cudnn_version = None
        env_info = {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn": cudnn_version,
            "device": device,
            "dtype_compute": str(dtype),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
            "seed": SEED,
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        json_data["provenance"] = {"env": env_info}
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
        json_data_prism = {"records": [], "pure_next_token_records": [], "copy_flag_columns": []}
        IMPORTANT_WORDS = ["Germany", "Berlin", "capital", "Answer", "word", "simply"]

        # prompt_id and prompt_variant for tagging records across passes
        current_prompt_id = "pos"
        current_prompt_variant = "orig"

        # Copy detector configuration (strict + soft windowed defaults)
        raw_soft_ks = getattr(config, "copy_soft_window_ks", (1, 2, 3))
        copy_soft_window_ks = tuple(sorted({int(k) for k in raw_soft_ks if int(k) > 0}))
        if not copy_soft_window_ks:
            copy_soft_window_ks = (max(1, int(getattr(config, "copy_window_k", 1))),)
        copy_soft_threshold = float(getattr(config, "copy_soft_threshold", 0.33))
        raw_soft_extra = getattr(config, "copy_soft_thresholds_extra", ())
        copy_soft_thresholds_extra = tuple(sorted({float(th) for th in raw_soft_extra}))
        copy_soft_thresholds_extra_filtered = tuple(
            th for th in copy_soft_thresholds_extra if abs(th - copy_soft_threshold) >= 1e-9
        )
        copy_strict_label = format_copy_strict_label(config.copy_threshold)
        # Threshold sweep for strict copy (001_LAYERS_BASELINE_PLAN §1.23)
        copy_strict_tau_list = (0.70, 0.80, 0.90, 0.95)
        copy_strict_labels_extra = [
            format_copy_strict_label(tau) for tau in copy_strict_tau_list
        ]
        copy_soft_labels = {k: format_copy_soft_label(k, copy_soft_threshold) for k in copy_soft_window_ks}
        copy_soft_extra_labels = {
            (k, th): format_copy_soft_label(k, th)
            for th in copy_soft_thresholds_extra_filtered
            for k in copy_soft_window_ks
        }
        # unique preserve order
        copy_flag_columns = []
        for lab in [copy_strict_label, *copy_strict_labels_extra]:
            if lab not in copy_flag_columns:
                copy_flag_columns.append(lab)
        copy_flag_columns.extend([copy_soft_labels[k] for k in sorted(copy_soft_labels)])
        copy_flag_columns.extend([
            label
            for (k, th), label in sorted(copy_soft_extra_labels.items(), key=lambda kv: (kv[0][1], kv[0][0]))
        ])
        json_data["copy_flag_columns"] = list(copy_flag_columns)
        json_data_prism["copy_flag_columns"] = list(copy_flag_columns)

        tuned_mode = str(getattr(CLI_ARGS, "tuned", "auto")).lower()
        if tuned_mode not in ("auto", "off", "require"):
            tuned_mode = "auto"

        tuned_adapter = None
        tuned_translator = None
        tuned_provenance = None
        tuned_diag_info = {"status": "disabled" if tuned_mode == "off" else "missing", "mode": tuned_mode}

        diag["copy_soft_config"] = {
            "threshold": copy_soft_threshold,
            "window_ks": list(copy_soft_window_ks),
            "extra_thresholds": list(copy_soft_thresholds_extra_filtered),
        }

        # Rolling window manager to isolate copy-detection state per lens and per variant
        window_mgr = WindowManager(getattr(config, "copy_window_k", 1), extra_window_ks=copy_soft_window_ks)

        # Baseline norm lens adapter (behavior-preserving)
        norm_lens = NormLensAdapter()

        # Norm-lens temperature calibration (CPU fallback)
        norm_temp_taus = None
        try:
            calibration_prompts = [context_prompt, context_prompt_nf, context_prompt_ctl]
            norm_temp_taus = fit_norm_temperatures(
                model,
                calibration_prompts,
                norm_lens,
                unembed_ctx,
            )
            diag["tau_norm_per_layer"] = [float(t) for t in norm_temp_taus]
        except Exception as exc:
            diag["tau_norm_per_layer_error"] = str(exc)

        tuned_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned_lenses", prism_clean)
        tuned_diag_info["path"] = tuned_dir
        if tuned_mode != "off":
            if os.path.isdir(tuned_dir):
                try:
                    tuned_translator, tuned_provenance = load_tuned_lens(tuned_dir, map_location=device)
                    tuned_translator = tuned_translator.to(device)
                    tuned_translator.eval()
                    tuned_adapter = TunedLensAdapter(translator=tuned_translator, strict=False)
                    tuned_diag_info["status"] = "loaded"
                except Exception as exc:
                    tuned_diag_info["status"] = "error"
                    tuned_diag_info["error"] = str(exc)
                    tuned_translator = None
                    tuned_adapter = None
                    tuned_provenance = None
                    if tuned_mode == "require":
                        raise RuntimeError(f"Tuned lens load failed for {model_id}: {exc}") from exc
            else:
                if tuned_mode == "require":
                    raise RuntimeError(
                        f"Tuned lens not found for {prism_clean}; generate with tuned_lens_fit.py first"
                    )
                else:
                    _vprint(f"⚠️ Tuned lens not found for {prism_clean}; run tuned_lens_fit.py --model-id {model_id}")

        tuned_spec = None
        json_data_tuned = None
        window_mgr_tuned = None
        if tuned_adapter is not None:
            window_mgr_tuned = WindowManager(getattr(config, "copy_window_k", 1), extra_window_ks=copy_soft_window_ks)
            json_data_tuned = {
                "records": [],
                "pure_next_token_records": [],
                "copy_flag_columns": list(copy_flag_columns),
            }
            tuned_spec = {
                "adapter": tuned_adapter,
                "json_data": json_data_tuned,
                "window_manager": window_mgr_tuned,
            }

        # Robust single-id decode (tensor or int)
        decode_id = make_decode_id(model.tokenizer)
        copy_mask_info = None
        try:
            vocab_size = getattr(model.cfg, "d_vocab", None)
            if vocab_size is None:
                vocab_size = getattr(model.cfg, "n_vocab", None)
            if vocab_size is None and model.tokenizer is not None:
                vocab_size = len(model.tokenizer)
            if vocab_size is not None:
                copy_mask_info = build_copy_ignore_mask(decode_id, int(vocab_size))
        except Exception as exc:
            diag_flags["copy_mask_error"] = str(exc)
        if copy_mask_info is not None:
            diag["copy_mask"] = copy_mask_info

        # Residual accessor now shared in layers_core.hooks.get_residual_safely

        
        # Tokenize the context prompt (without "Answer:" to avoid teacher-forcing)
        tokens = model.to_tokens(context_prompt)      # let Accelerate move it

        # Gold-token alignment (001_LAYERS_BASELINE_PLAN §1.7): prefer tokenizer path
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
                    decode_id=decode_id,
                    answer_str=ground_truth,
                )
            except (KeyError, AttributeError, ValueError, TypeError) as e:
                print(f"Warning: tokenizer-based gold alignment failed; trying TL tokens fallback: {e}")
                try:
                    ctx_ids_try = model.to_tokens(context_prompt)[0].tolist()
                    ctx_ans_ws_try = model.to_tokens(context_prompt + " " + ground_truth)[0].tolist()
                    ctx_ans_ns_try = model.to_tokens(context_prompt + ground_truth)[0].tolist()
                    dec = decode_id if hasattr(model, 'tokenizer') else None
                    gold_info = compute_gold_answer_info_from_sequences(
                        ctx_ids_try, ctx_ans_ws_try, ctx_ans_ns_try,
                        pieces_k=4,
                        convert_ids_to_tokens=getattr(getattr(model, 'tokenizer', None), 'convert_ids_to_tokens', None),
                        decode_id=dec,
                        answer_str=ground_truth,
                    )
                except (KeyError, AttributeError, ValueError, TypeError) as e2:
                    print(f"Warning: TL tokens fallback for gold alignment failed: {e2}")
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
                except Exception as e2:
                    print(f"Unexpected error during gold alignment fallbacks: {e2}")
                    raise

        # Provide ctx ids for copy detector and first answer id for metrics
        primary_alignment_entry = build_gold_alignment_entry("pos", "orig", gold_info)
        gold_alignment_entries.append(primary_alignment_entry)
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
                    decode_id=decode_id,
                    answer_str=control_ground_truth,
                )
            except (KeyError, AttributeError, ValueError, TypeError) as e:
                print(f"Warning: tokenizer-based control gold alignment failed; marking unresolved: {e}")
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
            except Exception as e:
                print(f"Unexpected error during control gold alignment: {e}")
                raise
        control_alignment_entry = build_gold_alignment_entry("ctl", "orig", gold_info_ctl)
        gold_alignment_entries.append(control_alignment_entry)
        ctx_ids_ctl = gold_info_ctl.get("ctx_ids", [])
        first_ans_id_ctl = gold_info_ctl.get("first_id", None)
        
        # Begin capturing residual streams
        with torch.no_grad():
            # MEMORY EFFICIENT: Use targeted caching instead of run_with_cache
            # (removed human-readable progress print)
            
            # Storage for only the residual streams we need
            residual_cache = {}
            # Build hook closure
            cache_hook = build_cache_hook(residual_cache)
            # Attach hooks and record handles
            hooks, _has_pos_embed = attach_residual_hooks(model, cache_hook)
            
            try:
                # Run forward pass with targeted hooks
                logits = model(tokens)

                # Cache final head reference distribution once (for KL-to-final)
                final_logits = logits[0, -1, :].float()
                final_probs = torch.softmax(final_logits, dim=0)
                # Representation-drift baseline direction (001_LAYERS_BASELINE_PLAN §1.5)
                _final_norm = torch.norm(final_logits) + 1e-12
                # Robust argmax with finite check
                if not torch.isfinite(final_probs).all():
                    print("Warning: non-finite values in final_probs; argmax may be unreliable")
                _arg = torch.argmax(final_probs)
                
                # Debug: print device placements of cached activations and tokens
                for name, t in residual_cache.items():
                    _vprint(f"[DEBUG CACHE] {name} device: {t.device}")
                _vprint(f"[DEBUG TOKENS] tokens device: {tokens.device}")
                
                # Show top predictions at different layers
                _vprint(f"\nLayer-by-layer analysis of context: '{context_prompt}'")
                _vprint("→ Predicting the first unseen token (what comes after the context)")
                if USE_NORM_LENS:
                    # Check if we'll actually be applying norms
                    if hasattr(model, 'blocks') and len(model.blocks) > 0:
                        first_norm = model.blocks[0].ln1
                        norm_type = type(first_norm).__name__
                        
                        if isinstance(first_norm, nn.LayerNorm):
                            _vprint("Using NORMALIZED residual stream (LayerNorm applied - more accurate)")
                        elif 'RMS' in norm_type:
                            if _get_rms_scale(first_norm) is None:
                                _vprint("Using NORMALIZED residual stream (RMS, no learnable scale)")
                            else:
                                _vprint("Using NORMALIZED residual stream (RMS + learned scale)")
                        else:
                            _vprint("Using RAW residual stream (unsupported normalization, skipping to avoid distortion)")
                    else:
                        _vprint("Using RAW residual stream (no normalization layers found)")
                else:
                    _vprint("Using RAW residual stream (normalization disabled)")
                _vprint("Note: Shown probabilities are from full softmax (calibrated and comparable)")
                _vprint(f"copy-collapse: top-1 ID-window in prompt & p>{config.copy_threshold}")
                if config.keep_residuals:
                    _vprint("💾 Saving residual tensors to disk (--keep-residuals enabled)")
                _vprint("-" * 60)
                
                # Use pass runner for the positive/orig prompt
                prism_ctx = PrismContext(stats=prism_stats, Q=prism_Q, active=prism_active)
                pass_summary, last_layer_consistency, detected_architecture, prism_diag = run_prompt_pass(
                    model=model,
                    context_prompt=context_prompt,
                    ground_truth=ground_truth,
                    prompt_id=current_prompt_id,
                    prompt_variant=current_prompt_variant,
                    window_manager=window_mgr,
                    norm_lens=norm_lens,
                    unembed_ctx=unembed_ctx,
                    copy_threshold=config.copy_threshold,
                    copy_margin=config.copy_margin,
                    entropy_collapse_threshold=getattr(config, 'entropy_collapse_threshold', 1.0),
                    top_k_record=TOP_K_RECORD,
                    top_k_verbose=TOP_K_VERBOSE,
                    keep_residuals=config.keep_residuals,
                    out_dir=config.out_dir,
                    copy_soft_threshold=copy_soft_threshold,
                    copy_soft_window_ks=copy_soft_window_ks,
                    copy_strict_label=copy_strict_label,
                    copy_soft_labels=copy_soft_labels,
                    copy_soft_extra_labels=copy_soft_extra_labels,
                    RAW_LENS_MODE=RAW_LENS_MODE,
                    json_data=json_data,
                    json_data_prism=json_data_prism,
                    prism_ctx=prism_ctx,
                    decode_id_fn=decode_id,
                    ctx_ids_list=ctx_ids,
                    first_ans_token_id=first_ans_id,
                    important_words=IMPORTANT_WORDS,
                    head_scale_cfg=head_scale_cfg,
                    head_softcap_cfg=head_softcap_cfg,
                    clean_model_name=clean_model_name(model_id),
                    enable_raw_lens_sampling=True,
                    tuned_spec=tuned_spec,
                    norm_temp_taus=norm_temp_taus,
                    copy_strict_thresholds=copy_strict_tau_list,
                )
                diag.update(pass_summary)
                semantic_margin_info = pass_summary.get("semantic_margin") or {}
                if diag.get("L_semantic") is not None:
                    margin_flag = semantic_margin_info.get("margin_ok_at_L_semantic_norm")
                    if margin_flag is False:
                        diag_flags.setdefault("rank_only_near_uniform", True)
                diag["gold_alignment"] = {
                    "ok": primary_alignment_entry["ok"],
                    "status": primary_alignment_entry["status"],
                    "variant": primary_alignment_entry.get("variant"),
                    "first_id": primary_alignment_entry.get("first_id"),
                    "answer_ids": primary_alignment_entry.get("answer_ids"),
                    "pieces": primary_alignment_entry.get("pieces"),
                }
                L_copy_orig = diag.get("L_copy")
                L_sem_orig = diag.get("L_semantic")
                json_data["diagnostics"] = diag
                try:
                    repeat_diag = (diag.get("repeatability") or {})
                    if repeat_diag.get("status") == "ok":
                        max_dev = float(repeat_diag.get("max_rank_dev", 0.0) or 0.0)
                        flip_rate = float(repeat_diag.get("top1_flip_rate", 0.0) or 0.0)
                        if max_dev > 5.0 or flip_rate > 0.02:
                            diag_flags["repeatability_variance_high"] = True
                    elif repeat_diag.get("status") == "skipped":
                        diag_flags.setdefault("repeatability_skipped", repeat_diag.get("reason"))
                    elif repeat_diag.get("status") == "unavailable":
                        diag_flags.setdefault("repeatability_unavailable", True)
                except Exception:
                    pass
                if last_layer_consistency is not None:
                    json_data["diagnostics"]["last_layer_consistency"] = last_layer_consistency
                # KL_temp percentile snapshots
                try:
                    if norm_temp_taus:
                        layer_rec = {
                            int(r.get("layer")): r
                            for r in json_data.get("pure_next_token_records", [])
                            if r.get("prompt_id") == current_prompt_id and r.get("prompt_variant") == current_prompt_variant
                        }
                        total_layers = int(model.cfg.n_layers)
                        for pct, frac in ((25, 0.25), (50, 0.50), (75, 0.75)):
                            li = max(0, min(total_layers, int(round(total_layers * frac))))
                            rec = layer_rec.get(li)
                            diag[f"kl_to_final_bits_norm_temp@{pct}%"] = None if rec is None else {
                                "layer": li,
                                "value": rec.get("kl_to_final_bits_norm_temp"),
                            }
                except Exception:
                    pass
                # Merge any runner-provided Prism diag deltas (e.g., placement_error)
                try:
                    if isinstance(diag.get("prism_summary"), dict) and isinstance(prism_diag, dict):
                        diag["prism_summary"].update(prism_diag)
                except Exception:
                    pass
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
            _vprint("=" * 60)
            _vprint("ACTUAL MODEL PREDICTION (for comparison):")

            # final_logits already computed above; ensure float
            final_logits = final_logits.float()
            # Clamp top-k to vocab size to support small-vocab test stubs
            _k_final = min(20, int(final_logits.shape[-1]))
            _, final_top_indices = torch.topk(final_logits, _k_final, largest=True, sorted=True)
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
            
            # Emit additional probing records for test prompts via probes module
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
            json_data["test_prompts"] = emit_test_prompts(model, test_prompts, decode_id)

            # Emit temperature exploration via probes module (memory-safe)
            temp_test_prompt = "Give the city name only, plain text. The capital of Germany is called simply"
            json_data["temperature_exploration"] = emit_temperature_exploration(model, temp_test_prompt, decode_id)
        
        _vprint("=== END OF INSPECTING ==============\n")

        # ---------------- Ablation pass: no-filler (001_LAYERS_BASELINE_PLAN §1.9) --------
        # Run a separate forward pass on the positive prompt without the stylistic filler
        current_prompt_id = "pos"
        current_prompt_variant = "no_filler"
        # Variant windows are reset inside the pass runner
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
                    decode_id=decode_id,
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
        nf_alignment_entry = build_gold_alignment_entry("pos", "no_filler", gold_info_nf)
        gold_alignment_entries.append(nf_alignment_entry)
        ctx_ids_nf = gold_info_nf.get("ctx_ids", [])
        first_ans_id_nf = gold_info_nf.get("first_id", None)
        prism_ctx = PrismContext(stats=prism_stats, Q=prism_Q, active=prism_active)
        pass_summary_nf, _, _, prism_diag_nf = run_prompt_pass(
            model=model,
            context_prompt=context_prompt_nf,
            ground_truth=ground_truth,
            prompt_id=current_prompt_id,
            prompt_variant=current_prompt_variant,
            window_manager=window_mgr,
            norm_lens=norm_lens,
            unembed_ctx=unembed_ctx,
            copy_threshold=config.copy_threshold,
            copy_margin=config.copy_margin,
            entropy_collapse_threshold=getattr(config, 'entropy_collapse_threshold', 1.0),
            top_k_record=TOP_K_RECORD,
            top_k_verbose=TOP_K_VERBOSE,
            keep_residuals=False,
            out_dir=config.out_dir,
            copy_soft_threshold=copy_soft_threshold,
            copy_soft_window_ks=copy_soft_window_ks,
            copy_strict_label=copy_strict_label,
            copy_soft_labels=copy_soft_labels,
            copy_soft_extra_labels=copy_soft_extra_labels,
            RAW_LENS_MODE=RAW_LENS_MODE,
            json_data=json_data,
            json_data_prism=json_data_prism,
            prism_ctx=prism_ctx,
            decode_id_fn=decode_id,
            ctx_ids_list=ctx_ids_nf,
            first_ans_token_id=first_ans_id_nf,
            important_words=IMPORTANT_WORDS,
                    head_scale_cfg=head_scale_cfg,
                    head_softcap_cfg=head_softcap_cfg,
                    clean_model_name=clean_model_name(model_id),
                    enable_raw_lens_sampling=False,
                    tuned_spec=tuned_spec,
                    norm_temp_taus=norm_temp_taus,
                    copy_strict_thresholds=copy_strict_tau_list,
                )
        try:
            if isinstance(diag.get("prism_summary"), dict) and isinstance(prism_diag_nf, dict) and prism_diag_nf.get("placement_error"):
                diag["prism_summary"]["placement_error_nf"] = prism_diag_nf["placement_error"]
        except Exception:
            pass
        L_copy_nf = pass_summary_nf.get("L_copy")
        L_sem_nf = pass_summary_nf.get("L_semantic")
        json_data["ablation_summary"] = {
            "L_copy_orig": L_copy_orig,
            "L_sem_orig": L_sem_orig,
            "L_copy_nf": L_copy_nf,
            "L_sem_nf": L_sem_nf,
            "delta_L_copy": (None if (L_copy_orig is None or L_copy_nf is None) else (L_copy_nf - L_copy_orig)),
            "delta_L_sem": (None if (L_sem_orig is None or L_sem_nf is None) else (L_sem_nf - L_sem_orig)),
        }

        # ---------------- Control pass (001_LAYERS_BASELINE_PLAN §1.8) --------------------
        prism_ctx = PrismContext(stats=prism_stats, Q=prism_Q, active=prism_active)
        _pass_summary_ctl, _, _, prism_diag_ctl = run_prompt_pass(
            model=model,
            context_prompt=context_prompt_ctl,
            ground_truth=control_ground_truth,
            prompt_id="ctl",
            prompt_variant="orig",
            window_manager=window_mgr,
            norm_lens=norm_lens,
            unembed_ctx=unembed_ctx,
            copy_threshold=config.copy_threshold,
            copy_margin=config.copy_margin,
            entropy_collapse_threshold=getattr(config, 'entropy_collapse_threshold', 1.0),
            top_k_record=TOP_K_RECORD,
            top_k_verbose=TOP_K_VERBOSE,
            keep_residuals=False,
            out_dir=config.out_dir,
            copy_soft_threshold=copy_soft_threshold,
            copy_soft_window_ks=copy_soft_window_ks,
            copy_strict_label=copy_strict_label,
            copy_soft_labels=copy_soft_labels,
            copy_soft_extra_labels=copy_soft_extra_labels,
            RAW_LENS_MODE=RAW_LENS_MODE,
            json_data=json_data,
            json_data_prism=json_data_prism,
            prism_ctx=prism_ctx,
            decode_id_fn=decode_id,
            ctx_ids_list=ctx_ids_ctl,
            first_ans_token_id=first_ans_id_ctl,
            important_words=IMPORTANT_WORDS,
                    head_scale_cfg=head_scale_cfg,
                    head_softcap_cfg=head_softcap_cfg,
                    clean_model_name=clean_model_name(model_id),
                    control_ids=(first_ans_id_ctl, first_ans_id),
                    enable_raw_lens_sampling=False,
                    tuned_spec=tuned_spec,
                    norm_temp_taus=norm_temp_taus,
                    copy_strict_thresholds=copy_strict_tau_list,
                )
        try:
            if isinstance(diag.get("prism_summary"), dict) and isinstance(prism_diag_ctl, dict) and prism_diag_ctl.get("placement_error"):
                diag["prism_summary"]["placement_error_ctl"] = prism_diag_ctl["placement_error"]
        except Exception:
            pass

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
            "gold_alignment": {
                "ok": control_alignment_entry["ok"],
                "status": control_alignment_entry["status"],
                "variant": control_alignment_entry.get("variant"),
                "first_id": control_alignment_entry.get("first_id"),
                "answer_ids": control_alignment_entry.get("answer_ids"),
                "pieces": control_alignment_entry.get("pieces"),
            },
        }
        json_data["control_summary"] = {
            "first_control_margin_pos": first_pos,
            "max_control_margin": max_margin,
        }

        try:
            rate = compute_gold_alignment_rate(gold_alignment_entries)
        except Exception:
            rate = None
        if rate is not None:
            diag["gold_alignment_rate"] = rate
            if rate < 1.0:
                diag_flags["gold_alignment_partial"] = True
        else:
            diag["gold_alignment_rate"] = None
        try:
            diag["gold_alignment_prompts"] = gold_alignment_entries
        except Exception:
            pass

        # ---------------- Unified sidecar summaries (001_LAYERS_BASELINE_PLAN §1.21) -----
        tuned_metrics = None
        try:
            total_layers = int(model.cfg.n_layers)
        except Exception:
            total_layers = None

        # Prism metrics (if sidecar is active)
        try:
            if total_layers is not None and prism_active and isinstance(json_data_prism, dict):
                base_records = json_data.get("pure_next_token_records", [])
                prism_records = json_data_prism.get("pure_next_token_records", [])
                metrics_prism = build_unified_lens_metrics(
                    baseline_records=base_records,
                    alt_records=prism_records,
                    n_layers=total_layers,
                    alt_label="prism",
                    prompt_id="pos",
                    prompt_variant="orig",
                )
                try:
                    diag.setdefault("prism_summary", {})["metrics"] = metrics_prism
                except Exception:
                    pass
        except Exception as exc:
            try:
                diag.setdefault("prism_summary", {})["metrics_error"] = str(exc)
            except Exception:
                pass

        # Tuned-lens metrics (if tuned adapter/data present)
        tuned_records = []
        try:
            if total_layers is not None and json_data_tuned is not None:
                tuned_records = json_data_tuned.get("pure_next_token_records", [])
                tuned_metrics = build_unified_lens_metrics(
                    baseline_records=json_data.get("pure_next_token_records", []),
                    alt_records=tuned_records,
                    n_layers=total_layers,
                    alt_label="tuned",
                    prompt_id="pos",
                    prompt_variant="orig",
                )
        except Exception:
            tuned_metrics = None

        tuned_provenance_snapshot = build_provenance_snapshot(tuned_provenance) if tuned_provenance is not None else None
        tuned_audit_summary = None
        tuned_audit_data = None
        tuned_is_calibration_only = None
        tuned_preferred_hint = None
        if tuned_spec is not None:
            audit_payload = tuned_spec.get("audit_data")
            tuned_audit_summary = build_tuned_audit_summary(audit_payload)
            tuned_audit_data = audit_payload
            tuned_audit_data_outer = audit_payload
            if isinstance(tuned_audit_summary, dict):
                tuned_is_calibration_only = tuned_audit_summary.get("tuned_is_calibration_only")
                tuned_preferred_hint = tuned_audit_summary.get("preferred_semantics_lens_hint")

        if tuned_adapter is not None and tuned_spec is not None and json_data_tuned is not None:
            tuned_diag_info = {**tuned_diag_info, "summaries": tuned_spec.get("summaries", [])}
        diag["tuned_lens"] = tuned_diag_info
        if tuned_adapter is not None and tuned_spec is not None and json_data_tuned is not None:
            tuned_block = {
                "status": tuned_diag_info.get("status"),
                "path": tuned_diag_info.get("path"),
                "summaries": tuned_spec.get("summaries", []),
                "provenance": tuned_provenance,
            }
            if tuned_provenance_snapshot is not None:
                tuned_block["provenance_snapshot"] = tuned_provenance_snapshot
            if tuned_metrics is not None:
                tuned_block["summary"] = {"metrics": tuned_metrics}
            if tuned_audit_summary is not None:
                tuned_block["audit_summary"] = tuned_audit_summary
            # Attribution and prefer_tuned gate (001_LAYERS_BASELINE_PLAN §1.26)
            try:
                attr = tuned_rotation_vs_temp_attribution(
                    baseline_records=[r for r in json_data.get("pure_next_token_records", []) if r.get("prompt_id") == "pos" and r.get("prompt_variant") == "orig"],
                    tuned_records=[r for r in tuned_records if r.get("prompt_id") == "pos" and r.get("prompt_variant") == "orig"],
                    n_layers=int(model.cfg.n_layers),
                )
                tuned_block["attribution"] = attr
            except Exception:
                pass
            json_data["tuned_lens"] = tuned_block
        else:
            json_data["tuned_lens"] = tuned_diag_info

        # Mirror key tuned summaries into diagnostics for quick access
        try:
            prim = (tuned_spec.get("summaries", []) if tuned_spec is not None else [])
            if prim:
                s0 = prim[0]
                for k, v in s0.items():
                    if isinstance(k, str) and k.endswith("_tuned"):
                        diag[k] = v
        except Exception:
            pass

        # Skip-layers sanity (advisory; not an acceptance gate)
        skip_results = {}
        if tuned_adapter is not None:
            try:
                skip_prompts = [
                    (context_prompt, first_ans_id),
                    (context_prompt_nf, first_ans_id_nf),
                    (context_prompt_ctl, first_ans_id_ctl),
                ]
                skip_results = evaluate_skip_layers(model, tuned_adapter, unembed_ctx, skip_prompts)
            except Exception as exc:
                diag["skip_layers_sanity_error"] = str(exc)
        tuned_reg = False
        if skip_results:
            diag["skip_layers_sanity"] = skip_results
            d2 = skip_results.get("m=2")
            if d2 is not None and d2 > 0.20:
                tuned_reg = True
                diag["tuned_lens_skip_warning"] = d2
        diag["tuned_lens_regression"] = tuned_reg

        # ---------------- Confirmed semantics & lens artefact score ------------
        try:
            L_norm = diag.get("L_semantic")
            conf = compute_confirmed_semantics(
                baseline_records=[r for r in json_data.get("pure_next_token_records", []) if r.get("prompt_id") == "pos" and r.get("prompt_variant") == "orig"],
                raw_full_rows=[r for r in json_data.get("raw_lens_full_records", []) if r.get("prompt_id") == "pos" and r.get("prompt_variant") == "orig"],
                tuned_records=[r for r in ((json_data_tuned or {}).get("pure_next_token_records", [])) if r.get("prompt_id") == "pos" and r.get("prompt_variant") == "orig"],
                L_semantic_norm=L_norm,
                delta_window=2,
            )
            diag["confirmed_semantics"] = conf
            if conf.get("L_semantic_confirmed") is not None:
                confirmed_layer = conf.get("L_semantic_confirmed")
                diag["L_semantic_confirmed"] = confirmed_layer
                try:
                    margin_block = diag.setdefault("semantic_margin", {})
                    margin_ok_layer = margin_block.get("L_semantic_margin_ok_norm")
                    if confirmed_layer is not None and margin_ok_layer is not None and int(confirmed_layer) == int(margin_ok_layer):
                        margin_block["L_semantic_confirmed_margin_ok_norm"] = confirmed_layer
                    else:
                        margin_block.setdefault("L_semantic_confirmed_margin_ok_norm", None)
                except Exception:
                    pass
        except Exception as e:
            try:
                diag["confirmed_semantics_error"] = str(e)
            except Exception:
                pass

        try:
            full_block = diag.get("raw_lens_full") or {}
            js_stats = (full_block.get("js_divergence_percentiles") or {})
            l1_stats = (full_block.get("l1_prob_diff_percentiles") or {})
            score_info = compute_lens_artifact_score(
                pct_layers_kl_ge_1=full_block.get("pct_layers_kl_ge_1.0"),
                pct_layers_kl_ge_0_5=full_block.get("pct_layers_kl_ge_0.5"),
                n_norm_only=int(full_block.get("n_norm_only_semantics_layers") or 0),
                max_kl_bits=full_block.get("max_kl_norm_vs_raw_bits"),
                js_p50=js_stats.get("p50"),
                l1_p50=l1_stats.get("p50"),
            )
            if full_block:
                full_block["score"] = score_info
                diag["raw_lens_full"] = full_block
            diag.setdefault("config", {})["lens_artifact_score"] = {
                "weights": {"pct_ge_1.0": 0.6, "norm_only": 0.3, "max_kl_scaled": 0.1},
                "thresholds": {"low": 0.2, "high": 0.5},
                "v2_addends": {"js_p50": 0.1, "l1_p50": 0.05},
            }
        except Exception as e:
            try:
                diag.setdefault("raw_lens_full", {})["score_error"] = str(e)
            except Exception:
                pass

        if not env_info.get("deterministic_algorithms", True):
            diag_flags["nondeterministic"] = True

        norm_spike = False
        try:
            per_layer_norm = (diag.get("normalization_provenance") or {}).get("per_layer", [])
            trajectory = classify_norm_trajectory(
                per_layer_norm,
                sem_layer=diag.get("L_semantic"),
            )
            if trajectory is not None:
                diag["norm_trajectory"] = trajectory
                norm_spike = bool(trajectory.get("n_spikes", 0))
            else:
                diag["norm_trajectory"] = None
        except Exception:
            diag["norm_trajectory"] = None
        if norm_spike:
            diag_flags["normalization_spike"] = True

        numeric_issue = False
        numeric_health = diag.get("numeric_health") or {}
        flagged_layers = numeric_health.get("layers_flagged") or []
        limits: list[int] = []
        for key in ("L_copy", "L_semantic"):
            val = diag.get(key)
            if isinstance(val, int):
                limits.append(int(val))
        for entry in flagged_layers:
            layer_idx = entry.get("layer")
            if not isinstance(layer_idx, int):
                continue
            if not limits:
                numeric_issue = True
                break
            if any(layer_idx <= limit for limit in limits):
                numeric_issue = True
                break
        if numeric_issue:
            diag_flags["numeric_health_caution"] = True

        if "copy_mask" not in diag:
            diag_flags.setdefault("copy_mask_missing", True)
        if not diag.get("layer_map"):
            diag_flags.setdefault("layer_map_missing", True)

        # ---------------- Measurement guidance (001_LAYERS_BASELINE_PLAN §1.22 & §1.28) ---
        try:
            warn_high_last_layer_kl = False
            try:
                warn_high_last_layer_kl = bool(
                    ((json_data.get("diagnostics") or {}).get("last_layer_consistency") or {}).get(
                        "warn_high_last_layer_kl"
                    )
                )
            except Exception:
                warn_high_last_layer_kl = False

            # Raw‑lens overall risk
            lens_artifact_risk = None
            try:
                lens_artifact_risk = (
                    ((json_data.get("raw_lens_check") or {}).get("summary") or {}).get("lens_artifact_risk")
                )
            except Exception:
                lens_artifact_risk = None
            high_lens_artifact_risk = (lens_artifact_risk == "high")

            # Windowed norm‑only semantics layers
            norm_only_layers = []
            try:
                norm_only_layers = (
                    ((json_data.get("diagnostics") or {}).get("raw_lens_window") or {}).get(
                        "norm_only_semantics_layers"
                    )
                ) or []
                if not isinstance(norm_only_layers, (list, tuple)):
                    norm_only_layers = []
            except Exception:
                norm_only_layers = []
            has_norm_only_semantics = bool(norm_only_layers)

            reasons = []
            if warn_high_last_layer_kl:
                reasons.append("warn_high_last_layer_kl")
            if has_norm_only_semantics:
                reasons.append("norm_only_semantics_window")
            if high_lens_artifact_risk:
                reasons.append("high_lens_artifact_risk")
            try:
                tier = ((json_data.get("diagnostics") or {}).get("raw_lens_full") or {}).get("score", {}).get("tier")
                if tier == "high":
                    reasons.append("high_lens_artifact_score")
            except Exception:
                pass
            if diag_flags.get("repeatability_variance_high"):
                reasons.append("repeatability_variance_high")
            if diag_flags.get("gold_alignment_partial"):
                reasons.append("gold_alignment_partial")
            if diag_flags.get("nondeterministic"):
                reasons.append("nondeterministic_env")
            if diag_flags.get("normalization_spike"):
                reasons.append("normalization_spike")
            if diag_flags.get("numeric_health_caution"):
                reasons.append("numeric_health_caution")
            if diag_flags.get("copy_mask_missing"):
                reasons.append("copy_mask_missing")
            if "copy_mask_error" in diag_flags:
                reasons.append("copy_mask_error")
            if diag_flags.get("layer_map_missing"):
                reasons.append("layer_map_missing")
            if diag_flags.get("rank_only_near_uniform"):
                reasons.append("rank_only_near_uniform")
            if tuned_is_calibration_only:
                reasons.append("tuned_is_calibration_only")

            reasons = list(dict.fromkeys(reasons))

            prefer_ranks = bool(reasons)
            suppress_abs_probs = prefer_ranks

            mg = {
                "prefer_ranks": prefer_ranks,
                "suppress_abs_probs": suppress_abs_probs,
                "reasons": reasons,
                "notes": "Family-level head calibration; treat probabilities comparatively only within model.",
            }
            try:
                prefer_tuned_gate = bool(((json_data.get("tuned_lens") or {}).get("attribution") or {}).get("prefer_tuned", False))
            except Exception:
                prefer_tuned_gate = False
            preferred_reporting = "tuned" if prefer_tuned_gate else "norm"
            if tuned_preferred_hint == "tuned_for_calibration_only":
                preferred_reporting = "norm"
            mg["preferred_lens_for_reporting"] = preferred_reporting
            if tuned_preferred_hint:
                mg["preferred_semantics_lens_hint"] = tuned_preferred_hint
            try:
                use_conf = bool(((json_data.get("diagnostics") or {}).get("L_semantic_confirmed") is not None))
                mg["use_confirmed_semantics"] = use_conf
            except Exception:
                pass
            extra_notes = []
            if diag_flags.get("nondeterministic"):
                extra_notes.append("Run uses nondeterministic kernels; prefer rank comparisons.")
            extra_notes.append("Prism is diagnostic-only; treat probabilities comparatively within model.")
            mg["notes_append"] = " ".join(extra_notes)
            json_data["measurement_guidance"] = mg
        except Exception:
            # Best-effort; absence should not fail the run
            pass

        try:
            evaluation_pack, evaluation_pack_milestones, evaluation_pack_artifacts = build_evaluation_pack(
                model_name=clean_name,
                n_layers=getattr(model.cfg, "n_layers", None),
                json_data=json_data,
                json_data_tuned=json_data_tuned,
                diag=diag,
                measurement_guidance=json_data.get("measurement_guidance"),
                tuned_audit_summary=tuned_audit_summary,
                tuned_audit_data=tuned_audit_data,
                clean_name=clean_name,
            )
            json_data["evaluation_pack"] = evaluation_pack
        except Exception as exc:
            evaluation_pack = None
            evaluation_pack_milestones = []
            evaluation_pack_artifacts = []
            try:
                json_data.setdefault("evaluation_pack_error", str(exc))
            except Exception:
                pass

        json_data_tuned_outer = json_data_tuned
        tuned_provenance_outer = tuned_provenance
        tuned_diag_info_outer = tuned_diag_info

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

        # Persist gold-answer transparency (001_LAYERS_BASELINE_PLAN §1.7)
        json_data["gold_answer"] = {
            "string": gold_info.get("string"),
            "pieces": gold_info.get("pieces", []),
            "first_id": gold_info.get("first_id"),
            "answer_ids": gold_info.get("answer_ids", []),
            "variant": gold_info.get("variant", "unknown"),
        }
        
        # Clean up model to free memory (though process will end anyway)
        if tuned_translator is not None:
            try:
                tuned_translator.to("cpu")
            except Exception:
                pass
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
        clean_name = clean_model_name(model_id)

        # Write CSV files FIRST (they need the full record lists)
        write_csv_files(json_data, csv_filepath, pure_csv_filepath, TOP_K_VERBOSE)
        # Prism sidecar CSVs (auto/on): only if artifacts were loaded and records exist
        try:
            prism_buf = json_data.get("prism_sidecar")
            prism_summary = (json_data.get("diagnostics") or {}).get("prism_summary") or {}
            if prism_summary.get("compatible") and prism_buf and (prism_buf["records"] or prism_buf["pure_next_token_records"]):
                out_dir = os.path.dirname(csv_filepath)
                csv_prism = os.path.join(out_dir, f"output-{clean_name}-records-prism.csv")
                pure_prism = os.path.join(out_dir, f"output-{clean_name}-pure-next-token-prism.csv")
                write_csv_files(prism_buf, csv_prism, pure_prism, TOP_K_VERBOSE)
                _vprint(f"✅ Prism Records CSV saved to: {csv_prism}")
                _vprint(f"✅ Prism Pure next-token CSV saved to: {pure_prism}")
            elif getattr(CLI_ARGS, "prism", "auto") != "off":
                # Single-line notice in auto mode when missing/incompatible
                err = prism_summary.get("error")
                if err:
                    _vprint(f"ℹ️ Prism sidecar disabled: {err}")
                else:
                    _vprint("ℹ️ Prism sidecar not written (no artifacts or empty buffers)")
        except Exception as e:
            print(f"⚠️ Failed to write Prism sidecar CSVs: {e}")

        window_records = json_data.get("raw_lens_window_records") or []
        if window_records:
            try:
                window_csv_path = os.path.join(
                    os.path.dirname(csv_filepath),
                    f"output-{clean_name}-pure-next-token-rawlens-window.csv",
                )
                write_raw_lens_window_csv(window_records, window_csv_path)
                _vprint(f"✅ Raw vs Norm window CSV saved to: {window_csv_path}")
            except Exception as e:
                print(f"⚠️ Failed to write raw-lens window CSV: {e}")

        full_rows = json_data.get("raw_lens_full_records") or []
        if full_rows:
            try:
                full_csv_path = os.path.join(
                    os.path.dirname(csv_filepath),
                    f"output-{clean_name}-pure-next-token-rawlens.csv",
                )
                write_raw_lens_full_csv(full_rows, full_csv_path)
                _vprint(f"✅ Raw vs Norm full CSV saved to: {full_csv_path}")
            except Exception as e:
                print(f"⚠️ Failed to write raw-lens full CSV: {e}")

        if json_data_tuned_outer is not None and (json_data_tuned_outer["records"] or json_data_tuned_outer["pure_next_token_records"]):
            tuned_records_path = os.path.join(os.path.dirname(csv_filepath), f"output-{clean_name}-records-tuned.csv")
            tuned_pure_path = os.path.join(os.path.dirname(csv_filepath), f"output-{clean_name}-pure-next-token-tuned.csv")
            try:
                write_csv_files(json_data_tuned_outer, tuned_records_path, tuned_pure_path, TOP_K_VERBOSE)
                _vprint(f"✅ Tuned Records CSV saved to: {tuned_records_path}")
                _vprint(f"✅ Tuned Pure next-token CSV saved to: {tuned_pure_path}")
            except Exception as e:
                print(f"⚠️ Failed to write tuned lens CSVs: {e}")

            audit_rows = (tuned_audit_data_outer or {}).get("variant_rows") if isinstance(tuned_audit_data_outer, dict) else []
            if audit_rows:
                try:
                    variants_path = os.path.join(
                        os.path.dirname(csv_filepath),
                        f"output-{clean_name}-pure-next-token-tuned-variants.csv",
                    )
                    write_tuned_variants_csv(audit_rows, variants_path)
                    _vprint(f"✅ Tuned variants CSV saved to: {variants_path}")
                except Exception as e:
                    print(f"⚠️ Failed to write tuned variants CSV: {e}")

            positional_rows = (tuned_audit_data_outer or {}).get("positional_rows") if isinstance(tuned_audit_data_outer, dict) else []
            if positional_rows:
                try:
                    positions_path = os.path.join(
                        os.path.dirname(csv_filepath),
                        f"output-{clean_name}-positions-tuned-audit.csv",
                    )
                    write_tuned_positions_csv(positional_rows, positions_path)
                    _vprint(f"✅ Tuned positional audit CSV saved to: {positions_path}")
                except Exception as e:
                    print(f"⚠️ Failed to write tuned positional audit CSV: {e}")

        if evaluation_pack_milestones:
            try:
                milestones_path = os.path.join(
                    os.path.dirname(csv_filepath),
                    f"output-{clean_name}-milestones.csv",
                )
                write_milestones_csv(evaluation_pack_milestones, milestones_path)
                _vprint(f"✅ Milestones CSV saved to: {milestones_path}")
            except Exception as e:
                print(f"⚠️ Failed to write milestones CSV: {e}")

        if evaluation_pack_artifacts:
            try:
                artifact_path = os.path.join(
                    os.path.dirname(csv_filepath),
                    f"output-{clean_name}-artifact-audit.csv",
                )
                write_artifact_audit_csv(evaluation_pack_artifacts, artifact_path)
                _vprint(f"✅ Artifact audit CSV saved to: {artifact_path}")
            except Exception as e:
                print(f"⚠️ Failed to write artifact audit CSV: {e}")

        # Strip bulky per-token records from JSON to keep it compact
        json_data_compact = {k: v for k, v in json_data.items()
                             if k not in ("records", "pure_next_token_records", "prism_sidecar", "raw_lens_window_records", "raw_lens_full_records")}

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
    _vprint(f"\n{'='*80}")
    _vprint(f"🚀 Starting subprocess for model: {model_id}")
    _vprint(f"{'='*80}")
    
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
            _vprint("📐 Device decision:")
            _vprint(f"   device={dev} dtype={dtype} est_peak={debug_info.get('est_peak')}B available={debug_info.get('available')}B")

        # Run the experiment
        raw_soft_ks = getattr(CLI_ARGS, "copy_soft_window_ks", [1, 2, 3])
        if not isinstance(raw_soft_ks, (list, tuple, set)):
            raw_soft_ks = [raw_soft_ks]
        soft_window_ks_tuple = tuple(sorted({int(k) for k in raw_soft_ks if int(k) > 0}))
        if not soft_window_ks_tuple:
            base_k = int(getattr(CLI_ARGS, "copy_window_k", 1))
            soft_window_ks_tuple = (max(1, base_k),)
        raw_soft_thresh_extra = getattr(CLI_ARGS, "copy_soft_thresh_list", [])
        if not isinstance(raw_soft_thresh_extra, (list, tuple, set)):
            raw_soft_thresh_extra = [raw_soft_thresh_extra]
        soft_thresh_extra_tuple = tuple(sorted({float(th) for th in raw_soft_thresh_extra}))

        cfg = ExperimentConfig(
            device=chosen_device,
            fp32_unembed=CLI_ARGS.fp32_unembed,
            keep_residuals=CLI_ARGS.keep_residuals,
            copy_threshold=CLI_ARGS.copy_threshold,
            copy_margin=CLI_ARGS.copy_margin,
            copy_window_k=int(getattr(CLI_ARGS, "copy_window_k", 1)),
            copy_soft_threshold=float(getattr(CLI_ARGS, "copy_soft_thresh", 0.33)),
            copy_soft_window_ks=soft_window_ks_tuple,
            copy_soft_thresholds_extra=soft_thresh_extra_tuple,
            out_dir=out_dir,
            self_test=CLI_ARGS.self_test,
        )
        _data = run_experiment_for_model(model_id, (meta_filepath, csv_filepath, pure_csv_filepath), cfg)

        if CLI_ARGS.self_test:
            _vprint("✅ Self-test complete. No artifacts written (by design).")
        else:
            _vprint(f"✅ Experiment complete. JSON metadata saved to: {meta_filepath}")
            _vprint(f"✅ Records CSV saved to: {csv_filepath}")
            _vprint(f"✅ Pure next-token CSV saved to: {pure_csv_filepath}")
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

from types import SimpleNamespace

# Provide default CLI args for test and programmatic usage; launcher overrides
CLI_ARGS = SimpleNamespace(
    device="auto",
    fp32_unembed=False,
    keep_residuals=False,
    copy_threshold=0.95,
    copy_margin=0.10,
    model_id=None,
    out_dir=None,
    self_test=False,
    quiet=False,
    prism=os.environ.get("LOGOS_PRISM", "auto"),
    prism_dir="prisms",
    copy_soft_thresh=0.33,
    copy_soft_window_ks=[1, 2, 3],
    copy_soft_thresh_list=[],
    tuned=os.environ.get("LOGOS_TUNED", "auto"),
)

if __name__ == "__main__":
    import sys
    # Ensure the running module is also available as 'run' to avoid a fresh import in launcher
    sys.modules.setdefault("run", sys.modules[__name__])
    # Provide a friendly help that mentions standalone self-test usage
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print("Layer-by-layer logit-lens sweep (worker)\n")
        print("Flags are parsed by the launcher. Common flags:")
        print("  --device {auto|cuda|mps|cpu}")
        print("  --fp32-unembed  --keep-residuals  --copy-threshold  --copy-margin  --self-test")
        print("\nSelf-test: validates normalization scaling (001_LAYERS_BASELINE_PLAN §1.1).\n"
              "Can also run standalone: python kl_sanity_test.py MODEL_ID")
        sys.exit(0)
    import launcher as _launcher
    _launcher.main()
