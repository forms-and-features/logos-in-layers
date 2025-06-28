# Interpretability Project - Development Notes

## Current State
- **001_layers_and_logits**: Complete layer-by-layer analysis with 4 models
- **File structure**: Reorganized from single script to proper experiment directory
- **Analysis**: Individual model reports + cross-model comparison (AI-generated)

## Key Technical Implementation Details

### RMSNorm vs LayerNorm Handling
**Observation**: All tested checkpoints employ **Pre-RMSNorm**.  A naive LayerNorm lens therefore scales residuals incorrectly.

**Current implementation** (`run.py ≥ 2025-06-28`):
```python
def is_safe_layernorm(norm_mod):
    return isinstance(norm_mod, nn.LayerNorm)

def rms_lens(resid, gamma, eps=1e-5):
    rms = torch.sqrt(resid.pow(2).mean(-1, keepdim=True) + eps)
    return resid / rms * gamma

def apply_norm_or_skip(resid, norm_mod, layer_info=""):
    if isinstance(norm_mod, nn.LayerNorm):
        return norm_mod(resid)            # vanilla LN
    if "RMS" in type(norm_mod).__name__:
        eps   = getattr(norm_mod, "eps", 1e-5)
        gamma = _get_rms_scale(norm_mod) or torch.ones(resid.shape[-1], device=resid.device)
        return rms_lens(resid, gamma, eps) # RMS lens
    return resid                           # fallback (rare)
```

This applies a *true* RMS lens rather than skipping the operation.  All four reference models therefore use **normalised** residuals in the lens analysis.

### Memory Optimization for Large Models
**Problem**: `run_with_cache()` loads full activations, causing OOM on 9B models

**Solution**: Targeted caching with hooks
```python
def make_cache_hook(cache_dict):
    def cache_residual_hook(tensor, hook):
        # Store **full sequence** to enable per-position analysis,
        # then move to CPU in fp32 for memory safety.
        cache_dict[hook.name] = tensor.cpu().float().detach()
    return cache_residual_hook
```
If memory is a constraint, slice to `[:, -1:]` for last-token-only caching.

### Device/Precision Management
```python
model = HookedTransformer.from_pretrained(
    model_id,
    device="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
```

## Model-Specific Gotchas Found

### Qwen3-8B
- Template token "Answer" gets p≈0.74 in mid-layers (blocks 17-24)
- Placeholder tokens "____" spike later
- Strong template prior can override factual circuits

### Meta-Llama-3-8B  
- Junk token "ABCDEFGHIJKLMNOP" gets 7% probability (blocks 17-21)
- Likely memorized artifact or placeholder feature
- Should be activation-patched to test circuit function

### Mistral-7B-v0.1
- Newline token dominates instruction prompts
- Oscillation between Berlin and Washington in blocks 23-31
- Shows formatting bias and semantic interference

### Gemma-2-9B
- First 9 blocks deterministic on ':' (entropy≈0)
- Early over-confidence masks signal until layer 10
- Limited long-range integration in early layers

## Development Environment Notes

### Hardware Requirements (MacBook Pro M2 Max 64GB)
- **Memory**: 64GB barely sufficient for 9B models + analysis
- **GPU**: Metal acceleration essential (MPS detection works)
- **Storage**: ~50GB for model caching

### Library Stack
- **TransformerLens**: Better than tuned_lens for model support
- **No quantization**: Apple Silicon compatibility issues
- **Raw format only**: GGUF unsupported

## Code Organization Patterns

### Experiment Structure
```
XXX_experiment_name/
├── run.py                       # Main script
├── evaluation-*.md              # Per-model analyses  
├── evaluation-cross-model.md    # Cross-model analysis (if applicable)
├── output-*.json                # JSON metadata
├── output-*-records.csv         # Layer-wise records
├── prompt-*.txt                 # Evaluation prompts
└── interpretability/            # AI-generated analysis
    └── XXX_experiment_name/
```

### Toggle Patterns
```python
USE_NORM_LENS = True  # Keep for backward compatibility
# Raw mode still needed for activation patching
```

### Experiment Toggles & Options

The main experiment script (`001_layers_and_logits/run.py`) exposes several research switches that influence interpretability accuracy versus memory-footprint:

- `USE_NORM_LENS` *(bool, default **True**)* – apply the **normalization lens** before unembedding. Falls back gracefully for RMSNorm-only checkpoints via `apply_norm_or_skip`.
- `USE_FP32_UNEMBED` *(bool, default **True**)* – cast the unembedding matrix (and optional bias) to **float32** so we can resolve logit gaps < 1 e-5 that would vanish in fp16.
- **Residual cache device** – the `cache_residual_hook` moves **full-sequence** residuals to CPU fp32.  For GPU-only workflows comment out the `.cpu()` call; for memory-tight runs slice to `[:, -1:]` first.

These flags live near the top of `evaluate_model()` – keep them there so downstream notebooks can `sed`/patch without parsing the full file.

### Analysis Pipeline

1. `run.py` (per-model) writes `output-{model}.json` (metadata) and `output-{model}-records.csv` (layer records).
2. `prompt-single-model-evaluation.txt` → **LLM** summarises each run into `evaluation-{model}.md`.
3. `prompt-cross-model-evaluation.txt` consumes the per-model markdown and writes `evaluation-cross-model.md` in the same experiment directory.
4. The meta prompt (`prompt-meta-evaluation.txt`) critiques methodology and suggests next probes; results stored next to the cross-model report.

Keep all artefacts in-repo for provenance – the evaluation markdown is machine-read by later automation so headings must remain stable.

## AI Evaluation System

### Prompt Templates
- `prompt-single-model-evaluation.txt`: Individual model analysis
- `prompt-cross-model-evaluation.txt`: Comparative analysis
- `prompt-meta-evaluation.txt`: Methodology analysis

### Usage Pattern
1. Run experiment → generate `output-*.json` (metadata) **and** `output-*-records.csv` (layer records)
2. Feed outputs to AI with evaluation prompts
3. Generate `evaluation-*.md` files
4. Cross-model analysis is written alongside per-model evaluations in the experiment directory


## Philosophical Project Context
- **Goal**: Use interpretability to inform nominalism vs realism debate
- **Current evidence**: Layer-relative perspective (early=nominalist templates, late=realist concepts)

## User Context
- Software engineer, growing ML interpretability knowledge
- No formal ML background but learning through implementation
- Prefers systematic, reproducible experiments with clear documentation

