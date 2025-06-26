# Interpretability Project - Development Notes

## Current State
- **001_layers_and_logits**: Complete layer-by-layer analysis with 4 models
- **File structure**: Reorganized from single script to proper experiment directory
- **Analysis**: Individual model reports + cross-model comparison (AI-generated)

## Key Technical Implementation Details

### RMSNorm vs LayerNorm Handling
**Problem**: All tested models use Pre-RMSNorm, not vanilla LayerNorm. Applying LayerNorm lens to RMSNorm creates artifacts.

**Solution**: 
```python
def is_safe_layernorm(norm_mod):
    return isinstance(norm_mod, nn.LayerNorm)

def apply_norm_or_skip(resid, norm_mod, layer_info=""):
    if isinstance(norm_mod, nn.LayerNorm):
        return norm_mod(resid)
    # Skip for RMSNorm to avoid distortion
    return resid
```

**Models tested**: All use Pre-RMSNorm (Meta, Mistral, Google, Alibaba)

### Memory Optimization for Large Models
**Problem**: `run_with_cache()` loads full activations, causing OOM on 9B models

**Solution**: Targeted caching with hooks
```python
def make_cache_hook(cache_dict):
    def cache_residual_hook(tensor, hook):
        # Only last token, move to CPU
        cache_dict[hook.name] = tensor[:, -1:, :].cpu().detach()
    return cache_residual_hook
```

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
├── run.py                  # Main script
├── evaluation-*.md         # Analysis results  
├── output-*.txt           # Raw outputs
├── prompt-*.txt           # Evaluation prompts
└── interpretability/      # AI-generated analysis
    └── XXX_experiment_name/
        └── evaluation-cross-model.md
```

### Toggle Patterns
```python
USE_NORM_LENS = True  # Keep for backward compatibility
# Raw mode still needed for activation patching
```

## AI Evaluation System

### Prompt Templates
- `prompt-single-model-evaluation.txt`: Individual model analysis
- `prompt-cross-model-evaluation.txt`: Comparative analysis
- `prompt-meta-evaluation.txt`: Methodology analysis

### Usage Pattern
1. Run experiment → generate `output-*.txt`
2. Feed outputs to AI with evaluation prompts
3. Generate `evaluation-*.md` files
4. Cross-model analysis in `interpretability/` subdirectory


### Known Issues
- Entropy values not comparable across models due to RMSNorm skipping
- Early layers show BPE noise artifacts (not harmful but clutters output)
- Temperature sweep only done on final output, not per-layer

## Philosophical Project Context
- **Goal**: Use interpretability to inform nominalism vs realism debate
- **Current evidence**: Layer-relative perspective (early=nominalist templates, late=realist concepts)

## User Context
- Software engineer, growing ML interpretability knowledge
- No formal ML background but learning through implementation
- Prefers systematic, reproducible experiments with clear documentation

