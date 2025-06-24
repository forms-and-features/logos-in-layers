# Interpretability Project Notes

## Project Goal
Experiment with interpretability of open-weights LLMs using systematic layer-by-layer analysis to understand internal computation patterns.

## Completed Experiments

### 000: Basic Chat (`000_basic_chat/run.py`)
- **Status**: Complete - basic chat interface for model testing
- **Purpose**: Simple interface for testing model responses and getting familiar with different models

### 001: Layer-by-Layer Logit Analysis (`001_layers_and_logits/run.py`)
- **Status**: Complete with comprehensive analysis in `001_layers_and_logits/analyses.md`
- **Models Tested**: Qwen3-8B, Meta-Llama-3-8B, Mistral-7B-v0.1, Gemma-2-9B
- **Key Technical Achievement**: LayerNorm lens implementation for accurate residual stream analysis
- **Sampling Strategy**: Key layers at 0, n_layers//6, n_layers//3, n_layers//2, 2*n_layers//3, 5*n_layers//6, n_layers-1

## Technical Implementation Notes

### LayerNorm Lens Implementation
**Critical insight**: Models never see raw residual streams. Each layer applies LayerNorm before processing, and `ln_final` is applied before unembedding.

#### Implementation Details:
- **RAW mode** (`USE_NORM_LENS = False`): Analyzes `resid_post` directly - useful for activation patching and causal interventions
- **NORMALIZED mode** (`USE_NORM_LENS = True`): Applies LayerNorm before unembedding - shows actual model decision-making
- **Layer 0 handling**: Uses `cache["embed"]` instead of `resid_post`
- **Final layer handling**: Uses `model.ln_final()` for proper normalization
- **Intermediate layers**: Uses `model.blocks[layer + 1].ln1()` to apply the LayerNorm that the next block would see

#### Why This Matters:
- Eliminates artificial confidence inflation seen in raw analysis
- Perfect alignment with actual model behavior
- Reveals true cognitive transitions between processing modes
- Essential for understanding calibration mechanisms

### Model Loading and Compatibility
- **Library**: TransformerLens (switched from tuned_lens for superior model support)
- **Device Management**: Automatic MPS detection for Apple Silicon
- **Memory Optimization**: Half-precision (float16) loading
- **Known Limitations**: GGUF files unsupported, extremely large models limited by hardware

### Sampling and Analysis Strategy
- **Prompt Format**: Q&A format ("Question: ... Answer:") more reliable than completion across models
- **Layer Sampling**: Strategic sampling across depth percentages for cross-model comparison
- **Additional Probing**: Directional bias testing, temperature exploration, alternative prompts
- **Probability Precision**: 6 decimal places for accurate confidence measurement

## Development Environment
- **Hardware**: MacBook Pro M2 Max 64GB
- **OS**: macOS (Metal GPU acceleration)
- **Python Environment**: Virtual environment with requirements.txt
- **Authentication**: Hugging Face CLI for gated model access

## Next Development Steps

### Immediate Technical Priorities
1. **Attention Pattern Analysis**: Implement attention head analysis using TransformerLens
   - Focus on layers 40-85% depth where semantic resolution occurs
   - Investigate attention to different token types (entities, relations, punctuation)

2. **Circuit Analysis**: Use TransformerLens advanced features for mechanistic analysis
   - Identify specific attention heads responsible for factual retrieval
   - Analyze MLPs vs attention contributions to prediction changes

3. **Activation Patching**: Implement causal interventions
   - Test hypotheses about layer functions discovered in 001 experiment
   - Use RAW mode for proper intervention on actual model computation

### Code Organization Guidelines
- **Directory Structure**: Each experiment gets its own directory `XXX_experiment_name/`
- **Numbering Convention**: `XXX_experiment_name/run.py` + supporting files
- **Results Storage**: Keep detailed numerical results in markdown files within experiment directories
- **Supporting Files**: Store prompts, configurations, and output files alongside the main script
- **Code Focus**: Scripts should be clean, reproducible, with minimal hardcoded values
- **Toggle Switches**: Maintain `USE_NORM_LENS` toggle for backward compatibility with activation patching

## Technical Lessons Learned

### Library Selection
- **TransformerLens**: Superior model support, comprehensive features, but slower than specialized libraries
- **Quantization**: Avoided due to Apple Silicon compatibility issues
- **Model Format**: Raw transformer format required, GGUF unsupported

### Hardware Optimization
- **Memory Management**: 64GB barely sufficient for 9B models with analysis overhead
- **GPU Acceleration**: Metal Performance Shaders essential for reasonable speed
- **Storage**: Model caching requires significant disk space planning

### Analysis Best Practices
- **Normalization**: Always use LayerNorm lens for interpretability analysis
- **Raw Mode**: Reserve for activation patching and causal interventions only
- **Cross-Model Comparison**: Use depth percentages rather than absolute layer numbers
- **Temperature Exploration**: Essential for understanding true vs calibrated confidence

## Context for Future Development
- **User Profile**: Software engineer with no Python background
- **Hardware**: MacBook Pro M2 Max 64GB
- **Focus**: Systematic, reproducible experiments with clear technical documentation

