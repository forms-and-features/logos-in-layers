#!/usr/bin/env python3
"""
KL sanity test from PROJECT_NOTES.md section 1.1:
Test multiple layers: decode with γ=1 vs learned γ.
The KL between them should match KL between raw hidden states with/without γ.

This validates that the normalization scaling fixes are working correctly.
"""

import torch
import copy
from layers_core.norm_utils import detect_model_architecture, get_correct_norm_module, apply_norm_or_skip


def run_kl_sanity_test(model, tokenizer):
    """
    KL sanity test from PROJECT_NOTES.md section 1.1:
    Test multiple layers: decode with γ=1 vs learned γ.
    The KL between them should match KL between raw hidden states with/without γ.
    """
    print("\n" + "="*50)
    print("RUNNING KL SANITY TEST (Section 1.1)")
    print("="*50)
    
    test_prompt = "The capital of Germany is"
    tokens = tokenizer.encode(test_prompt, return_tensors="pt").to(model.device)
    
    architecture = detect_model_architecture(model)
    print(f"Architecture: {architecture}")
    
    n_layers = model.cfg.n_layers
    # Test at 25%, 50%, 75% depth and layer 0
    test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4]
    test_layers = [l for l in test_layers if l < n_layers]  # Remove invalid
    test_layers = sorted(list(set(test_layers)))  # Remove duplicates
    
    print(f"Testing layers: {test_layers} out of {n_layers} total layers")
    
    with torch.no_grad():
        # Get all hidden states
        outputs = model(tokens, output_hidden_states=True)
        all_failures = []
        
        for layer_idx in test_layers:
            print(f"\n--- Testing Layer {layer_idx} ---")
            
            # Get residual after this layer
            residual = outputs.hidden_states[layer_idx + 1]  # +1 because hidden_states[0] is embeddings
            
            # Get correct norm module for this layer
            norm_module = get_correct_norm_module(model, layer_idx, probe_after_block=True)
            print(f"Layer {layer_idx} norm module: {norm_module}")
            
            if norm_module is None:
                print(f"❌ Layer {layer_idx}: No norm module found - skipping")
                all_failures.append(f"Layer {layer_idx}: No norm module")
                continue
            
            # Test 1: Decode with learned γ
            normalized_learned = apply_norm_or_skip(residual, norm_module)
            logits_learned = model.lm_head(normalized_learned)
            
            # Test 2: Decode with γ=1 (create modified norm module)
            norm_module_unit = copy.deepcopy(norm_module)
            if hasattr(norm_module_unit, 'weight'):
                original_weight = norm_module_unit.weight.data.clone()
                norm_module_unit.weight.data.fill_(1.0)
            elif hasattr(norm_module_unit, 'scale'):
                original_weight = norm_module_unit.scale.data.clone()
                norm_module_unit.scale.data.fill_(1.0)
            else:
                print(f"❌ Layer {layer_idx}: Norm module has no learnable scale - skipping")
                all_failures.append(f"Layer {layer_idx}: No learnable scale")
                continue
            
            normalized_unit = apply_norm_or_skip(residual, norm_module_unit)
            logits_unit = model.lm_head(normalized_unit)
            
            # Test 3: Raw residuals with and without scaling (as reference)
            raw_resid = residual
            scaled_raw_resid = raw_resid * original_weight.to(raw_resid.device, dtype=raw_resid.dtype)
            
            # Compute KL divergences (use final token position)
            final_pos = -1
            
            kl_logits = torch.kl_div(
                torch.log_softmax(logits_unit[0, final_pos], dim=-1),
                torch.softmax(logits_learned[0, final_pos], dim=-1),
                reduction='sum'
            ).item()
            
            kl_raw = torch.kl_div(
                torch.log_softmax(model.lm_head(raw_resid)[0, final_pos], dim=-1),
                torch.softmax(model.lm_head(scaled_raw_resid)[0, final_pos], dim=-1),
                reduction='sum'
            ).item()
            
            print(f"  KL divergence (normalized logits): {kl_logits:.6f}")
            print(f"  KL divergence (raw residual):      {kl_raw:.6f}")
            print(f"  KL difference:                     {abs(kl_logits - kl_raw):.6f}")
            
            # The KL divergences should be approximately equal
            tolerance = 0.1
            if abs(kl_logits - kl_raw) < tolerance:
                print(f"  ✅ Layer {layer_idx}: KL divergences match - scaling is consistent")
            else:
                print(f"  ❌ Layer {layer_idx}: KL mismatch exceeds tolerance of {tolerance}")
                all_failures.append(f"Layer {layer_idx}: KL mismatch {abs(kl_logits - kl_raw):.6f}")
        
        # Overall result
        print(f"\n{'='*50}")
        if not all_failures:
            print("✅ ALL LAYERS PASS: Normalization scaling is correct across all tested depths")
            return True
        else:
            print("❌ SOME LAYERS FAILED:")
            for failure in all_failures:
                print(f"  - {failure}")
            print("This indicates normalization scaling issues that need attention!")
            return False


def main():
    """Run KL sanity test standalone"""
    import argparse
    from transformers import AutoTokenizer
    from transformer_lens import HookedTransformer
    
    parser = argparse.ArgumentParser(description="Run KL sanity test for normalization scaling")
    parser.add_argument("model_id", help="Model ID to test")
    parser.add_argument("--device", default="cpu", choices=["cuda", "mps", "cpu"], 
                        help="Device to run on")
    args = parser.parse_args()
    
    print(f"Loading model {args.model_id} on {args.device}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_id,
        device=args.device,
        torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    
    # Run the test
    success = run_kl_sanity_test(model, tokenizer)
    
    if success:
        print("\n🎉 KL sanity test PASSED - normalization scaling is correct!")
        exit(0)
    else:
        print("\n💥 KL sanity test FAILED - normalization scaling needs attention!")
        exit(1)


if __name__ == "__main__":
    main()
