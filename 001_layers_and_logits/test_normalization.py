#!/usr/bin/env python3
"""
Unit test for normalization scaling fixes as specified in PROJECT_NOTES.md section 1.1.

Tests epsilon placement, architecture detection, and scaling consistency.
For comprehensive KL testing across multiple layers, see kl_sanity_test.py.
"""

import torch
import math
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from run import apply_norm_or_skip, get_correct_norm_module, detect_model_architecture

def test_normalization_scaling_synthetic():
    """
    Test that decoding with γ=1 vs learned γ gives consistent KL on synthetic data.
    This validates that our apply_norm_or_skip function works correctly.
    """
    print("Testing normalization scaling with synthetic data")
    
    # Create synthetic transformer-lens style model structure
    class MockModel:
        class MockConfig:
            n_layers = 2
            d_model = 64
            
        def __init__(self):
            self.cfg = self.MockConfig()
            self.blocks = [MockBlock(), MockBlock()]
            self.ln_final = MockRMSNorm(64)
    
    class MockBlock:
        def __init__(self):
            self.ln1 = MockRMSNorm(64)  
            self.ln2 = MockRMSNorm(64)
            self.mlp = "mock_mlp"  # Make it pre-norm
        
        def children(self):
            return [self.ln1, self.ln2, self.mlp]
    
    class MockRMSNorm(torch.nn.Module):
        def __init__(self, d_model, eps=1e-5):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(d_model) * 0.1 + 1.0)  # Near 1.0 with noise
            self.eps = eps
    
    model = MockModel()
    architecture = detect_model_architecture(model)
    print(f"Mock model architecture: {architecture}")
    
    # Test different layer indices and architectures
    test_cases = [
        (0, True),   # Layer 0, probe after block
        (0, False),  # Layer 0, probe before block  
        (1, True),   # Layer 1, probe after block
    ]
    
    all_passed = True
    
    for layer_idx, probe_after_block in test_cases:
        print(f"\nTesting layer {layer_idx}, probe_after_block={probe_after_block}")
        
        # Create synthetic residual
        batch_size, seq_len, d_model = 1, 5, 64
        residual = torch.randn(batch_size, seq_len, d_model)
        
        # Get the correct norm module
        norm_module = get_correct_norm_module(model, layer_idx, probe_after_block)
        if norm_module is None:
            print(f"  No norm module found for layer {layer_idx}")
            continue
            
        print(f"  Using norm module from: {norm_module}")
        
        # Test 1: Apply with learned weights
        normalized_learned = apply_norm_or_skip(residual, norm_module)
        
        # Test 2: Apply with unit weights  
        norm_module_unit = copy.deepcopy(norm_module)
        norm_module_unit.weight.data.fill_(1.0)
        normalized_unit = apply_norm_or_skip(residual, norm_module_unit)
        
        # Test 3: Manual scaling
        # First normalize with unit weights, then scale
        normalized_manual = normalized_unit * norm_module.weight.detach()
        
        # Compare results
        diff = torch.abs(normalized_learned - normalized_manual).max().item()
        print(f"  Max difference: {diff:.10f}")
        
        tolerance = 1e-6
        if diff < tolerance:
            print("  ✅ PASS: Scaling is consistent")
        else:
            print(f"  ❌ FAIL: Scaling inconsistency exceeds tolerance of {tolerance}")
            all_passed = False
    
    return all_passed

def test_architecture_aware_norm_selection():
    """
    Test that norm module selection works correctly for different architectures.
    """
    print("Testing architecture-aware norm module selection")
    
    # Mock pre-norm model (like Llama)
    class PreNormModel:
        class MockConfig:
            n_layers = 3
            model_name = "llama-test"
            
        def __init__(self):
            self.cfg = self.MockConfig()
            self.blocks = [MockBlock() for _ in range(3)]
            self.ln_final = MockRMSNorm(64)
    
    class PostNormModel:
        class MockConfig:
            n_layers = 3
            model_name = "gpt-j-test"
            
        def __init__(self):
            self.cfg = self.MockConfig()
            self.blocks = [PostNormBlock() for _ in range(3)]
            self.ln_final = MockLayerNorm(64)
    
    class MockBlock:
        def __init__(self):
            self.ln1 = MockRMSNorm(64)  
            self.attn = "mock_attention"
            self.ln2 = MockRMSNorm(64)
            self.mlp = "mock_mlp"  # MLP is last - pre-norm pattern
            self.hook_resid_pre = "mock_hook"
        
        def children(self):
            """Return children in pre-norm order - MLP is last"""
            return [self.ln1, self.attn, self.ln2, self.mlp]
    
    class PostNormBlock:
        """Mock post-norm block where the last child is a normalization layer"""
        def __init__(self):
            self.attn = "mock_attention"
            self.ln1 = MockLayerNorm(64)  # After attention
            self.mlp = "mock_mlp"
            self.ln2 = MockLayerNorm(64)  # After MLP - this is the last child
        
        def children(self):
            """Return children in order - ln2 should be last for post-norm detection"""
            return [self.attn, self.ln1, self.mlp, self.ln2]
    
    class MockRMSNorm(torch.nn.Module):
        def __init__(self, d_model, eps=1e-5):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(d_model))
            self.eps = eps
    
    class MockLayerNorm(torch.nn.LayerNorm):
        def __init__(self, d_model):
            super().__init__(d_model)
    
    # Test pre-norm model
    pre_model = PreNormModel()
    pre_arch = detect_model_architecture(pre_model)
    print(f"Pre-norm architecture detected: {pre_arch}")
    
    # Test cases for pre-norm
    pre_test_cases = [
        # (layer_idx, probe_after_block, expected_norm_source)
        (0, True, "next_block_ln1"),   # Should use block[1].ln1 
        (1, True, "next_block_ln1"),   # Should use block[2].ln1
        (2, True, "ln_final"),         # Should use ln_final (last layer)
        (0, False, "current_block_ln1"), # Should use block[0].ln1
    ]
    
    all_passed = True
    
    for layer_idx, probe_after_block, expected_source in pre_test_cases:
        norm_module = get_correct_norm_module(pre_model, layer_idx, probe_after_block)
        
        if expected_source == "next_block_ln1" and layer_idx < len(pre_model.blocks) - 1:
            expected_module = pre_model.blocks[layer_idx + 1].ln1
        elif expected_source == "ln_final":
            expected_module = pre_model.ln_final
        elif expected_source == "current_block_ln1":
            expected_module = pre_model.blocks[layer_idx].ln1
        else:
            expected_module = None
        
        if norm_module is expected_module:
            print(f"  ✅ PRE-NORM Layer {layer_idx}, after_block={probe_after_block}: correct norm module")
        else:
            print(f"  ❌ PRE-NORM Layer {layer_idx}, after_block={probe_after_block}: wrong norm module")
            print(f"     Expected: {expected_module}, Got: {norm_module}")
            all_passed = False
    
    # Test post-norm model  
    post_model = PostNormModel()
    post_arch = detect_model_architecture(post_model)
    print(f"Post-norm architecture detected: {post_arch}")
    
    # Critical assertion: detector must identify post-norm correctly
    if post_arch != "post_norm":
        print(f"❌ CRITICAL: Post-norm detector failed! Got '{post_arch}', expected 'post_norm'")
        all_passed = False
        return all_passed
    
    # Test cases for post-norm
    post_test_cases = [
        # (layer_idx, probe_after_block, expected_norm_source)
        (0, True, "current_block_ln2"),   # Should use block[0].ln2
        (1, True, "current_block_ln2"),   # Should use block[1].ln2
        (2, True, "current_block_ln2"),   # Should use block[2].ln2 (or ln_final)
        (0, False, "current_block_ln1"),  # Should use block[0].ln1
    ]
    
    for layer_idx, probe_after_block, expected_source in post_test_cases:
        norm_module = get_correct_norm_module(post_model, layer_idx, probe_after_block)
        
        if expected_source == "current_block_ln2":
            expected_module = getattr(post_model.blocks[layer_idx], 'ln2', None)
        elif expected_source == "current_block_ln1":
            expected_module = getattr(post_model.blocks[layer_idx], 'ln1', None)
        else:
            expected_module = None
        
        if norm_module is expected_module:
            print(f"  ✅ POST-NORM Layer {layer_idx}, after_block={probe_after_block}: correct norm module")
        else:
            print(f"  ❌ POST-NORM Layer {layer_idx}, after_block={probe_after_block}: wrong norm module")
            print(f"     Expected: {expected_module}, Got: {norm_module}")
            all_passed = False
    
    # Specific critical assertion for layer 0 post-norm
    norm_module_0 = get_correct_norm_module(post_model, 0, probe_after_block=True)
    if norm_module_0 is not post_model.blocks[0].ln2:
        print(f"❌ CRITICAL: Post-norm layer 0 uses wrong γ! Expected block[0].ln2, got {norm_module_0}")
        all_passed = False
    
    return all_passed

def test_epsilon_placement():
    """
    Test that epsilon is correctly placed inside the sqrt for RMSNorm.
    """
    print("Testing epsilon placement in RMSNorm")
    
    # Create synthetic data
    batch_size, seq_len, d_model = 1, 10, 64
    residual = torch.randn(batch_size, seq_len, d_model)
    
    # Create mock RMSNorm module
    class MockRMSNorm(torch.nn.Module):
        def __init__(self, d_model, eps=1e-5):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(d_model))
            self.eps = eps
    
    norm_module = MockRMSNorm(d_model)
    
    # Apply our corrected normalization
    normalized = apply_norm_or_skip(residual, norm_module)
    
    # Manually compute correct RMSNorm with eps inside sqrt
    expected_rms = torch.sqrt(residual.pow(2).mean(-1, keepdim=True) + norm_module.eps)
    expected_normalized = residual / expected_rms * norm_module.weight
    
    # Compare results
    diff = torch.abs(normalized - expected_normalized).max().item()
    print(f"Max difference from expected: {diff:.10f}")
    
    tolerance = 1e-6
    if diff < tolerance:
        print("✅ PASS: Epsilon placement is correct")
        return True
    else:
        print(f"❌ FAIL: Epsilon placement error exceeds tolerance of {tolerance}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("NORMALIZATION SCALING TESTS")
    print("=" * 60)
    
    # Test epsilon placement
    test1_pass = test_epsilon_placement()
    print()
    
    # Test architecture-aware norm selection
    test2_pass = test_architecture_aware_norm_selection()
    print()
    
    # Test scaling consistency with synthetic data
    test3_pass = test_normalization_scaling_synthetic()
    print()
    
    print("=" * 60)
    if test1_pass and test2_pass and test3_pass:
        print("🎉 ALL TESTS PASSED")
        exit(0)
    else:
        print("💥 SOME TESTS FAILED")
        exit(1)