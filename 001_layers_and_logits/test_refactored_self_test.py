#!/usr/bin/env python3
"""
Test that the refactored self-test functionality works correctly.
"""

def test_imports():
    """Test that all imports work correctly"""
    try:
        # Test that kl_sanity_test can import from run.py
        from kl_sanity_test import run_kl_sanity_test
        print("✅ kl_sanity_test imports successfully")
        
        # Test that the kl_sanity_test module has the needed functions
        from run import detect_model_architecture, get_correct_norm_module, apply_norm_or_skip
        print("✅ kl_sanity_test can access run.py functions")
        
        return True
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_module_structure():
    """Test that the module has expected structure"""
    import kl_sanity_test
    
    # Check that main function exists for standalone usage
    if hasattr(kl_sanity_test, 'main'):
        print("✅ kl_sanity_test has main() for standalone usage")
    else:
        print("❌ kl_sanity_test missing main() function")
        return False
    
    # Check that run_kl_sanity_test function exists
    if hasattr(kl_sanity_test, 'run_kl_sanity_test'):
        print("✅ kl_sanity_test has run_kl_sanity_test() function")
    else:
        print("❌ kl_sanity_test missing run_kl_sanity_test() function")
        return False
    
    return True

def test_help_text():
    """Test that help text mentions both usage modes"""
    import subprocess
    import sys
    import os
    
    try:
        run_py = os.path.join(os.path.dirname(__file__), "run.py")
        result = subprocess.run([sys.executable, run_py, "--help"], 
                              capture_output=True, text=True, timeout=10)
        if "kl_sanity_test.py" in result.stdout:
            print("✅ run.py help text mentions standalone kl_sanity_test.py")
            return True
        else:
            print("❌ run.py help text doesn't mention standalone usage")
            return False
    except Exception as e:
        print(f"❌ Help text test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing refactored self-test functionality...")
    
    all_passed = True
    all_passed &= test_imports()
    all_passed &= test_module_structure()
    all_passed &= test_help_text()
    
    if all_passed:
        print("\n🎉 All refactoring tests passed!")
        exit(0)
    else:
        print("\n💥 Some refactoring tests failed!")
        exit(1)
