#!/usr/bin/env python3
"""
Test that the refactored self-test functionality works correctly.
"""

import _pathfix  # noqa: F401


def test_imports():
    """Test that all imports work correctly"""
    # Test that kl_sanity_test can import from run.py
    from kl_sanity_test import run_kl_sanity_test  # noqa: F401
    # Test that the kl_sanity_test module has the needed functions
    from run import detect_model_architecture, get_correct_norm_module, apply_norm_or_skip  # noqa: F401
    # If imports succeed, assertions hold; no return needed.


def test_module_structure():
    """Test that the module has expected structure"""
    import kl_sanity_test

    # Check that main function exists for standalone usage
    assert hasattr(kl_sanity_test, 'main')
    # Check that run_kl_sanity_test function exists
    assert hasattr(kl_sanity_test, 'run_kl_sanity_test')


def test_help_text():
    """Test that help text mentions both usage modes"""
    import subprocess
    import sys
    import os

    run_py = os.path.join(os.path.dirname(__file__), "..", "run.py")
    result = subprocess.run([sys.executable, run_py, "--help"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "kl_sanity_test.py" in result.stdout


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
