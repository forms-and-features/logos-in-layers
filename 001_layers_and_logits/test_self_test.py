#!/usr/bin/env python3
"""
Test the --self-test CLI flag to validate KL sanity test works.
"""

import subprocess
import sys

def test_self_test_flag():
    """Test that --self-test flag works without errors"""
    print("Testing --self-test flag...")
    
    try:
        # Run the script with --self-test flag on a small model
        cmd = [
            sys.executable, "run.py", 
            "mistralai/Mistral-7B-v0.1",
            "--self-test",
            "--device", "cpu"  # Force CPU to avoid CUDA issues
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")  
            print(result.stderr)
        
        if result.returncode == 0:
            if "KL SANITY TEST" in result.stdout and "PASS" in result.stdout:
                print("✅ --self-test flag works correctly")
                return True
            else:
                print("❌ --self-test flag didn't produce expected output")
                return False
        else:
            print(f"❌ --self-test failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ --self-test timed out")
        return False
    except Exception as e:
        print(f"❌ --self-test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_self_test_flag()
    sys.exit(0 if success else 1)