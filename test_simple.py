#!/usr/bin/env python3
"""
Simple test to verify the test runner works via subprocess.
"""
import subprocess
import sys

def test_runner():
    """Test the test runner via subprocess."""
    try:
        result = subprocess.run([
            sys.executable, 'tests/test_runner.py', '--smoke'
        ], capture_output=True, text=True, timeout=300)
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT length: {len(result.stdout)}")
        print(f"STDERR length: {len(result.stderr)}")
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr[:500])
        
        if result.stdout:
            print("STDOUT (last 500 chars):")
            print(result.stdout[-500:])
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_runner()
    print(f"Test runner success: {success}")
    sys.exit(0 if success else 1) 