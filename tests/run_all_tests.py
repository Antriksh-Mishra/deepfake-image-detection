import unittest
import sys
import os
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_test_suite():
    """Master runner discovering and running all platform test suites."""
    print("=" * 70)
    print("🛡️ DEEPFAKE DETECTION PLATFORM - COMPREHENSIVE TEST SUITE RUNNER")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    t0 = time.time()
    result = runner.run(suite)
    elapsed = time.time() - t0

    print("=" * 70)
    print(f"TEST RUN SUMMARY: {result.testsRun} tests executed in {elapsed:.2f}s")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == "__main__":
    run_test_suite()
