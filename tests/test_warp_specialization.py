#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test for persistent warp specialization in GEMM operations.
"""

import numpy as np
import sys
import os
import time

# Add the parent directory to the path so we can import celeris
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import celeris
    print("Successfully imported celeris")
except ImportError as e:
    print(f"Failed to import celeris: {e}")
    print("Make sure the library is built and installed correctly.")
    sys.exit(1)

def test_warp_specialization():
    """Test if GEMM operations utilize persistent warp specialization."""
    print("\n=== Testing Persistent Warp Specialization ===")
    
    # Create large matrices to test GEMM performance
    # Using sizes that are multiples of warp size (32) and typical block sizes
    sizes = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]
    
    for m, n, k in sizes:
        print(f"\nTesting GEMM with matrices of size: A({m}x{k}), B({k}x{n}), C({m}x{n})")
        
        # Create random matrices
        a_np = np.random.randn(m, k).astype(np.float32)
        b_np = np.random.randn(k, n).astype(np.float32)
        
        # Convert to Celeris tensors
        a = celeris.from_numpy(a_np)
        b = celeris.from_numpy(b_np)
        
        # Warm-up run
        c = celeris.matmul(a, b)
        
        # Measure performance
        start_time = time.time()
        c = celeris.matmul(a, b)
        end_time = time.time()
        
        # Calculate FLOPS (2*m*n*k for matrix multiplication)
        flops = 2 * m * n * k
        elapsed_time = end_time - start_time
        gflops = flops / (elapsed_time * 1e9)
        
        print(f"Time: {elapsed_time:.4f} seconds")
        print(f"Performance: {gflops:.2f} GFLOPS")
        
        # Check if performance is reasonable for a GPU implementation
        # A basic implementation without warp specialization would be much slower
        if gflops > 100:  # This threshold depends on the GPU, but should be high for Ampere
            print("✓ Performance suggests effective warp utilization")
        else:
            print("✗ Performance is lower than expected, warp specialization may not be optimal")

if __name__ == "__main__":
    test_warp_specialization() 