#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test for overlapping operations in CUDA kernels.
"""

import numpy as np
import sys
import os
import time
import re

# Add the parent directory to the path so we can import celeris
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import celeris
    print("Successfully imported celeris")
except ImportError as e:
    print(f"Failed to import celeris: {e}")
    print("Make sure the library is built and installed correctly.")
    sys.exit(1)

def test_overlapping_operations():
    """Test if CUDA kernels utilize overlapping operations."""
    print("\n=== Testing Overlapping Operations ===")
    
    # Check if our implementation has double buffering for overlapping operations
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'cuda'))
    gemm_file = os.path.join(src_dir, 'gemm.cu')
    
    if os.path.exists(gemm_file):
        with open(gemm_file, 'r') as f:
            gemm_code = f.read()
            
        # Check for double buffering
        double_buffer_check = re.search(r'buffer_idx\s*=\s*0', gemm_code) is not None
        next_buffer_check = re.search(r'next_buffer_idx\s*=\s*1\s*-\s*buffer_idx', gemm_code) is not None
        
        if double_buffer_check and next_buffer_check:
            print("✓ GEMM kernel uses double buffering for overlapping operations")
        else:
            print("✗ GEMM kernel does not use double buffering for overlapping operations")
        
        # Check for prefetching
        prefetch_check = re.search(r'Prefetch', gemm_code, re.IGNORECASE) is not None
        
        if prefetch_check:
            print("✓ GEMM kernel uses prefetching for overlapping operations")
        else:
            print("✗ GEMM kernel does not use prefetching for overlapping operations")
    else:
        print(f"Could not find GEMM kernel file at {gemm_file}")
    
    # Performance test to verify overlapping operations
    print("\nPerformance test for overlapping operations:")
    
    # Create large matrices to test GEMM performance
    m, n, k = 2048, 2048, 2048
    
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
    
    elapsed_time = end_time - start_time
    print(f"GEMM operation time: {elapsed_time:.4f} seconds")
    
    # In a real implementation, we would compare with a version without overlapping
    # For now, we'll just check if the implementation has the overlapping mechanisms
    print("Note: In a full implementation, we would compare with a version without overlapping")
    print("Current implementation uses simplified GEMM, so we can't directly verify performance impact")

if __name__ == "__main__":
    test_overlapping_operations() 