#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test for block scheduling and rasterization in CUDA kernels.
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

def test_block_scheduling():
    """Test if CUDA kernels utilize efficient block scheduling and rasterization."""
    print("\n=== Testing Block Scheduling and Rasterization ===")
    
    # Check if our implementation has grid-stride loops and efficient block scheduling
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'cuda'))
    gemm_file = os.path.join(src_dir, 'gemm.cu')
    
    if os.path.exists(gemm_file):
        with open(gemm_file, 'r') as f:
            gemm_code = f.read()
            
        # Check for grid-stride loops
        grid_stride_check = re.search(r'gridDim', gemm_code) is not None
        
        if grid_stride_check:
            print("✓ GEMM kernel uses grid-stride loops for efficient block scheduling")
        else:
            print("✗ GEMM kernel does not use grid-stride loops for efficient block scheduling")
            
        # Check for specific grid-stride loop pattern
        grid_stride_loop = re.search(r'for\s*\(\s*int\s+\w+\s*=\s*blockIdx\.\w+\s*;\s*.*;\s*.*gridDim', gemm_code) is not None
        
        if grid_stride_loop:
            print("✓ GEMM kernel implements proper grid-stride loop pattern")
        else:
            print("✗ GEMM kernel does not implement proper grid-stride loop pattern")
        
        # Check for block size tuning
        block_size_check = re.search(r'block_size_[mn]', gemm_code) is not None
        
        if block_size_check:
            print("✓ GEMM kernel uses tuned block sizes for efficient rasterization")
        else:
            print("✗ GEMM kernel does not use tuned block sizes for efficient rasterization")
            
        # Check for block size adjustment for non-power-of-2 sizes
        block_size_adjust = re.search(r'block_size_\w+\s*=\s*\(\s*block_size_\w+\s*\/\s*\w+\s*\)\s*\*\s*\w+', gemm_code) is not None
        
        if block_size_adjust:
            print("✓ GEMM kernel adjusts block sizes for better rasterization")
        else:
            print("✗ GEMM kernel does not adjust block sizes for better rasterization")
            
        # Overall assessment
        block_scheduling_score = 0
        block_scheduling_score += 1 if grid_stride_check else 0
        block_scheduling_score += 1 if grid_stride_loop else 0
        block_scheduling_score += 1 if block_size_check else 0
        block_scheduling_score += 1 if block_size_adjust else 0
        
        print(f"\nBlock scheduling score: {block_scheduling_score}/4")
        
        if block_scheduling_score >= 3:
            print("✓ GEMM kernel has excellent block scheduling and rasterization")
        elif block_scheduling_score >= 2:
            print("⚠ GEMM kernel has moderate block scheduling and rasterization, but could be improved")
        else:
            print("✗ GEMM kernel has poor block scheduling and rasterization")
    else:
        print(f"Could not find GEMM kernel file at {gemm_file}")
    
    # Performance test with different matrix sizes to verify block scheduling
    print("\nPerformance test for block scheduling:")
    
    # Test with matrices of different sizes to see how block scheduling affects performance
    # Include both power-of-2 and non-power-of-2 sizes to test rasterization efficiency
    sizes = [
        (1024, 1024),  # Power of 2
        (1023, 1023),  # Non-power of 2 (slightly smaller)
        (1025, 1025),  # Non-power of 2 (slightly larger)
        (1536, 1536),  # 1.5 * power of 2
        (2048, 2048),  # Power of 2
        (2047, 2047),  # Non-power of 2
    ]
    
    results = {}
    
    for size in sizes:
        m, n = size
        k = m
        
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
        
        elapsed_time = end_time - start_time
        print(f"GEMM operation time: {elapsed_time:.4f} seconds")
        
        # Calculate GFLOPS
        flops = 2 * m * n * k  # 2*M*N*K for matrix multiplication
        gflops = flops / (elapsed_time * 1e9)
        print(f"Performance: {gflops:.2f} GFLOPS")
        
        results[size] = gflops
    
    # Analyze performance across different matrix sizes
    print("\nPerformance analysis across different matrix sizes:")
    
    # Compare power-of-2 vs non-power-of-2 performance
    pow2_perf = [results[s] for s in sizes if s[0] in (1024, 2048)]
    nonpow2_perf = [results[s] for s in sizes if s[0] not in (1024, 2048)]
    
    avg_pow2 = sum(pow2_perf) / len(pow2_perf) if pow2_perf else 0
    avg_nonpow2 = sum(nonpow2_perf) / len(nonpow2_perf) if nonpow2_perf else 0
    
    print(f"Average performance for power-of-2 sizes: {avg_pow2:.2f} GFLOPS")
    print(f"Average performance for non-power-of-2 sizes: {avg_nonpow2:.2f} GFLOPS")
    
    if avg_nonpow2 > 0:
        perf_ratio = avg_pow2 / avg_nonpow2
        print(f"Performance ratio (power-of-2 / non-power-of-2): {perf_ratio:.2f}x")
        
        if perf_ratio < 1.2:
            print("✓ Excellent block scheduling: Non-power-of-2 performance is close to power-of-2 performance")
        elif perf_ratio < 1.5:
            print("⚠ Good block scheduling: Some performance drop for non-power-of-2 sizes")
        else:
            print("✗ Poor block scheduling: Significant performance drop for non-power-of-2 sizes")
    
    print("\nBlock scheduling and rasterization impact:")
    print("Efficient block scheduling can improve performance by 20-50% for non-power-of-2 matrix sizes")
    print("This is achieved by ensuring good GPU utilization regardless of matrix dimensions")

if __name__ == "__main__":
    test_block_scheduling() 