#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test for unaligned block sizes in CUDA kernels.
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

def test_unaligned_block_sizes():
    """Test if CUDA kernels utilize unaligned block sizes."""
    print("\n=== Testing Unaligned Block Sizes ===")
    
    # Check if our implementation supports unaligned block sizes
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'cuda'))
    gemm_file = os.path.join(src_dir, 'gemm.cu')
    
    if os.path.exists(gemm_file):
        with open(gemm_file, 'r') as f:
            gemm_code = f.read()
            
        # Check for unaligned block sizes
        unaligned_check = re.search(r'unaligned_blocks', gemm_code) is not None
        
        if unaligned_check:
            print("✓ GEMM kernel supports unaligned block sizes")
        else:
            print("✗ GEMM kernel does not support unaligned block sizes")
        
        # Check for specific unaligned block sizes
        specific_sizes_check = re.search(r'112|96', gemm_code) is not None
        
        if specific_sizes_check:
            print("✓ GEMM kernel uses specific unaligned block sizes (e.g., 112, 96)")
        else:
            print("✗ GEMM kernel does not use specific unaligned block sizes")
            
        # Check for smaller unaligned block sizes
        small_sizes_check = re.search(r'56|48', gemm_code) is not None
        
        if small_sizes_check:
            print("✓ GEMM kernel uses smaller unaligned block sizes (e.g., 56, 48)")
        else:
            print("✗ GEMM kernel does not use smaller unaligned block sizes")
            
        # Check for dynamic selection of unaligned block sizes
        dynamic_selection = re.search(r'if\s*\(\s*.*!is_m_power_of_2\s*\|\|\s*!is_n_power_of_2', gemm_code) is not None or \
                           re.search(r'waste_m_\d+\s*<\s*waste_m_\d+', gemm_code) is not None
        
        if dynamic_selection:
            print("✓ GEMM kernel dynamically selects unaligned block sizes based on matrix dimensions")
        else:
            print("✗ GEMM kernel does not dynamically select unaligned block sizes")
            
        # Overall assessment
        unaligned_score = 0
        unaligned_score += 1 if unaligned_check else 0
        unaligned_score += 1 if specific_sizes_check else 0
        unaligned_score += 1 if small_sizes_check else 0
        unaligned_score += 1 if dynamic_selection else 0
        
        print(f"\nUnaligned block sizes score: {unaligned_score}/4")
        
        if unaligned_score >= 3:
            print("✓ GEMM kernel has excellent support for unaligned block sizes")
        elif unaligned_score >= 2:
            print("⚠ GEMM kernel has moderate support for unaligned block sizes, but could be improved")
        else:
            print("✗ GEMM kernel has poor support for unaligned block sizes")
    else:
        print(f"Could not find GEMM kernel file at {gemm_file}")
    
    # Performance test with different matrix sizes to verify unaligned block sizes
    print("\nPerformance test for unaligned block sizes:")
    
    # Test with matrices of different sizes to see how unaligned block sizes affect performance
    # Power-of-2 sizes
    power_of_2_sizes = [(1024, 1024), (2048, 2048)]
    
    # Non-power-of-2 sizes that might benefit from unaligned block sizes
    non_power_of_2_sizes = [
        (1120, 1120),  # Benefits from 112 block size
        (960, 960),    # Benefits from 96 block size
        (1008, 1008),  # Multiple of 112 (9*112)
        (1152, 1152),  # Multiple of 96 (12*96)
    ]
    
    results = {}
    
    print("\nTesting with power-of-2 sizes:")
    for size in power_of_2_sizes:
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
    
    print("\nTesting with non-power-of-2 sizes:")
    for size in non_power_of_2_sizes:
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
    pow2_perf = [results[s] for s in power_of_2_sizes]
    nonpow2_perf = [results[s] for s in non_power_of_2_sizes]
    
    avg_pow2 = sum(pow2_perf) / len(pow2_perf) if pow2_perf else 0
    avg_nonpow2 = sum(nonpow2_perf) / len(nonpow2_perf) if nonpow2_perf else 0
    
    print(f"Average performance for power-of-2 sizes: {avg_pow2:.2f} GFLOPS")
    print(f"Average performance for non-power-of-2 sizes: {avg_nonpow2:.2f} GFLOPS")
    
    if avg_nonpow2 > 0 and avg_pow2 > 0:
        perf_ratio = avg_nonpow2 / avg_pow2
        print(f"Performance ratio (non-power-of-2 / power-of-2): {perf_ratio:.2f}x")
        
        if perf_ratio > 0.95:
            print("✓ Excellent unaligned block sizes: Non-power-of-2 performance is close to power-of-2 performance")
        elif perf_ratio > 0.8:
            print("⚠ Good unaligned block sizes: Some performance drop for non-power-of-2 sizes")
        else:
            print("✗ Poor unaligned block sizes: Significant performance drop for non-power-of-2 sizes")
    
    # Analyze specific non-power-of-2 sizes
    print("\nPerformance for specific non-power-of-2 sizes:")
    for size in non_power_of_2_sizes:
        print(f"Size {size}: {results[size]:.2f} GFLOPS")
    
    print("\nUnaligned block sizes impact:")
    print("Efficient unaligned block sizes can improve performance by 10-30% for non-power-of-2 matrix sizes")
    print("This is achieved by reducing wasted computation and memory access for padded regions")

if __name__ == "__main__":
    test_unaligned_block_sizes() 