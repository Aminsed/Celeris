#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test for Just-In-Time (JIT) compilation in CUDA kernels.
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

def test_jit_compilation():
    """Test if CUDA kernels utilize Just-In-Time (JIT) compilation."""
    print("\n=== Testing Just-In-Time (JIT) Compilation ===")
    
    # Check if our implementation has JIT compilation
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'cuda'))
    jit_file = os.path.join(src_dir, 'jit.cpp')
    
    if os.path.exists(jit_file):
        with open(jit_file, 'r') as f:
            jit_code = f.read()
            
        # Check for JIT compilation functionality
        jit_compile_check = re.search(r'compile\(', jit_code) is not None
        
        if jit_compile_check:
            print("✓ Implementation includes JIT compilation functionality")
        else:
            print("✗ Implementation does not include JIT compilation functionality")
        
        # Check if it's a simplified implementation
        simplified_check = re.search(r'Simplified implementation', jit_code) is not None
        
        if simplified_check:
            print("⚠ JIT compilation is simplified for testing purposes")
        else:
            print("✓ JIT compilation is fully implemented")
            
        # Check for NVRTC usage
        nvrtc_check = re.search(r'nvrtc', jit_code) is not None
        
        if nvrtc_check:
            print("✓ JIT compilation uses NVRTC for runtime compilation")
        else:
            print("✗ JIT compilation does not use NVRTC for runtime compilation")
            
        # Check for kernel caching
        cache_check = re.search(r'cache_', jit_code) is not None
        
        if cache_check:
            print("✓ JIT compilation uses kernel caching for better performance")
        else:
            print("✗ JIT compilation does not use kernel caching")
            
        # Overall assessment
        jit_score = 0
        jit_score += 1 if jit_compile_check else 0
        jit_score += 1 if not simplified_check else 0
        jit_score += 1 if nvrtc_check else 0
        jit_score += 1 if cache_check else 0
        
        print(f"\nJIT compilation score: {jit_score}/4")
        
        if jit_score >= 3:
            print("✓ JIT compilation is well implemented")
        elif jit_score >= 2:
            print("⚠ JIT compilation is partially implemented")
        else:
            print("✗ JIT compilation is poorly implemented")
    else:
        print(f"Could not find JIT implementation file at {jit_file}")
    
    # Test JIT compilation with different matrix sizes
    print("\nTesting JIT compilation with different matrix sizes:")
    
    # First run should trigger JIT compilation
    print("\nFirst run (should trigger JIT compilation):")
    a = celeris.randn([1024, 1024])
    b = celeris.randn([1024, 1024])
    
    # Warm up GPU
    _ = celeris.matmul(a, b)
    
    # Measure first run time
    start_time = time.time()
    c = celeris.matmul(a, b)
    end_time = time.time()
    
    first_run_time = end_time - start_time
    print(f"First run time: {first_run_time:.4f} seconds")
    
    # Second run should use cached kernel
    print("\nSecond run (should use cached kernel):")
    start_time = time.time()
    c = celeris.matmul(a, b)
    end_time = time.time()
    
    second_run_time = end_time - start_time
    print(f"Second run time: {second_run_time:.4f} seconds")
    
    # Calculate speedup
    if second_run_time > 0:
        speedup = first_run_time / second_run_time
        print(f"Speedup from first to second run: {speedup:.2f}x")
    
    # In a real implementation, the first run should be slower due to JIT compilation
    if first_run_time > second_run_time:
        print("✓ First run is slower than second run, suggesting JIT compilation overhead")
    else:
        print("⚠ First run is not slower than second run, JIT compilation may not be working as expected")
    
    # Test with different matrix sizes to verify dynamic kernel generation
    print("\nTesting with different matrix sizes to verify dynamic kernel generation:")
    
    # Different matrix sizes should trigger different kernel compilations
    sizes = [(512, 512), (1024, 1024), (2048, 2048)]
    
    run_times = {}
    
    for size in sizes:
        m, n = size
        k = m
        
        print(f"\nTesting GEMM with matrices of size: A({m}x{k}), B({k}x{n}), C({m}x{n})")
        
        a = celeris.randn([m, k])
        b = celeris.randn([k, n])
        
        # First run (compilation)
        _ = celeris.matmul(a, b)
        
        # Second run (execution)
        start_time = time.time()
        c = celeris.matmul(a, b)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        print(f"GEMM operation time: {elapsed_time:.4f} seconds")
        
        # Calculate GFLOPS
        flops = 2 * m * n * k  # 2*M*N*K for matrix multiplication
        gflops = flops / (elapsed_time * 1e9)
        print(f"Performance: {gflops:.2f} GFLOPS")
        
        run_times[size] = elapsed_time
    
    # Analyze performance across different matrix sizes
    print("\nPerformance analysis across different matrix sizes:")
    for size, time_taken in run_times.items():
        print(f"Size {size}: {time_taken:.4f} seconds")
    
    print("\nJIT compilation impact:")
    print("Efficient JIT compilation can improve performance by dynamically generating optimized kernels")
    print("This is achieved by specializing kernels for specific matrix sizes and data types")

if __name__ == "__main__":
    test_jit_compilation() 