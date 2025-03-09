#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Compatibility Test for Celeris.

This script tests Celeris's compatibility with different GPU architectures
by running a series of operations and benchmarks.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import sys
import os

# Add the parent directory to the path to import celeris
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Celeris
import celeris
from celeris.utils import (
    get_gpu_info, 
    get_memory_info, 
    benchmark_matmul, 
    print_gpu_summary,
    get_optimal_dtype
)
from celeris.config import get_config, set_config

def test_basic_operations():
    """Test basic tensor operations."""
    print("\n=== Testing Basic Operations ===")
    
    # Create tensors
    a = celeris.randn(1000, 1000)
    b = celeris.randn(1000, 1000)
    
    # Test operations
    start = time.time()
    c = celeris.add(a, b)
    add_time = time.time() - start
    
    start = time.time()
    d = celeris.mul(a, b)
    mul_time = time.time() - start
    
    start = time.time()
    e = celeris.matmul(a, b)
    matmul_time = time.time() - start
    
    print(f"Addition time: {add_time*1000:.2f} ms")
    print(f"Multiplication time: {mul_time*1000:.2f} ms")
    print(f"Matrix multiplication time: {matmul_time*1000:.2f} ms")
    
    return {
        "add_time": add_time,
        "mul_time": mul_time,
        "matmul_time": matmul_time
    }

def test_matrix_sizes():
    """Test matrix multiplication with different sizes."""
    print("\n=== Testing Matrix Sizes ===")
    
    sizes = [512, 1024, 2048, 4096]
    times = []
    gflops = []
    
    for size in sizes:
        print(f"Testing {size}x{size} matrix multiplication...")
        result = benchmark_matmul(size)
        times.append(result["time_seconds"])
        gflops.append(result["gflops"])
        print(f"  Time: {result['time_seconds']:.4f} seconds")
        print(f"  Performance: {result['gflops']:.2f} GFLOPS")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(sizes, times, marker='o')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Multiplication Time')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(sizes, gflops, marker='o', color='r')
    plt.xlabel('Matrix Size')
    plt.ylabel('Performance (GFLOPS)')
    plt.title('Matrix Multiplication Performance')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('matrix_size_benchmark.png')
    plt.close()
    
    print(f"Results saved to matrix_size_benchmark.png")
    
    return {
        "sizes": sizes,
        "times": times,
        "gflops": gflops
    }

def test_data_types():
    """Test different data types."""
    print("\n=== Testing Data Types ===")
    
    size = 1024
    dtypes = [torch.float32]
    
    # Add FP16 if supported
    if torch.cuda.is_available():
        dtypes.append(torch.float16)
    
    # Add BF16 if supported
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    
    times = []
    gflops = []
    dtype_names = []
    
    for dtype in dtypes:
        dtype_name = str(dtype).split(".")[-1]
        dtype_names.append(dtype_name)
        print(f"Testing {dtype_name}...")
        result = benchmark_matmul(size, dtype)
        times.append(result["time_seconds"])
        gflops.append(result["gflops"])
        print(f"  Time: {result['time_seconds']:.4f} seconds")
        print(f"  Performance: {result['gflops']:.2f} GFLOPS")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(dtype_names, times)
    plt.xlabel('Data Type')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Multiplication Time by Data Type')
    
    plt.subplot(1, 2, 2)
    plt.bar(dtype_names, gflops, color='r')
    plt.xlabel('Data Type')
    plt.ylabel('Performance (GFLOPS)')
    plt.title('Matrix Multiplication Performance by Data Type')
    
    plt.tight_layout()
    plt.savefig('data_type_benchmark.png')
    plt.close()
    
    print(f"Results saved to data_type_benchmark.png")
    
    return {
        "dtypes": dtype_names,
        "times": times,
        "gflops": gflops
    }

def test_tensor_cores():
    """Test tensor core performance."""
    print("\n=== Testing Tensor Cores ===")
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping tensor core test.")
        return None
    
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    
    if capability[0] < 7:
        print(f"GPU with compute capability {capability[0]}.{capability[1]} does not support tensor cores. Skipping test.")
        return None
    
    # Test with tensor cores enabled
    print("Testing with tensor cores enabled...")
    set_config("performance", "use_tensor_cores", True)
    tc_enabled_result = benchmark_matmul(4096)
    
    # Test with tensor cores disabled
    print("Testing with tensor cores disabled...")
    set_config("performance", "use_tensor_cores", False)
    tc_disabled_result = benchmark_matmul(4096)
    
    # Reset config
    set_config("performance", "use_tensor_cores", True)
    
    speedup = tc_disabled_result["time_seconds"] / tc_enabled_result["time_seconds"]
    
    print(f"Tensor cores enabled: {tc_enabled_result['gflops']:.2f} GFLOPS")
    print(f"Tensor cores disabled: {tc_disabled_result['gflops']:.2f} GFLOPS")
    print(f"Speedup: {speedup:.2f}x")
    
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.bar(["Tensor Cores Disabled", "Tensor Cores Enabled"], 
            [tc_disabled_result["gflops"], tc_enabled_result["gflops"]])
    plt.ylabel('Performance (GFLOPS)')
    plt.title('Tensor Core Performance Comparison')
    plt.savefig('tensor_core_benchmark.png')
    plt.close()
    
    print(f"Results saved to tensor_core_benchmark.png")
    
    return {
        "tc_enabled_gflops": tc_enabled_result["gflops"],
        "tc_disabled_gflops": tc_disabled_result["gflops"],
        "speedup": speedup
    }

def test_non_power_of_2():
    """Test performance with non-power-of-2 matrix sizes."""
    print("\n=== Testing Non-Power-of-2 Matrix Sizes ===")
    
    # Power of 2 sizes
    pow2_sizes = [1024, 2048]
    
    # Non-power of 2 sizes
    nonpow2_sizes = [1023, 1025, 2047, 2049]
    
    pow2_times = []
    pow2_gflops = []
    nonpow2_times = []
    nonpow2_gflops = []
    
    print("Testing power-of-2 sizes...")
    for size in pow2_sizes:
        print(f"  Testing {size}x{size}...")
        result = benchmark_matmul(size)
        pow2_times.append(result["time_seconds"])
        pow2_gflops.append(result["gflops"])
    
    print("Testing non-power-of-2 sizes...")
    for size in nonpow2_sizes:
        print(f"  Testing {size}x{size}...")
        result = benchmark_matmul(size)
        nonpow2_times.append(result["time_seconds"])
        nonpow2_gflops.append(result["gflops"])
    
    # Calculate average performance
    avg_pow2_gflops = sum(pow2_gflops) / len(pow2_gflops)
    avg_nonpow2_gflops = sum(nonpow2_gflops) / len(nonpow2_gflops)
    
    print(f"Average power-of-2 performance: {avg_pow2_gflops:.2f} GFLOPS")
    print(f"Average non-power-of-2 performance: {avg_nonpow2_gflops:.2f} GFLOPS")
    print(f"Ratio (non-power-of-2 / power-of-2): {avg_nonpow2_gflops / avg_pow2_gflops:.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    all_sizes = pow2_sizes + nonpow2_sizes
    all_gflops = pow2_gflops + nonpow2_gflops
    
    # Create color list (blue for power-of-2, red for non-power-of-2)
    colors = ['blue'] * len(pow2_sizes) + ['red'] * len(nonpow2_sizes)
    
    plt.bar(range(len(all_sizes)), all_gflops, color=colors)
    plt.xticks(range(len(all_sizes)), all_sizes)
    plt.xlabel('Matrix Size')
    plt.ylabel('Performance (GFLOPS)')
    plt.title('Power-of-2 vs. Non-Power-of-2 Performance')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Power of 2'),
        Patch(facecolor='red', label='Non-Power of 2')
    ]
    plt.legend(handles=legend_elements)
    
    plt.savefig('non_power_of_2_benchmark.png')
    plt.close()
    
    print(f"Results saved to non_power_of_2_benchmark.png")
    
    return {
        "pow2_sizes": pow2_sizes,
        "pow2_gflops": pow2_gflops,
        "nonpow2_sizes": nonpow2_sizes,
        "nonpow2_gflops": nonpow2_gflops,
        "ratio": avg_nonpow2_gflops / avg_pow2_gflops
    }

def main():
    parser = argparse.ArgumentParser(description='Test Celeris GPU compatibility')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--basic', action='store_true', help='Test basic operations')
    parser.add_argument('--sizes', action='store_true', help='Test different matrix sizes')
    parser.add_argument('--dtypes', action='store_true', help='Test different data types')
    parser.add_argument('--tensor-cores', action='store_true', help='Test tensor core performance')
    parser.add_argument('--non-pow2', action='store_true', help='Test non-power-of-2 matrix sizes')
    
    args = parser.parse_args()
    
    # If no specific tests are requested, run all tests
    if not (args.basic or args.sizes or args.dtypes or args.tensor_cores or args.non_pow2):
        args.all = True
    
    # Print GPU information
    print_gpu_summary()
    
    # Run tests
    if args.all or args.basic:
        test_basic_operations()
    
    if args.all or args.sizes:
        test_matrix_sizes()
    
    if args.all or args.dtypes:
        test_data_types()
    
    if args.all or args.tensor_cores:
        test_tensor_cores()
    
    if args.all or args.non_pow2:
        test_non_power_of_2()
    
    print("\nAll tests completed successfully!")

if __name__ == '__main__':
    main() 