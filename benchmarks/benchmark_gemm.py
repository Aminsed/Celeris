#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark script for comparing GEMM performance between Celeris and cuBLAS.
"""

import numpy as np
import time
import sys
import os
import argparse
import matplotlib.pyplot as plt
from tabulate import tabulate

# Add the parent directory to the path so we can import celeris
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import celeris
    print("Successfully imported celeris")
except ImportError as e:
    print(f"Failed to import celeris: {e}")
    print("Make sure the library is built and installed correctly.")
    sys.exit(1)

try:
    import torch
    print("Successfully imported PyTorch (for cuBLAS comparison)")
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("PyTorch CUDA is not available. Only Celeris will be benchmarked.")
        torch = None
except ImportError:
    print("PyTorch not found. Only Celeris will be benchmarked.")
    torch = None

def benchmark_celeris(M, N, K, dtype=np.float32, num_runs=10):
    """Benchmark Celeris GEMM performance."""
    # Create random matrices
    a_np = np.random.randn(M, K).astype(dtype)
    b_np = np.random.randn(K, N).astype(dtype)
    
    # Convert to Celeris tensors
    a = celeris.from_numpy(a_np)
    b = celeris.from_numpy(b_np)
    
    # Warm-up run
    c = celeris.matmul(a, b)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        c = celeris.matmul(a, b)
    end_time = time.time()
    
    # Calculate average time and GFLOPS
    avg_time = (end_time - start_time) / num_runs
    flops = 2 * M * N * K  # 2*M*N*K for matrix multiplication
    gflops = flops / (avg_time * 1e9)
    
    return avg_time, gflops

def benchmark_cublas(M, N, K, dtype=np.float32, num_runs=10):
    """Benchmark cuBLAS GEMM performance using PyTorch."""
    if torch is None:
        return None, None
    
    # Create random matrices
    a_np = np.random.randn(M, K).astype(dtype)
    b_np = np.random.randn(K, N).astype(dtype)
    
    # Convert to PyTorch tensors
    a = torch.tensor(a_np, device='cuda')
    b = torch.tensor(b_np, device='cuda')
    
    # Warm-up run
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate average time and GFLOPS
    avg_time = (end_time - start_time) / num_runs
    flops = 2 * M * N * K  # 2*M*N*K for matrix multiplication
    gflops = flops / (avg_time * 1e9)
    
    return avg_time, gflops

def run_benchmarks(sizes, num_runs=10, plot=True, save_plot=False):
    """Run benchmarks for various matrix sizes."""
    results = []
    
    for size in sizes:
        if isinstance(size, tuple):
            M, N, K = size
            size_str = f"{M}x{N}x{K}"
        else:
            M = N = K = size
            size_str = f"{size}x{size}"
        
        print(f"\nBenchmarking GEMM with matrices of size: A({M}x{K}), B({K}x{N}), C({M}x{N})")
        
        # Benchmark Celeris
        celeris_time, celeris_gflops = benchmark_celeris(M, N, K, num_runs=num_runs)
        print(f"Celeris: {celeris_time:.4f} seconds, {celeris_gflops:.2f} GFLOPS")
        
        # Benchmark cuBLAS
        cublas_time, cublas_gflops = benchmark_cublas(M, N, K, num_runs=num_runs)
        if cublas_time is not None:
            print(f"cuBLAS:  {cublas_time:.4f} seconds, {cublas_gflops:.2f} GFLOPS")
            
            # Calculate speedup
            if cublas_gflops > 0:
                speedup = celeris_gflops / cublas_gflops
                print(f"Speedup: {speedup:.2f}x")
                
                results.append([size_str, celeris_time, celeris_gflops, cublas_time, cublas_gflops, speedup])
            else:
                results.append([size_str, celeris_time, celeris_gflops, cublas_time, cublas_gflops, "N/A"])
        else:
            results.append([size_str, celeris_time, celeris_gflops, "N/A", "N/A", "N/A"])
    
    # Print results table
    headers = ["Size", "Celeris Time (s)", "Celeris GFLOPS", "cuBLAS Time (s)", "cuBLAS GFLOPS", "Speedup"]
    print("\n" + tabulate(results, headers=headers, tablefmt="grid"))
    
    # Plot results
    if plot and torch is not None:
        plot_results(sizes, results, save_plot)
    
    return results

def plot_results(sizes, results, save_plot=False):
    """Plot benchmark results."""
    # Extract data for plotting
    size_labels = [r[0] for r in results]
    celeris_gflops = [r[2] for r in results]
    cublas_gflops = [r[4] if r[4] != "N/A" else 0 for r in results]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Bar plot
    x = np.arange(len(size_labels))
    width = 0.35
    
    plt.bar(x - width/2, celeris_gflops, width, label='Celeris')
    plt.bar(x + width/2, cublas_gflops, width, label='cuBLAS')
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Performance (GFLOPS)')
    plt.title('GEMM Performance: Celeris vs. cuBLAS')
    plt.xticks(x, size_labels)
    plt.legend()
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(celeris_gflops):
        plt.text(i - width/2, v + 50, f"{v:.0f}", ha='center')
    
    for i, v in enumerate(cublas_gflops):
        if v > 0:
            plt.text(i + width/2, v + 50, f"{v:.0f}", ha='center')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('gemm_benchmark.png', dpi=300)
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Benchmark GEMM performance between Celeris and cuBLAS.')
    parser.add_argument('--sizes', type=int, nargs='+', default=[1024, 2048, 4096],
                        help='Matrix sizes to benchmark (default: 1024 2048 4096)')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of runs for each benchmark (default: 10)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting')
    parser.add_argument('--save-plot', action='store_true',
                        help='Save plot to file')
    parser.add_argument('--non-power-of-2', action='store_true',
                        help='Include non-power-of-2 sizes in benchmark')
    
    args = parser.parse_args()
    
    # Prepare sizes
    sizes = args.sizes
    
    # Add non-power-of-2 sizes if requested
    if args.non_power_of_2:
        non_pow2_sizes = []
        for size in sizes:
            non_pow2_sizes.append(size - 1)
            non_pow2_sizes.append(size + 1)
            if size >= 1000:
                non_pow2_sizes.append(size // 10 * 9)  # 90% of size
                non_pow2_sizes.append(size // 10 * 11)  # 110% of size
        sizes.extend(non_pow2_sizes)
        sizes.sort()
    
    # Run benchmarks
    run_benchmarks(sizes, num_runs=args.runs, plot=not args.no_plot, save_plot=args.save_plot)

if __name__ == "__main__":
    main() 