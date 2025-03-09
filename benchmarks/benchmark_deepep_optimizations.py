#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark script to compare the performance of DeepEP optimizations in Celeris.
"""

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path so we can import celeris
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import celeris
try:
    import celeris
    print("Celeris imported successfully")
except ImportError:
    print("Celeris not found. Please install it first.")
    sys.exit(1)

# Try to import PyTorch for comparison
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch not found. Only Celeris will be benchmarked.")
    torch = None

def benchmark_celeris_standard(M, N, K, dtype=np.float32, num_runs=10):
    """Benchmark standard Celeris GEMM performance."""
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

def benchmark_celeris_aligned(M, N, K, dtype=np.float32, num_runs=10):
    """Benchmark Celeris GEMM performance with aligned memory."""
    # Create random matrices
    a_np = np.random.randn(M, K).astype(dtype)
    b_np = np.random.randn(K, N).astype(dtype)
    
    # Convert to aligned Celeris tensors
    a = celeris.create_aligned_tensor([M, K], celeris.FLOAT32, celeris.CUDA)
    b = celeris.create_aligned_tensor([K, N], celeris.FLOAT32, celeris.CUDA)
    
    # Copy data to tensors
    a.copy_from_numpy(a_np)
    b.copy_from_numpy(b_np)
    
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

def benchmark_pytorch(M, N, K, dtype=np.float32, num_runs=10):
    """Benchmark PyTorch GEMM performance."""
    if torch is None or not torch.cuda.is_available():
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
    """Run benchmarks for different matrix sizes."""
    results = {
        'sizes': sizes,
        'celeris_standard': {'time': [], 'gflops': []},
        'pytorch': {'time': [], 'gflops': []}
    }
    
    for size in sizes:
        print(f"\nBenchmarking matrix size: {size}x{size}")
        
        # Benchmark Celeris standard
        time_celeris_std, gflops_celeris_std = benchmark_celeris_standard(size, size, size, num_runs=num_runs)
        results['celeris_standard']['time'].append(time_celeris_std)
        results['celeris_standard']['gflops'].append(gflops_celeris_std)
        print(f"Celeris Standard: {time_celeris_std:.4f} seconds, {gflops_celeris_std:.2f} GFLOPS")
        
        # Benchmark PyTorch
        time_pytorch, gflops_pytorch = benchmark_pytorch(size, size, size, num_runs=num_runs)
        results['pytorch']['time'].append(time_pytorch)
        results['pytorch']['gflops'].append(gflops_pytorch)
        if time_pytorch is not None:
            print(f"PyTorch: {time_pytorch:.4f} seconds, {gflops_pytorch:.2f} GFLOPS")
        else:
            print("PyTorch: N/A")
    
    # Print summary table
    print("\nPerformance Summary (GFLOPS):")
    print("=" * 60)
    print(f"{'Size':<10} | {'Celeris Standard':<20} | {'PyTorch':<20}")
    print("-" * 60)
    for i, size in enumerate(sizes):
        celeris_std = f"{results['celeris_standard']['gflops'][i]:.2f}" if results['celeris_standard']['gflops'][i] is not None else "N/A"
        pytorch = f"{results['pytorch']['gflops'][i]:.2f}" if results['pytorch']['gflops'][i] is not None else "N/A"
        print(f"{size:<10} | {celeris_std:<20} | {pytorch:<20}")
    print("=" * 60)
    
    # Plot results
    if plot:
        plot_results(sizes, results, save_plot)
    
    return results

def plot_results(sizes, results, save_plot=False):
    """Plot benchmark results."""
    plt.figure(figsize=(12, 6))
    
    # Plot GFLOPS
    plt.subplot(1, 2, 1)
    plt.plot(sizes, results['celeris_standard']['gflops'], 'o-', label='Celeris Standard')
    if results['pytorch']['gflops'][0] is not None:
        plt.plot(sizes, results['pytorch']['gflops'], '^-', label='PyTorch')
    plt.xlabel('Matrix Size')
    plt.ylabel('GFLOPS')
    plt.title('GEMM Performance (higher is better)')
    plt.grid(True)
    plt.legend()
    
    # Plot execution time
    plt.subplot(1, 2, 2)
    plt.plot(sizes, results['celeris_standard']['time'], 'o-', label='Celeris Standard')
    if results['pytorch']['time'][0] is not None:
        plt.plot(sizes, results['pytorch']['time'], '^-', label='PyTorch')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time (lower is better)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('deepep_optimizations_benchmark.png', dpi=300)
    else:
        plt.show()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Benchmark DeepEP optimizations in Celeris')
    parser.add_argument('--sizes', type=int, nargs='+', default=[1024, 2048, 4096],
                        help='Matrix sizes to benchmark')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of runs for each benchmark')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting')
    parser.add_argument('--save-plot', action='store_true',
                        help='Save plot to file')
    parser.add_argument('--non-power-of-2', action='store_true',
                        help='Use non-power-of-2 matrix sizes')
    args = parser.parse_args()
    
    # Use non-power-of-2 sizes if requested
    if args.non_power_of_2:
        sizes = [918, 1023, 1024, 1025, 1836, 2047, 2048, 4096]
    else:
        sizes = args.sizes
    
    run_benchmarks(sizes, num_runs=args.runs, plot=not args.no_plot, save_plot=args.save_plot)

if __name__ == '__main__':
    main() 