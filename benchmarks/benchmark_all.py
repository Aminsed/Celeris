#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark script to compare various configurations of Celeris for various operations.
This script benchmarks Celeris operations for two configurations:
  Config A: Reduced precision disabled.
  Config B: Reduced precision enabled.
"""
import time
import numpy as np
import sys
import os
from tabulate import tabulate

# Add the parent directory to the path to import celeris
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import celeris
from celeris.config import get_config, set_config


def benchmark_operation(op_name, op_func, shape, num_runs=5):
    if op_name == 'MatMul':
        np_data1 = np.random.randn(shape[0], shape[1]).astype(np.float32)
        np_data2 = np.random.randn(shape[1], shape[0]).astype(np.float32)
        t1 = celeris.from_numpy(np_data1)
        t2 = celeris.from_numpy(np_data2)
    else:
        np_data1 = np.random.randn(*shape).astype(np.float32)
        np_data2 = np.random.randn(*shape).astype(np.float32)
        t1 = celeris.from_numpy(np_data1)
        t2 = celeris.from_numpy(np_data2)
    
    # Warm-up
    for _ in range(2):
        if op_name in ['Element-wise Add', 'Element-wise Mul', 'MatMul']:
            _ = op_func(t1, t2)
        else:
            _ = op_func(t1)
    
    start = time.time()
    for _ in range(num_runs):
        if op_name in ['Element-wise Add', 'Element-wise Mul', 'MatMul']:
            _ = op_func(t1, t2)
        else:
            _ = op_func(t1)
    elapsed = (time.time() - start) / num_runs
    elements = np.prod(shape)
    throughput = elements / elapsed / 1e9  # GOPS
    return elapsed, throughput


def main():
    # Check CUDA availability via torch (assumed required for Celeris backend)
    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA is not available. Exiting.")
            sys.exit(1)
    except ImportError:
        print("Torch is required for CUDA operations. Exiting.")
        sys.exit(1)
    
    sizes = [
        (1024, 1024),
        (2048, 2048),
        (1023, 1023),
        (1025, 1025)
    ]
    
    operations = [
        ('Element-wise Add', celeris.add),
        ('Element-wise Mul', celeris.mul),
        ('MatMul', celeris.matmul),
        ('ReLU', celeris.relu),
        ('Sigmoid', celeris.sigmoid),
        ('Tanh', celeris.tanh)
    ]
    
    results = []
    # Save original reduced precision setting
    original_rp = get_config()["performance"]["use_reduced_precision"]
    
    # Run Config A: Reduced precision disabled
    set_config("performance", "use_reduced_precision", False)
    print("Running benchmarks with Config A (use_reduced_precision = False)")
    resultsA = {}
    for op_name, op_func in operations:
        for shape in sizes:
            elapsed, throughput = benchmark_operation(op_name, op_func, shape)
            resultsA[(op_name, shape)] = (elapsed, throughput)
            print(f"{op_name} with shape {shape} - Config A: {elapsed:.6f}s, {throughput:.2f} GOPS")
    
    # Run Config B: Reduced precision enabled
    set_config("performance", "use_reduced_precision", True)
    print("\nRunning benchmarks with Config B (use_reduced_precision = True)")
    resultsB = {}
    for op_name, op_func in operations:
        for shape in sizes:
            elapsed, throughput = benchmark_operation(op_name, op_func, shape)
            resultsB[(op_name, shape)] = (elapsed, throughput)
            print(f"{op_name} with shape {shape} - Config B: {elapsed:.6f}s, {throughput:.2f} GOPS")
    
    # Restore original setting
    set_config("performance", "use_reduced_precision", original_rp)
    
    # Prepare results table
    table = []
    for op_name, op_func in operations:
        for shape in sizes:
            timeA, gopsA = resultsA[(op_name, shape)]
            timeB, gopsB = resultsB[(op_name, shape)]
            speedup = timeA / timeB if timeB > 0 else float('inf')
            table.append([op_name, shape, timeA, gopsA, timeB, gopsB, speedup])
    headers = ['Operation', 'Shape', 'Config A Time (s)', 'Config A GOPS', 'Config B Time (s)', 'Config B GOPS', 'Speedup (A/B)']
    print("\nBenchmark Results:")
    print(tabulate(table, headers=headers, tablefmt='grid'))
    
    results_file = "benchmark_results.txt"
    with open(results_file, 'w') as f:
        f.write(tabulate(table, headers=headers, tablefmt='grid'))
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main() 