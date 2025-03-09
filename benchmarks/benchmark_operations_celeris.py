#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark script to systematically compare various operations between Celeris and PyTorch.
"""

import time
import numpy as np
import sys
import os
from tabulate import tabulate

# Add the parent directory to the path so we can import celeris
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import celeris
try:
    import celeris
    print("Successfully imported celeris")
except ImportError:
    print("Celeris not found. Please install it first.")
    sys.exit(1)

# Import PyTorch
try:
    import torch
    print(f"Successfully imported PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch not found. Please install it first.")
    sys.exit(1)

# Define operations to benchmark
operations = [
    ('Element-wise Add', lambda x, y: x + y, lambda x, y: celeris.add(x, y)),
    ('Element-wise Mul', lambda x, y: x * y, lambda x, y: celeris.mul(x, y)),
    ('MatMul', lambda x, y: torch.matmul(x, y), lambda x, y: celeris.matmul(x, y)),
    ('ReLU', lambda x: torch.relu(x), lambda x: celeris.relu(x)),
    ('Sigmoid', lambda x: torch.sigmoid(x), lambda x: celeris.sigmoid(x)),
    ('Tanh', lambda x: torch.tanh(x), lambda x: celeris.tanh(x))
]

# Benchmark function
def benchmark_operation(op_name, torch_op, celeris_op, shape, num_runs=5):
    # Generate random data
    if op_name == 'MatMul':
        # For matrix multiplication, we need compatible shapes
        if len(shape) == 2:
            np_data1 = np.random.randn(shape[0], shape[1]).astype(np.float32)
            np_data2 = np.random.randn(shape[1], shape[0]).astype(np.float32)
        else:
            # Handle other shapes if needed
            np_data1 = np.random.randn(*shape).astype(np.float32)
            np_data2 = np.random.randn(*shape).astype(np.float32)
    else:
        np_data1 = np.random.randn(*shape).astype(np.float32)
        np_data2 = np.random.randn(*shape).astype(np.float32)

    # PyTorch benchmark
    torch_tensor1 = torch.tensor(np_data1, device='cuda')
    torch_tensor2 = torch.tensor(np_data2, device='cuda')
    
    # Warm-up
    for _ in range(2):
        if op_name in ['Element-wise Add', 'Element-wise Mul', 'MatMul']:
            result = torch_op(torch_tensor1, torch_tensor2)
        else:
            result = torch_op(torch_tensor1)
        torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        if op_name in ['Element-wise Add', 'Element-wise Mul', 'MatMul']:
            result = torch_op(torch_tensor1, torch_tensor2)
        else:
            result = torch_op(torch_tensor1)
        torch.cuda.synchronize()
    torch_time = (time.time() - start) / num_runs
    
    # Celeris benchmark
    celeris_tensor1 = celeris.from_numpy(np_data1)
    celeris_tensor2 = celeris.from_numpy(np_data2)
    
    # Warm-up
    for _ in range(2):
        if op_name in ['Element-wise Add', 'Element-wise Mul', 'MatMul']:
            result = celeris_op(celeris_tensor1, celeris_tensor2)
        else:
            result = celeris_op(celeris_tensor1)
    
    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        if op_name in ['Element-wise Add', 'Element-wise Mul', 'MatMul']:
            result = celeris_op(celeris_tensor1, celeris_tensor2)
        else:
            result = celeris_op(celeris_tensor1)
    celeris_time = (time.time() - start) / num_runs
    
    # Calculate throughput (operations per second)
    elements = np.prod(shape)
    torch_throughput = elements / torch_time / 1e9  # GOPS (Giga Operations Per Second)
    celeris_throughput = elements / celeris_time / 1e9  # GOPS
    
    # Calculate speedup
    speedup = torch_time / celeris_time if celeris_time > 0 else float('inf')
    
    return torch_time, torch_throughput, celeris_time, celeris_throughput, speedup

# Main benchmarking loop
def main():
    # Check if CUDA is available for PyTorch
    if not torch.cuda.is_available():
        print("CUDA is not available for PyTorch. Exiting.")
        sys.exit(1)
    
    # Define tensor sizes to benchmark
    sizes = [
        (1024, 1024),      # Standard size
        (2048, 2048),      # Larger standard size
        (1023, 1023),      # Non-power-of-2 size
        (1025, 1025)       # Non-power-of-2 size
    ]
    
    results = []
    
    # Run benchmarks
    for op_name, torch_op, celeris_op in operations:
        for shape in sizes:
            print(f"\nBenchmarking {op_name} with shape {shape}...")
            try:
                torch_time, torch_throughput, celeris_time, celeris_throughput, speedup = benchmark_operation(
                    op_name, torch_op, celeris_op, shape)
                results.append([op_name, shape, torch_time, torch_throughput, celeris_time, celeris_throughput, speedup])
                print(f"  PyTorch: {torch_time:.6f} seconds, {torch_throughput:.2f} GOPS")
                print(f"  Celeris: {celeris_time:.6f} seconds, {celeris_throughput:.2f} GOPS")
                print(f"  Speedup (PyTorch/Celeris): {speedup:.2f}x")
            except Exception as e:
                print(f"Error benchmarking {op_name} with shape {shape}: {e}")
    
    # Print results
    headers = ['Operation', 'Shape', 'PyTorch Time (s)', 'PyTorch GOPS', 'Celeris Time (s)', 'Celeris GOPS', 'Speedup (PyTorch/Celeris)']
    print("\nBenchmark Results:")
    print(tabulate(results, headers=headers, tablefmt='grid'))
    
    # Save results to file
    with open('celeris_vs_pytorch_benchmark_results.txt', 'w') as f:
        f.write(tabulate(results, headers=headers, tablefmt='grid'))
    print("\nResults saved to celeris_vs_pytorch_benchmark_results.txt")

if __name__ == '__main__':
    main() 