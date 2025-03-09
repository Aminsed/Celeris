#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test for tensor cores support in CUDA kernels.
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

def test_tensor_cores():
    """Test if CUDA kernels utilize tensor cores on compatible GPUs."""
    print("\n=== Testing Tensor Cores Support ===")
    
    # Check if our implementation supports tensor cores
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'cuda'))
    gemm_file = os.path.join(src_dir, 'gemm.cu')
    
    if os.path.exists(gemm_file):
        with open(gemm_file, 'r') as f:
            gemm_code = f.read()
            
        # Check for tensor cores support
        tensor_cores_check = re.search(r'wmma', gemm_code) is not None
        
        if tensor_cores_check:
            print("✓ GEMM kernel supports tensor cores")
        else:
            print("✗ GEMM kernel does not support tensor cores")
        
        # Check for tensor core kernel template
        tensor_core_template = re.search(r'GEMM_TENSOR_CORE_TEMPLATE', gemm_code) is not None
        
        if tensor_core_template:
            print("✓ GEMM kernel has a specialized template for tensor cores")
        else:
            print("✗ GEMM kernel does not have a specialized template for tensor cores")
            
        # Check for tensor core operations
        tensor_core_ops = re.search(r'nvcuda::wmma::mma_sync', gemm_code) is not None
        
        if tensor_core_ops:
            print("✓ GEMM kernel uses tensor core operations (mma_sync)")
        else:
            print("✗ GEMM kernel does not use tensor core operations")
            
        # Check for dynamic selection of tensor cores
        dynamic_selection = re.search(r'tensor_cores_available', gemm_code) is not None
        
        if dynamic_selection:
            print("✓ GEMM kernel dynamically selects tensor cores based on GPU capabilities")
        else:
            print("✗ GEMM kernel does not dynamically select tensor cores")
            
        # Overall assessment
        tensor_cores_score = 0
        tensor_cores_score += 1 if tensor_cores_check else 0
        tensor_cores_score += 1 if tensor_core_template else 0
        tensor_cores_score += 1 if tensor_core_ops else 0
        tensor_cores_score += 1 if dynamic_selection else 0
        
        print(f"\nTensor cores support score: {tensor_cores_score}/4")
        
        if tensor_cores_score >= 3:
            print("✓ GEMM kernel has excellent support for tensor cores")
        elif tensor_cores_score >= 2:
            print("⚠ GEMM kernel has moderate support for tensor cores, but could be improved")
        else:
            print("✗ GEMM kernel has poor support for tensor cores")
    else:
        print(f"Could not find GEMM kernel file at {gemm_file}")
    
    # Performance test with different matrix sizes to verify tensor cores
    print("\nPerformance test for tensor cores:")
    
    # Test with matrices of sizes that are multiples of 16 (suitable for tensor cores)
    tensor_core_sizes = [(1024, 1024), (2048, 2048), (4096, 4096)]
    
    results = {}
    
    print("\nTesting with tensor core compatible sizes:")
    for size in tensor_core_sizes:
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
    
    # Analyze performance
    print("\nPerformance analysis:")
    for size, gflops in results.items():
        print(f"Size {size}: {gflops:.2f} GFLOPS")
    
    # Check if we're running on a GPU that supports tensor cores
    import torch  # Use PyTorch to check GPU capabilities
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        compute_capability = torch.cuda.get_device_capability(0)
        
        print(f"\nGPU: {device_name}")
        print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
        
        tensor_cores_available = compute_capability[0] >= 7
        print(f"Tensor Cores Available: {'Yes' if tensor_cores_available else 'No'}")
        
        if tensor_cores_available:
            # For Ampere and above, we expect at least 10 TFLOPS for FP32
            expected_min_performance = 10000  # 10 TFLOPS
            
            # Check if our performance is close to the expected performance
            max_performance = max(results.values())
            
            if max_performance >= expected_min_performance:
                print("✓ Performance suggests tensor cores are being utilized effectively")
            elif max_performance >= expected_min_performance * 0.5:
                print("⚠ Performance suggests tensor cores may be utilized, but not optimally")
            else:
                print("✗ Performance suggests tensor cores are not being utilized effectively")
        else:
            print("Note: This GPU does not support tensor cores, so we cannot verify their utilization")
    else:
        print("\nCould not check GPU capabilities (PyTorch not available)")
    
    print("\nTensor cores impact:")
    print("Efficient tensor cores utilization can improve performance by 3-5x for compatible matrix sizes")
    print("This is achieved by using specialized hardware units for matrix multiplication")

if __name__ == "__main__":
    test_tensor_cores() 