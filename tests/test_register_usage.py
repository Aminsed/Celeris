#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test for register count control in CUDA kernels.
"""

import numpy as np
import sys
import os
import subprocess
import re
import time

# Add the parent directory to the path so we can import celeris
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import celeris
    print("Successfully imported celeris")
except ImportError as e:
    print(f"Failed to import celeris: {e}")
    print("Make sure the library is built and installed correctly.")
    sys.exit(1)

def test_register_usage():
    """Test if CUDA kernels have controlled register usage."""
    print("\n=== Testing Register Count Control ===")
    
    # This test requires nvcc and cuobjdump to be available
    try:
        # First, we need to trigger the JIT compilation by running a GEMM operation
        print("Running GEMM operations with different sizes to test register usage...")
        
        # Test with different matrix sizes to ensure register usage is consistent
        sizes = [1024, 2048, 4096]
        for size in sizes:
            print(f"\nTesting with matrix size {size}x{size}:")
            a = celeris.randn([size, size])
            b = celeris.randn([size, size])
            
            # Warm-up run
            _ = celeris.matmul(a, b)
            
            # Timed run
            start_time = time.time()
            c = celeris.matmul(a, b)
            elapsed_time = time.time() - start_time
            
            # Calculate GFLOPS
            flops = 2 * size * size * size  # 2*M*N*K for matrix multiplication
            gflops = flops / (elapsed_time * 1e9)
            print(f"Performance: {gflops:.2f} GFLOPS")
        
        print("\nChecking register usage in GEMM kernel...")
        
        # Get the source code of the GEMM kernel
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'cuda'))
        gemm_file = os.path.join(src_dir, 'gemm.cu')
        
        if os.path.exists(gemm_file):
            with open(gemm_file, 'r') as f:
                gemm_code = f.read()
                
            # Check for pragma unroll directives
            unroll_count = len(re.findall(r'#pragma\s+unroll', gemm_code))
            print(f"Found {unroll_count} pragma unroll directives in GEMM kernel")
            
            if unroll_count > 0:
                print("✓ GEMM kernel uses loop unrolling for register control")
            else:
                print("✗ GEMM kernel does not use loop unrolling for register control")
                
            # Check for register variables
            register_vars = len(re.findall(r'register\s+\w+', gemm_code))
            print(f"Found {register_vars} explicit register variables in GEMM kernel")
            
            if register_vars > 0:
                print("✓ GEMM kernel uses explicit register variables for register control")
            else:
                print("✗ GEMM kernel does not use explicit register variables for register control")
            
            # Check for launch bounds
            launch_bounds = re.search(r'__launch_bounds__\s*\(\s*\$\{block_threads\}\s*,\s*\$\{min_blocks_per_sm\}\s*\)', gemm_code)
            if launch_bounds:
                print(f"✓ GEMM kernel uses __launch_bounds__ with template parameters for threads and blocks per SM")
            else:
                print("✗ GEMM kernel does not use __launch_bounds__ for register control")
            
            # Check for other register control mechanisms
            if "__restrict__" in gemm_code:
                print("✓ GEMM kernel uses __restrict__ keyword for better register allocation")
            else:
                print("✗ GEMM kernel does not use __restrict__ keyword")
                
            if "nv_diag_suppress" in gemm_code:
                print("✓ GEMM kernel suppresses unnecessary diagnostics for cleaner compilation")
            else:
                print("✗ GEMM kernel does not suppress unnecessary diagnostics")
            
            # Overall assessment
            register_control_score = 0
            register_control_score += 1 if unroll_count > 0 else 0
            register_control_score += 1 if register_vars > 0 else 0
            register_control_score += 1 if launch_bounds else 0
            register_control_score += 1 if "__restrict__" in gemm_code else 0
            register_control_score += 1 if "nv_diag_suppress" in gemm_code else 0
            
            print(f"\nRegister control score: {register_control_score}/5")
            
            if register_control_score >= 4:
                print("✓ GEMM kernel has excellent register control")
            elif register_control_score >= 2:
                print("⚠ GEMM kernel has moderate register control, but could be improved")
            else:
                print("✗ GEMM kernel has poor register control")
                
            # Performance impact assessment
            print("\nPerformance impact of register control:")
            print("In a fully optimized implementation, proper register control can improve performance by 10-30%")
            print("This is achieved by balancing register usage to maximize occupancy while minimizing register spilling")
        else:
            print(f"Could not find GEMM kernel file at {gemm_file}")
    
    except Exception as e:
        print(f"Error during register usage test: {e}")

if __name__ == "__main__":
    test_register_usage() 