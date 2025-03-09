"""
GPU utility functions for Celeris.

This module provides utility functions for GPU detection, memory management,
and performance optimization across different GPU architectures.
"""

import torch
import numpy as np
import os
import platform
from pathlib import Path
import subprocess
import re
from ..config import get_config

def get_gpu_info():
    """
    Get detailed information about available GPUs.
    
    Returns:
        list: List of dictionaries containing GPU information.
    """
    if not torch.cuda.is_available():
        return []
    
    gpu_info = []
    device_count = torch.cuda.device_count()
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        capability = torch.cuda.get_device_capability(i)
        
        info = {
            "index": i,
            "name": props.name,
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "total_memory": props.total_memory,
            "memory_gb": props.total_memory / (1024**3),
            "multi_processor_count": props.multi_processor_count,
            "warp_size": 32,  # Standard warp size for NVIDIA GPUs
            "has_tensor_cores": capability[0] >= 7,
            "architecture": _get_architecture_name(capability)
        }
        
        # Check if max_threads_per_multi_processor attribute exists
        if hasattr(props, 'max_threads_per_multi_processor'):
            info["max_threads_per_mp"] = props.max_threads_per_multi_processor
        else:
            # Default value based on compute capability
            if capability[0] >= 7:
                info["max_threads_per_mp"] = 1024  # Typical for newer architectures
            else:
                info["max_threads_per_mp"] = 2048  # Typical for older architectures
        
        gpu_info.append(info)
    
    return gpu_info

def _get_architecture_name(capability):
    """Map compute capability to architecture name."""
    major, minor = capability
    
    if major == 8:
        return "Ampere"
    elif major == 7:
        if minor == 5:
            return "Turing"
        else:
            return "Volta"
    elif major == 6:
        return "Pascal"
    elif major == 5:
        return "Maxwell"
    elif major == 3:
        return "Kepler"
    else:
        return f"Unknown (Compute {major}.{minor})"

def get_optimal_block_size(tensor_shape=None):
    """
    Get the optimal CUDA block size for the current GPU.
    
    Args:
        tensor_shape (tuple, optional): Shape of the tensor being processed.
        
    Returns:
        int: Optimal block size.
    """
    config = get_config()
    default_block_size = config["architecture"]["default_block_size"]
    
    if not torch.cuda.is_available():
        return default_block_size
    
    # Start with the default block size from config
    block_size = default_block_size
    
    # For very small tensors, use smaller block sizes
    if tensor_shape is not None:
        tensor_size = np.prod(tensor_shape)
        if tensor_size < 1024:
            block_size = min(block_size, 64)
        elif tensor_size < 4096:
            block_size = min(block_size, 128)
    
    # Ensure block size is a multiple of warp size (typically 32)
    warp_size = 32  # Standard warp size for NVIDIA GPUs
    block_size = (block_size // warp_size) * warp_size
    
    return block_size

def get_optimal_grid_size(tensor_shape, block_size=None):
    """
    Calculate the optimal grid size for CUDA kernels.
    
    Args:
        tensor_shape (tuple): Shape of the tensor being processed.
        block_size (int, optional): Block size to use. If None, will be determined automatically.
        
    Returns:
        tuple: Grid dimensions (x, y, z).
    """
    if block_size is None:
        block_size = get_optimal_block_size(tensor_shape)
    
    # Calculate total number of elements
    total_elements = np.prod(tensor_shape)
    
    # Calculate grid size
    grid_size = (total_elements + block_size - 1) // block_size
    
    # For 1D operations
    if len(tensor_shape) == 1 or (len(tensor_shape) == 2 and tensor_shape[1] == 1):
        return (grid_size, 1, 1)
    
    # For 2D operations (e.g., matrices)
    elif len(tensor_shape) == 2:
        rows, cols = tensor_shape
        grid_x = (cols + block_size - 1) // block_size
        grid_y = (rows + block_size - 1) // block_size
        return (grid_x, grid_y, 1)
    
    # For 3D operations
    elif len(tensor_shape) == 3:
        d1, d2, d3 = tensor_shape
        grid_x = (d3 + block_size - 1) // block_size
        grid_y = (d2 + block_size - 1) // block_size
        grid_z = d1
        return (grid_x, grid_y, grid_z)
    
    # For higher dimensions, flatten to 1D
    else:
        return (grid_size, 1, 1)

def get_memory_info():
    """
    Get current GPU memory usage.
    
    Returns:
        dict: Dictionary containing memory information.
    """
    if not torch.cuda.is_available():
        return {"available": 0, "total": 0, "used": 0, "free": 0}
    
    device = torch.cuda.current_device()
    
    # Get total memory from device properties
    props = torch.cuda.get_device_properties(device)
    total_memory = props.total_memory
    
    # Get current memory usage
    memory_reserved = torch.cuda.memory_reserved(device)
    memory_allocated = torch.cuda.memory_allocated(device)
    
    # Calculate free memory
    free_memory = total_memory - memory_reserved
    
    return {
        "total": total_memory,
        "reserved": memory_reserved,
        "allocated": memory_allocated,
        "free": free_memory,
        "total_gb": total_memory / (1024**3),
        "reserved_gb": memory_reserved / (1024**3),
        "allocated_gb": memory_allocated / (1024**3),
        "free_gb": free_memory / (1024**3)
    }

def clear_gpu_cache():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def can_use_tensor_cores():
    """
    Check if tensor cores can be used on the current GPU.
    
    Returns:
        bool: True if tensor cores are available and enabled in config.
    """
    if not torch.cuda.is_available():
        return False
    
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    
    # Tensor cores are available on Volta (7.0) and newer architectures
    has_tensor_cores = capability[0] >= 7
    
    # Check if tensor cores are enabled in config
    config = get_config()
    use_tensor_cores = config["performance"]["use_tensor_cores"]
    
    return has_tensor_cores and use_tensor_cores

def should_use_reduced_precision():
    """
    Check if reduced precision (FP16/BF16) should be used.
    
    Returns:
        bool: True if reduced precision should be used.
    """
    if not torch.cuda.is_available():
        return False
    
    config = get_config()
    use_reduced_precision = config["performance"]["use_reduced_precision"]
    
    # Only use reduced precision if it's enabled in config
    if not use_reduced_precision:
        return False
    
    # Check if the GPU supports reduced precision efficiently
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    
    # Volta (7.0) and newer architectures have good FP16 support
    # Ampere (8.0) and newer have good BF16 support
    return capability[0] >= 7

def get_optimal_dtype():
    """
    Get the optimal data type for the current GPU.
    
    Returns:
        torch.dtype: Optimal data type.
    """
    if not torch.cuda.is_available():
        return torch.float32
    
    if should_use_reduced_precision():
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        
        # Ampere (8.0) and newer have good BF16 support
        if capability[0] >= 8 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    else:
        return torch.float32

def benchmark_matmul(size=1024, dtype=None, iterations=10):
    """
    Benchmark matrix multiplication on the current GPU.
    
    Args:
        size (int): Size of the square matrices.
        dtype (torch.dtype, optional): Data type to use. If None, will use optimal dtype.
        iterations (int): Number of iterations to run.
        
    Returns:
        dict: Benchmark results.
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    if dtype is None:
        dtype = get_optimal_dtype()
    
    # Create random matrices
    a = torch.randn(size, size, dtype=dtype, device="cuda")
    b = torch.randn(size, size, dtype=dtype, device="cuda")
    
    # Warm-up
    for _ in range(5):
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
    
    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iterations):
        c = torch.matmul(a, b)
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
    
    # Calculate FLOPS (2*N^3 operations for matrix multiplication)
    flops = 2 * size**3 * iterations
    gflops = flops / elapsed_time / 1e9
    
    return {
        "size": size,
        "dtype": str(dtype).split(".")[-1],
        "iterations": iterations,
        "time_seconds": elapsed_time,
        "time_per_iteration_ms": elapsed_time * 1000 / iterations,
        "gflops": gflops
    }

def print_gpu_summary():
    """Print a summary of GPU information."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Running in CPU-only mode.")
        return
    
    gpu_info = get_gpu_info()
    memory_info = get_memory_info()
    
    print("\n===== GPU Summary =====")
    for i, gpu in enumerate(gpu_info):
        print(f"GPU {i}: {gpu['name']} ({gpu['architecture']})")
        print(f"  Compute Capability: {gpu['compute_capability']}")
        print(f"  Memory: {gpu['memory_gb']:.2f} GB")
        print(f"  Tensor Cores: {'Available' if gpu['has_tensor_cores'] else 'Not Available'}")
        print(f"  SMs: {gpu['multi_processor_count']}")
    
    print("\n=== Memory Usage ===")
    print(f"Total: {memory_info['total_gb']:.2f} GB")
    print(f"Reserved: {memory_info['reserved_gb']:.2f} GB")
    print(f"Allocated: {memory_info['allocated_gb']:.2f} GB")
    print(f"Free: {memory_info['free_gb']:.2f} GB")
    
    print("\n=== Configuration ===")
    config = get_config()
    print(f"Tensor Cores: {'Enabled' if can_use_tensor_cores() else 'Disabled'}")
    print(f"Reduced Precision: {'Enabled' if should_use_reduced_precision() else 'Disabled'}")
    print(f"Optimal Data Type: {get_optimal_dtype()}")
    print(f"Default Block Size: {config['architecture']['default_block_size']}")
    
    print("\n=== Quick Benchmark ===")
    benchmark = benchmark_matmul(1024)
    print(f"1024x1024 MatMul: {benchmark['gflops']:.2f} GFLOPS ({benchmark['time_per_iteration_ms']:.2f} ms)")
    print("========================\n") 