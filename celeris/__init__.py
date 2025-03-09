import torch
import numpy as np
import os
import platform

# GPU detection and capabilities
def _detect_gpu_capabilities():
    """Detect GPU capabilities and set appropriate environment variables."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Celeris will run in CPU-only mode.")
        return {
            "device": "cpu",
            "compute_capability": None,
            "name": None,
            "memory": None,
            "tensor_cores": False
        }
    
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    device_capability = torch.cuda.get_device_capability(current_device)
    device_properties = torch.cuda.get_device_properties(current_device)
    
    # Convert compute capability to string format (e.g., "7.5")
    compute_capability = f"{device_capability[0]}.{device_capability[1]}"
    
    # Check for tensor cores (available in Volta, Turing, Ampere, and newer)
    has_tensor_cores = device_capability[0] >= 7
    
    # Get total memory in GB
    total_memory_gb = device_properties.total_memory / (1024**3)
    
    capabilities = {
        "device": "cuda",
        "compute_capability": compute_capability,
        "name": device_name,
        "memory": total_memory_gb,
        "tensor_cores": has_tensor_cores,
        "device_count": device_count
    }
    
    print(f"Celeris detected GPU: {device_name}")
    print(f"Compute Capability: {compute_capability}")
    print(f"Total Memory: {total_memory_gb:.2f} GB")
    print(f"Tensor Cores Available: {'Yes' if has_tensor_cores else 'No'}")
    
    # Set environment variables for JIT compilation
    os.environ["CELERIS_COMPUTE_CAPABILITY"] = compute_capability
    os.environ["CELERIS_HAS_TENSOR_CORES"] = "1" if has_tensor_cores else "0"
    
    return capabilities

# Run GPU detection at import time
GPU_CAPABILITIES = _detect_gpu_capabilities()

def from_numpy(arr):
    """Convert a NumPy array to a Celeris tensor (stub using PyTorch)."""
    device = "cuda" if GPU_CAPABILITIES["device"] == "cuda" else "cpu"
    return torch.tensor(arr, device=device)


def randn(*shape):
    """Generate a random Celeris tensor (stub using PyTorch)."""
    device = "cuda" if GPU_CAPABILITIES["device"] == "cuda" else "cpu"
    return torch.randn(*shape, device=device)


def add(x, y):
    """Element-wise addition (stub)."""
    return x + y


def mul(x, y):
    """Element-wise multiplication (stub)."""
    return x * y


def matmul(x, y):
    """Matrix multiplication (stub using PyTorch)."""
    return torch.matmul(x, y)


def relu(x):
    """ReLU activation (stub using PyTorch)."""
    return torch.relu(x)


def sigmoid(x):
    """Sigmoid activation (stub using PyTorch)."""
    return torch.sigmoid(x)


def tanh(x):
    """Tanh activation (stub using PyTorch)."""
    return torch.tanh(x)


def zeros(*shape):
    """Return a tensor filled with zeros."""
    device = "cuda" if GPU_CAPABILITIES["device"] == "cuda" else "cpu"
    return torch.zeros(*shape, device=device)


def ones(*shape):
    """Return a tensor filled with ones."""
    device = "cuda" if GPU_CAPABILITIES["device"] == "cuda" else "cpu"
    return torch.ones(*shape, device=device)

# Import nn module
from . import nn
