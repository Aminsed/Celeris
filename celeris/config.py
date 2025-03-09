"""
Configuration module for Celeris.

This module provides configuration options for Celeris to adapt to different
GPU architectures and optimize performance based on hardware capabilities.
"""

import os
import json
from pathlib import Path
import torch

# Default configuration
DEFAULT_CONFIG = {
    # Memory management
    "memory": {
        "max_workspace_size": 1024 * 1024 * 1024,  # 1 GB default workspace
        "memory_pool_enabled": True,
        "cudnn_benchmark": True,
    },
    
    # Performance tuning
    "performance": {
        "use_tensor_cores": True,  # Use tensor cores if available
        "use_reduced_precision": False,  # Use FP16/BF16 when possible
        "auto_tune": True,  # Auto-tune kernels for the specific GPU
        "optimize_memory_access": True,
    },
    
    # Architecture-specific settings
    "architecture": {
        "default_block_size": 128,  # Default CUDA block size
        "min_blocks_per_sm": 2,  # Minimum blocks per streaming multiprocessor
        "max_registers_per_thread": 64,  # Maximum registers per thread
        "use_shared_memory": True,  # Use shared memory for caching
    },
    
    # JIT compilation
    "jit": {
        "enabled": True,
        "cache_dir": str(Path.home() / ".celeris" / "kernel_cache"),
        "optimization_level": 3,  # 0-3, with 3 being the most aggressive
    }
}

# User configuration file path
USER_CONFIG_PATH = Path.home() / ".celeris" / "config.json"

# Global configuration
_config = DEFAULT_CONFIG.copy()

def load_config():
    """Load configuration from the user config file if it exists."""
    global _config
    
    if USER_CONFIG_PATH.exists():
        try:
            with open(USER_CONFIG_PATH, 'r') as f:
                user_config = json.load(f)
                
            # Update the default config with user settings
            for section, settings in user_config.items():
                if section in _config:
                    _config[section].update(settings)
                else:
                    _config[section] = settings
                    
            print(f"Loaded Celeris configuration from {USER_CONFIG_PATH}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    # Apply environment variable overrides
    _apply_env_overrides()
    
    # Adjust settings based on GPU capabilities
    _adjust_for_gpu()
    
    return _config

def _apply_env_overrides():
    """Apply configuration overrides from environment variables."""
    # Environment variables take the form CELERIS_SECTION_SETTING
    for key, value in os.environ.items():
        if key.startswith("CELERIS_"):
            parts = key[8:].lower().split('_', 1)
            if len(parts) == 2:
                section, setting = parts
                if section in _config and setting in _config[section]:
                    # Convert value to the appropriate type
                    orig_value = _config[section][setting]
                    if isinstance(orig_value, bool):
                        _config[section][setting] = value.lower() in ('true', '1', 'yes')
                    elif isinstance(orig_value, int):
                        _config[section][setting] = int(value)
                    elif isinstance(orig_value, float):
                        _config[section][setting] = float(value)
                    else:
                        _config[section][setting] = value

def _adjust_for_gpu():
    """Adjust configuration based on the detected GPU capabilities."""
    if not torch.cuda.is_available():
        # Disable GPU-specific features in CPU mode
        _config["performance"]["use_tensor_cores"] = False
        _config["performance"]["use_reduced_precision"] = False
        _config["jit"]["enabled"] = False
        return
    
    # Get GPU properties
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    compute_capability = torch.cuda.get_device_capability(device)
    
    # Adjust block size based on GPU architecture
    if compute_capability[0] >= 8:  # Ampere or newer
        _config["architecture"]["default_block_size"] = 256
    elif compute_capability[0] >= 7:  # Volta/Turing
        _config["architecture"]["default_block_size"] = 128
    else:  # Older architectures
        _config["architecture"]["default_block_size"] = 64
    
    # Adjust tensor core usage
    has_tensor_cores = compute_capability[0] >= 7
    _config["performance"]["use_tensor_cores"] = has_tensor_cores and _config["performance"]["use_tensor_cores"]
    
    # Adjust memory settings based on available GPU memory
    total_memory_gb = properties.total_memory / (1024**3)
    if total_memory_gb < 4:
        # For GPUs with less than 4GB memory, reduce workspace size
        _config["memory"]["max_workspace_size"] = 256 * 1024 * 1024  # 256 MB
    elif total_memory_gb < 8:
        _config["memory"]["max_workspace_size"] = 512 * 1024 * 1024  # 512 MB
    
    # Adjust register usage based on architecture
    if compute_capability[0] >= 8:  # Ampere or newer
        _config["architecture"]["max_registers_per_thread"] = 256
    elif compute_capability[0] >= 7:  # Volta/Turing
        _config["architecture"]["max_registers_per_thread"] = 128
    else:
        _config["architecture"]["max_registers_per_thread"] = 64

def get_config():
    """Get the current configuration."""
    return _config

def set_config(section, setting, value):
    """Set a configuration value."""
    if section not in _config:
        _config[section] = {}
    _config[section][setting] = value

def save_config():
    """Save the current configuration to the user config file."""
    # Create directory if it doesn't exist
    USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(USER_CONFIG_PATH, 'w') as f:
        json.dump(_config, f, indent=2)
    
    print(f"Saved Celeris configuration to {USER_CONFIG_PATH}")

# Initialize configuration
load_config() 