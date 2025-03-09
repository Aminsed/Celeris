"""
Utility functions for Celeris.
"""

from .gpu_utils import (
    get_gpu_info,
    get_memory_info,
    clear_gpu_cache,
    can_use_tensor_cores,
    should_use_reduced_precision,
    get_optimal_dtype,
    benchmark_matmul,
    print_gpu_summary,
    get_optimal_block_size,
    get_optimal_grid_size
)

__all__ = [
    'get_gpu_info',
    'get_memory_info',
    'clear_gpu_cache',
    'can_use_tensor_cores',
    'should_use_reduced_precision',
    'get_optimal_dtype',
    'benchmark_matmul',
    'print_gpu_summary',
    'get_optimal_block_size',
    'get_optimal_grid_size'
] 