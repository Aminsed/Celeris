# Celeris vs. PyTorch Performance Benchmark Summary

This document summarizes the performance comparison between Celeris and PyTorch for various operations.

## Matrix Multiplication (GEMM) Performance

### Power-of-2 Matrix Sizes

| Size | Celeris (GFLOPS) | PyTorch (GFLOPS) | Speedup (Celeris/PyTorch) |
|------|------------------|------------------|---------------------------|
| 1024 | 1,779.23         | 16,205.83        | 0.11x                     |
| 2048 | 1,904.80         | 22,764.14        | 0.08x                     |
| 4096 | 1,880.60         | 25,885.78        | 0.07x                     |

### Non-Power-of-2 Matrix Sizes

| Size | Celeris (GFLOPS) | PyTorch (GFLOPS) | Speedup (Celeris/PyTorch) |
|------|------------------|------------------|---------------------------|
| 918  | 1,677.85         | 10,330.47        | 0.16x                     |
| 1023 | 1,771.86         | 17,513.33        | 0.10x                     |
| 1025 | 1,690.23         | 16,241.66        | 0.10x                     |
| 1836 | 1,891.57         | 20,557.85        | 0.09x                     |
| 2047 | 1,864.36         | 22,551.27        | 0.08x                     |
| 2049 | 1,838.85         | 21,064.63        | 0.09x                     |
| 3681 | 1,835.97         | 25,160.92        | 0.07x                     |
| 4095 | 1,862.72         | 25,768.93        | 0.07x                     |
| 4097 | 1,843.25         | 25,140.46        | 0.07x                     |

## Element-wise Operations Performance

| Operation | Shape | PyTorch (GOPS) | Celeris (GOPS) | Speedup (PyTorch/Celeris) |
|-----------|-------|----------------|----------------|---------------------------|
| Add       | 1024  | 39.48          | 7.06           | 5.59x                     |
| Add       | 2048  | 53.63          | 8.41           | 6.38x                     |
| Add       | 1023  | 41.57          | 7.01           | 5.93x                     |
| Add       | 1025  | 41.73          | 5.31           | 7.86x                     |
| Mul       | 1024  | 41.41          | 7.03           | 5.89x                     |
| Mul       | 2048  | 53.96          | 8.41           | 6.42x                     |
| Mul       | 1023  | 41.49          | 7.02           | 5.91x                     |
| Mul       | 1025  | 41.73          | 5.30           | 7.87x                     |

## Activation Functions Performance

| Operation | Shape | PyTorch (GOPS) | Celeris (GOPS) | Speedup (PyTorch/Celeris) |
|-----------|-------|----------------|----------------|---------------------------|
| ReLU      | 1024  | 52.99          | 7.38           | 7.18x                     |
| ReLU      | 2048  | 76.03          | 8.81           | 8.63x                     |
| ReLU      | 1023  | 52.63          | 7.33           | 7.18x                     |
| ReLU      | 1025  | 53.22          | 5.44           | 9.78x                     |
| Sigmoid   | 1024  | 52.48          | 7.38           | 7.11x                     |
| Sigmoid   | 2048  | 76.09          | 8.82           | 8.63x                     |
| Sigmoid   | 1023  | 52.89          | 7.31           | 7.23x                     |
| Sigmoid   | 1025  | 54.27          | 5.45           | 9.96x                     |
| Tanh      | 1024  | 52.48          | 7.36           | 7.13x                     |
| Tanh      | 2048  | 76.09          | 8.82           | 8.63x                     |
| Tanh      | 1023  | 52.26          | 7.34           | 7.12x                     |
| Tanh      | 1025  | 53.48          | 5.46           | 9.79x                     |

## Analysis

### Matrix Multiplication (GEMM)

- Celeris achieves consistent performance around 1,700-1,900 GFLOPS for GEMM operations across different matrix sizes.
- PyTorch significantly outperforms Celeris for GEMM operations, achieving 10,000-26,000 GFLOPS.
- The performance gap is smaller for smaller matrices and non-power-of-2 sizes.
- Celeris performs best relative to PyTorch for the smallest non-power-of-2 size (918x918), achieving 16% of PyTorch's performance.

### Element-wise Operations

- PyTorch outperforms Celeris by 5.5-8x for element-wise operations.
- Celeris shows more consistent performance across different matrix sizes for element-wise operations compared to activation functions.
- The performance gap is smaller for power-of-2 sizes compared to non-power-of-2 sizes.

### Activation Functions

- PyTorch outperforms Celeris by 7-10x for activation functions.
- The performance gap is larger for non-power-of-2 sizes, especially for 1025x1025.
- Celeris shows similar performance across different activation functions (ReLU, Sigmoid, Tanh).

## Conclusion

Based on the benchmark results, Celeris does not outperform PyTorch in any of the tested operations. However, Celeris shows the smallest performance gap in the following scenarios:

1. **Matrix Multiplication with Small Non-Power-of-2 Sizes**: For the 918x918 matrix size, Celeris achieves 16% of PyTorch's performance, which is the best relative performance observed.

2. **Element-wise Operations with Power-of-2 Sizes**: For element-wise addition and multiplication with 1024x1024 matrices, Celeris achieves about 18% of PyTorch's performance.

3. **Consistent Performance Across Matrix Sizes**: Celeris shows more consistent performance across different matrix sizes, especially for GEMM operations, while PyTorch's performance varies more significantly.

The superior performance of PyTorch can be attributed to its use of highly optimized CUDA libraries (like cuBLAS and cuDNN) that have been developed and refined over many years by NVIDIA. These libraries leverage proprietary optimizations and are specifically tuned for NVIDIA GPUs.

## Potential Areas for Improvement in Celeris

1. **Optimize Tensor Core Usage**: Ensure Celeris is fully utilizing tensor cores for compatible operations.
2. **Implement Mixed Precision Operations**: Add support for FP16 and BF16 operations to leverage tensor cores more effectively.
3. **Enhance Memory Access Patterns**: Optimize memory access patterns to better utilize GPU memory bandwidth.
4. **Algorithm Selection**: Implement multiple algorithms for each operation and select the optimal one based on input size and characteristics.
5. **Kernel Fusion**: Implement kernel fusion for common operation sequences to reduce memory traffic.
6. **Specialized Implementations**: Develop specialized implementations for specific matrix sizes and shapes that are commonly used in deep learning workloads. 