# Celeris In-Depth Performance and Profiling Summary

This document summarizes the detailed profiling and benchmarking results of Celeris. The analysis was performed using the `benchmarks/benchmark_profiling.py` script, which executed a series of core operations on different matrix sizes. All generated outputs (data tables, plots, and profiling logs) are available in the `docs/images` directory.

## GPU Environment

The benchmark was executed on the following GPU:

- **GPU Model**: NVIDIA RTX A6000
- **Compute Capability**: 8.6
- **Total Memory**: 47.41 GB
- **Tensor Cores**: Available

The environment summary (printed at the start of the benchmark script) is as follows:

```
Celeris detected GPU: NVIDIA RTX A6000
Compute Capability: 8.6
Total Memory: 47.41 GB
Tensor Cores Available: Yes
```

## Benchmarking Results

The in-depth benchmarking evaluated the performance of core operations including element-wise addition, element-wise multiplication, matrix multiplication (MatMul), and activation functions (ReLU, Sigmoid, Tanh) across three different matrix sizes: 512x512, 1024x1024, and 2048x2048.

Below is a summary of the results (average execution time per run and throughput in GOPS):

| Operation         | Matrix Size | Avg Time (s) | Throughput (GOPS) |
|-------------------|-------------|--------------|-------------------|
| Element-wise Add  | 512x512     | 0.000005     | 50.44             |
| Element-wise Add  | 1024x1024   | 0.000005     | 232.70            |
| Element-wise Add  | 2048x2048   | 0.000005     | 925.90            |
| Element-wise Mul  | 512x512     | 0.000005     | 54.43             |
| Element-wise Mul  | 1024x1024   | 0.000005     | 230.26            |
| Element-wise Mul  | 2048x2048   | 0.000005     | 911.51            |
| MatMul            | 512x512     | 0.000007     | 35.81             |
| MatMul            | 1024x1024   | 0.000008     | 128.60            |
| MatMul            | 2048x2048   | 0.000008     | 523.58            |
| ReLU              | 512x512     | 0.000005     | 50.44             |
| ReLU              | 1024x1024   | 0.000005     | 217.73            |
| ReLU              | 2048x2048   | 0.000005     | 875.23            |
| Sigmoid           | 512x512     | 0.000005     | 55.81             |
| Sigmoid           | 1024x1024   | 0.000005     | 232.70            |
| Sigmoid           | 2048x2048   | 0.000004     | 945.82            |
| Tanh              | 512x512     | 0.000005     | 55.53             |
| Tanh              | 1024x1024   | 0.000004     | 239.02            |
| Tanh              | 2048x2048   | 0.000005     | 925.90            |

*Note: The timing values are the average over 10 runs for each operation and matrix size.*

The complete benchmark data is also saved in a markdown table at `docs/images/benchmark_profiling.md`.

## Generated Visualizations

The following performance plots were generated and are available in the `docs/images` directory:

- **Element-wise Add Performance**: `element-wise_add_performance.png`
- **Element-wise Mul Performance**: `element-wise_mul_performance.png`
- **MatMul Performance**: `matmul_performance.png`
- **ReLU Performance**: `relu_performance.png`
- **Sigmoid Performance**: `sigmoid_performance.png`
- **Tanh Performance**: `tanh_performance.png`

These plots illustrate how throughput scales with matrix size for the corresponding operations.

## cProfile Analysis

A detailed cProfile analysis of a 1024x1024 matrix multiplication was performed. The profiling results, which include function call breakdowns and cumulative execution times, are available in the file:

- `matmul_cprofile.txt`

This profiling data helps identify which parts of the matrix multiplication operation are most time-consuming and provides insights into potential optimization opportunities.

## Conclusion

The in-depth profiling and benchmarking provide a comprehensive view of Celeris' performance on a high-end GPU (NVIDIA RTX A6000). The results demonstrate the efficiency of basic tensor operations and matrix computations, offering valuable insights for further optimizations. The detailed data and visualizations serve as a robust reference for understanding Celeris' internal processing and performance characteristics. 