# Celeris: A GPU Library for Rapid Matrix Operations

Celeris is a high-performance GPU library designed for rapid matrix operations and neural network computations. It aims to provide a familiar interface similar to PyTorch while leveraging dynamic GPU detection and configurable performance optimizations.

---

## Table of Contents

- [Overview](#overview)
- [Features and Techniques](#features-and-techniques)
- [Installation and Setup](#installation-and-setup)
- [Benchmarking](#benchmarking)
  - [Benchmark Results](#benchmark-results)
- [Testing and Examples](#testing-and-examples)
- [Contribution and Future Work](#contribution-and-future-work)
- [License](#license)
- [Contact](#contact)

---

## Repository

Clone the repository using the following command:

```bash
git clone git@github.com:Aminsed/Celeris.git
```

## Overview

Celeris is a high-performance GPU library designed for rapid matrix operations and neural network computations. It automatically detects GPU capabilities and optimizes runtime performance accordingly. With a familiar interface modeled after PyTorch, Celeris eases the process of deploying machine learning models on GPUs.

## Features and Techniques

- **Dynamic GPU Detection**: Automatically detects GPU properties at import time and adjusts configurations accordingly. See `celeris/__init__.py` and `celeris/config.py`.
- **Configurable Performance Tuning**: Customize memory management, kernel parameters, and precision settings via the `celeris/config.py` module.
- **Optimized Operations**: Provides core operations (element-wise add, multiply, matrix multiplication, and activation functions) with a design that allows future incorporation of advanced CUDA kernels, tensor core optimizations, and kernel fusion.
- **Interoperability with PyTorch**: The `celeris.nn` module mimics the PyTorch `nn` module, simplifying integration and migration.
- **Comprehensive Benchmark Suite**: Includes multiple scripts to benchmark operations, evaluate data type performance, test non-power-of-2 sizes, and assess tensor core benefits. Benchmarking scripts can be found in the `benchmarks/` and `examples/` directories.

## Installation and Setup
- For detailed installation instructions, please refer to our [Installation Guide](docs/INSTALLATION.md).

The basic steps are:
  1. Clone the repository and navigate into the project folder.
  2. Set up a Python virtual environment to keep your environment isolated.
  3. Install the dependencies and Celeris in editable mode.
  4. (Optional) Run the test suite to ensure everything is working fine.

Follow the [Installation Guide](docs/INSTALLATION.md) for step-by-step instructions, troubleshooting tips, and next steps.

## Benchmarking

Celeris includes several benchmark and profiling scripts to evaluate its performance.

### In-Depth Benchmarking and Profiling Results

The in-depth profiling and benchmarking script (`benchmarks/benchmark_profiling.py`) was executed, generating detailed performance data and visualizations for core operations.

The following table summarizes the average execution time (in seconds) and throughput (in GOPS) for various core operations on different matrix sizes:

[Insert benchmark table from `docs/images/benchmark_profiling.md` here]

The data above highlights the performance of key operations:
 - **Element-wise Add and Mul**: Basic tensor operations.
 - **MatMul**: Matrix multiplication showing computation efficiency.
 - **Activation Functions (ReLU, Sigmoid, Tanh)**: Performance of common nonlinear operations.

The following plots have been generated and are available in the `docs/images` directory:
 - `element-wise_add_performance.png`
 - `element-wise_mul_performance.png`
 - `matmul_performance.png`
 - `relu_performance.png`
 - `sigmoid_performance.png`
 - `tanh_performance.png`

A detailed cProfile analysis of a 1024x1024 matrix multiplication was also performed. The profiling results, documenting function call breakdowns and cumulative execution times, are available in `docs/images/matmul_cprofile.txt`.

These comprehensive benchmarks and profiling insights demonstrate the performance characteristics of Celeris and provide guidance for further optimization and development.

## Testing and Examples

Celeris comes with a suite of examples to demonstrate usage and provide baseline performance data:

- **Classification Example**: Train an MLP on the MNIST dataset. ([examples/classification_example.py](examples/classification_example.py))
- **Regression Example**: Perform linear regression on synthetic data. ([examples/regression_example.py](examples/regression_example.py))
- **LSTM Example**: Predict a sine wave using an LSTM model. ([examples/lstm_example.py](examples/lstm_example.py))
- **Transformer Example**: Train a transformer encoder for sequence classification. ([examples/transformer_example.py](examples/transformer_example.py))
- **CNN Example**: Train a CNN on CIFAR-10. ([examples/cnn_example.py](examples/cnn_example.py))

To run an example, execute a command such as:

```bash
python examples/classification_example.py
```

## Contribution and Future Work

We welcome contributions and suggestions. Planned future enhancements include:

- **Advanced CUDA Kernels**: Integration of custom CUDA kernels with tensor core and kernel fusion optimizations.
- **Mixed Precision Operations**: Full support for FP16 and BF16 computations to accelerate training and inference.
- **Auto-Tuning**: Dynamic selection of optimal kernel parameters based on real-time GPU performance analysis.
- **Expanded Benchmark Suite**: Additional tests covering a broader range of operations and GPU architectures to further detail performance characteristics.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or contributions, please contact [Your Name] at [your.email@example.com].

---

*This documentation provides a comprehensive overview of the Celeris library, detailed benchmarking results, and instructions for testing and contributions. For more details, please refer to the source code and accompanying benchmark scripts.* 