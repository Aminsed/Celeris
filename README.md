# Celeris: GPU-Accelerated Matrix Operations Library

Celeris is a high-performance GPU library designed for rapid matrix operations, optimized for machine learning and scientific computing workloads. Inspired by [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), Celeris aims to provide efficient matrix operations with a clean, accessible implementation.

## Features

- Optimized GPU-accelerated matrix operations
- Tensor core support for compatible NVIDIA GPUs
- Automatic memory management
- PyTorch-like API for easy integration
- Support for various data types (FP32, FP16, BF16)
- Dynamic kernel optimization for different GPU architectures

## Inspiration and Techniques

Celeris draws inspiration from [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), a clean and efficient GEMM kernel library developed by DeepSeek AI. Several key techniques have been adopted:

- **Lightweight JIT Compilation**: Compiling kernels at runtime for optimal performance without heavy installation requirements
- **Warp-Specialized Execution**: Overlapping data movement and computation for maximum efficiency
- **Dynamic Block Scheduling**: Optimizing workload distribution across GPU SMs
- **Unaligned Block Sizes**: Supporting non-power-of-2 block sizes for better SM utilization
- **Automatic Data Type Selection**: Choosing optimal precision based on hardware capabilities

While DeepGEMM focuses on FP8 operations with fine-grained scaling for Hopper architecture, Celeris provides a more general-purpose library supporting various data types and GPU architectures.

## Installation

### Prerequisites

- Python 3.7+
- CUDA Toolkit 11.0+ (for GPU support)
- A compatible NVIDIA GPU
- PyTorch 1.8+ (Celeris uses PyTorch as a backend)
- NumPy

### Install from Source

```bash
# Clone the repository
git clone https://github.com/Aminsed/Celeris.git
cd Celeris

# Install dependencies
pip install numpy torch torchvision

# Install Celeris in development mode
pip install -e .
```

### Verify Installation

After installation, you can verify that Celeris is working correctly by running the GPU compatibility test:

```bash
python examples/gpu_compatibility_test.py
```

This will detect your GPU, run a series of tests, and report performance metrics.

## Quick Start

```python
import celeris
import numpy as np
import torch  # Celeris uses PyTorch as a backend

# Create tensors
a = celeris.randn(1000, 1000)
b = celeris.randn(1000, 1000)

# Perform operations
c = celeris.add(a, b)
d = celeris.mul(a, b)
e = celeris.matmul(a, b)

# Convert to/from NumPy
numpy_array = np.random.rand(100, 100).astype(np.float32)
tensor = celeris.from_numpy(numpy_array)
result = tensor.cpu().detach().numpy()  # Convert tensor to numpy
```

## Example: Linear Regression

```python
import celeris
import numpy as np
import torch

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1).astype(np.float32) * 10
y = 2 * X + 1 + np.random.randn(100, 1).astype(np.float32)

# Convert to Celeris tensors
X_tensor = celeris.from_numpy(X)
y_tensor = celeris.from_numpy(y)

# Initialize model parameters
W = celeris.randn([1, 1])
b = celeris.zeros([1])

# Training parameters
learning_rate = 0.01
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    y_pred = celeris.matmul(X_tensor, W) + b
    
    # Compute loss (MSE)
    loss = torch.mean((y_pred - y_tensor) ** 2)
    
    # Compute gradients manually
    dW = 2 * celeris.matmul(X_tensor.t(), (y_pred - y_tensor)) / X.shape[0]
    db = 2 * torch.mean(y_pred - y_tensor)
    
    # Update parameters
    W = W - learning_rate * dW
    b = b - learning_rate * db
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Convert final parameters to numpy
W_numpy = W.cpu().detach().numpy()
b_numpy = b.cpu().detach().numpy()
print(f"Final parameters: W = {W_numpy[0][0]:.4f}, b = {b_numpy[0]:.4f}")
```

## Example: Neural Network for MNIST

```python
import celeris
import torch
import torch.nn.functional as F
import numpy as np

# Load and preprocess data (simplified)
# In a real example, you would load MNIST data

# Initialize weights with small random values
W1 = celeris.randn([784, 128]) * 0.01  # Input -> Hidden
b1 = celeris.zeros([128])
W2 = celeris.randn([128, 10]) * 0.01   # Hidden -> Output
b2 = celeris.zeros([10])

# Training parameters
learning_rate = 0.01
batch_size = 100

# Training loop (simplified)
# Forward pass
z1 = celeris.matmul(X_batch, W1) + b1
a1 = celeris.relu(z1)
z2 = celeris.matmul(a1, W2) + b2

# Use PyTorch's functions for numerical stability
log_probs = F.log_softmax(z2, dim=1)
loss = F.nll_loss(log_probs, labels)

# Compute gradients
probs = F.softmax(z2, dim=1)
# ... gradient computation and parameter updates
```

## GPU Compatibility Testing

To test if Celeris works correctly with your GPU:

```bash
python examples/gpu_compatibility_test.py
```

This will run a series of tests to verify compatibility and performance on your specific GPU hardware.

## Advanced Configuration

Celeris can be configured for optimal performance on your specific hardware:

```python
from celeris.config import get_config, set_config

# Get current configuration
config = get_config()

# Modify configuration
config["performance"]["use_tensor_cores"] = True
config["performance"]["use_reduced_precision"] = True

# Apply configuration
set_config(config)
```

## Available Examples

The `examples/` directory contains various examples demonstrating Celeris capabilities:

- `linear_regression.py`: Simple linear regression model
- `classification_example.py`: Basic classification using PyTorch-like API
- `regression_example.py`: Regression using PyTorch-like API
- `mnist_classifier.py`: Neural network for MNIST digit classification
- `cnn_example.py`: Convolutional neural network for image classification
- `lstm_example.py`: LSTM network for sequence prediction
- `transformer_example.py`: Transformer model for sequence classification
- `gpu_compatibility_test.py`: Tests GPU compatibility and performance

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'celeris'**
   - Make sure you've installed Celeris with `pip install -e .`
   - Check that you're running Python from the correct environment

2. **CUDA not available**
   - Verify CUDA installation with `torch.cuda.is_available()`
   - Check that your NVIDIA drivers are up to date

3. **Numerical instability in training**
   - Use PyTorch's numerically stable functions like `F.log_softmax` and `F.nll_loss`
   - Initialize weights with small random values (multiply by 0.01)

## License

[MIT License](LICENSE)

## Acknowledgements

We would like to thank the developers of [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) for their innovative work on efficient matrix multiplication kernels, which has been a significant inspiration for this project. 