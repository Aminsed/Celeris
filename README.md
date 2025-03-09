# Celeris: GPU-Accelerated Matrix Operations Library

Celeris is a high-performance GPU library designed for rapid matrix operations, optimized for machine learning and scientific computing workloads.

## Features

- Optimized GPU-accelerated matrix operations
- Tensor core support for compatible NVIDIA GPUs
- Automatic memory management
- PyTorch-like API for easy integration
- Support for various data types (FP32, FP16, BF16)
- Dynamic kernel optimization for different GPU architectures

## Installation

### Prerequisites

- Python 3.7+
- CUDA Toolkit 11.0+ (for GPU support)
- A compatible NVIDIA GPU

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/celeris.git
cd celeris

# Install dependencies and the package
pip install -e .
```

## Quick Start

```python
import celeris
import numpy as np

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
result = tensor.numpy()
```

## Example: Linear Regression

```python
import celeris
import numpy as np

# Generate synthetic data
X = np.random.rand(100, 1).astype(np.float32) * 10
y = 2 * X + 1 + np.random.randn(100, 1).astype(np.float32)

# Convert to Celeris tensors
X_tensor = celeris.from_numpy(X)
y_tensor = celeris.from_numpy(y)

# Initialize model parameters
W = celeris.randn([1, 1])
b = celeris.zeros([1])
W.requires_grad = True
b.requires_grad = True

# Training parameters
learning_rate = 0.01
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    y_pred = celeris.matmul(X_tensor, W) + b
    
    # Compute loss
    loss = celeris.mse_loss(y_pred, y_tensor)
    
    # Backward pass and update parameters
    loss.backward()
    W = W - learning_rate * W.grad
    b = b - learning_rate * b.grad
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
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

## Available Models and Examples

The `examples/` directory contains various examples demonstrating Celeris capabilities:

- Linear regression
- Classification
- CNN models
- LSTM networks
- Transformer models

## License

[MIT License](LICENSE) 