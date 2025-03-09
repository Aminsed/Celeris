# Usage Guide for Celeris

Celeris is designed to help users who are already familiar with PyTorch quickly build and deploy models on GPU using a simplified interface. This guide explains how to use Celeris to create tensors, build neural network models using the celeris.nn module, and train them on your data.

## Table of Contents

1. [Basic Tensor Operations](#basic-tensor-operations)
2. [Building Neural Networks with celeris.nn](#building-neural-networks-with-celerisnn)
3. [Example: Building an MLP for Classification](#example-building-an-mlp-for-classification)
4. [Running Models on GPU and CPU](#running-models-on-gpu-and-cpu)
5. [Additional Tips and Troubleshooting](#additional-tips-and-troubleshooting)

## Basic Tensor Operations

Celeris provides simple functions for creating and manipulating tensors. Here are some of the key operations:

- **from_numpy(array)**: Converts a NumPy array to a Celeris tensor.
- **randn(*shape)**: Generates a random tensor with the specified shape.
- **add(x, y)**: Performs element-wise addition.
- **mul(x, y)**: Performs element-wise multiplication.
- **matmul(x, y)**: Performs matrix multiplication.
- **Activation functions**: Functions like `relu(x)`, `sigmoid(x)`, and `tanh(x)` apply common activation functions.

**Example:**

```python
import numpy as np
import celeris

# Create a tensor from a NumPy array
data = np.array([[1, 2], [3, 4]], dtype=np.float32)
tensor = celeris.from_numpy(data)

# Generate a random tensor
random_tensor = celeris.randn(3, 3)

# Perform element-wise addition
result = celeris.add(tensor, tensor)
print(result)
```

## Building Neural Networks with celeris.nn

The `celeris.nn` module mimics the PyTorch `nn` module to help you build neural networks easily.

**Example:** Building a simple feed-forward neural network (MLP):

```python
import torch
from celeris.nn import Linear, ReLU, Sequential

# Define a simple MLP model
model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
)

# Print the model architecture
print(model)
```

## Example: Building an MLP for Classification

Below is a complete example of building a simple MLP model for classifying images (e.g., from the MNIST dataset):

```python
import torch
from celeris.nn import Linear, ReLU, Sequential

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = Sequential(
            Linear(784, 128),
            ReLU(),
            Linear(128, 10)
        )
    
    def forward(self, x):
        # Flatten the input image tensor
        x = x.view(x.size(0), -1)
        return self.model(x)

# Initialize and print the model
model = MLP()
print(model)
```

This model can be trained using standard PyTorch training loops, and it works similarly to how you would build models in PyTorch.

## Running Models on GPU and CPU

Celeris automatically detects your GPU if CUDA is available. Tensors are placed on the GPU by default; otherwise, they run on the CPU. You can adjust the device settings via the configuration options in `celeris/config.py`.

## Additional Tips and Troubleshooting

- **Familiarity with PyTorch:** If you already know PyTorch, you'll find that switching to Celeris is straightforward because of its similar layer and model-building APIs.
- **Explore Examples:** Check out the example scripts in the `examples/` directory (e.g., classification, regression, CNN, LSTM, Transformer) to see how full models are built and trained.
- **Configuration and Debugging:** For troubleshooting and optimizing performance, refer to the [Installation Guide](docs/INSTALLATION.md) and the configuration options in `celeris/config.py`.
- **Need Help?** Contact the maintainers or check the issue tracker for support.

Happy modeling with Celeris! 