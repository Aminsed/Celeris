#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic tests for the Celeris library.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import celeris
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import celeris
    print("Successfully imported celeris")
except ImportError as e:
    print(f"Failed to import celeris: {e}")
    print("Make sure the library is built and installed correctly.")
    sys.exit(1)

def test_tensor_creation():
    """Test tensor creation from numpy arrays."""
    print("\n=== Testing tensor creation ===")
    
    # Create a numpy array
    np_array = np.random.randn(2, 3).astype(np.float32)
    print(f"NumPy array shape: {np_array.shape}, dtype: {np_array.dtype}")
    
    # Create a tensor from the numpy array
    try:
        tensor = celeris.from_numpy(np_array)
        print(f"Created tensor: {tensor}")
        print(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")
        return tensor
    except Exception as e:
        print(f"Failed to create tensor: {e}")
        return None

def test_tensor_operations(tensor):
    """Test basic tensor operations."""
    if tensor is None:
        print("Skipping tensor operations test as tensor creation failed.")
        return
    
    print("\n=== Testing tensor operations ===")
    
    try:
        # Test addition
        result = tensor + tensor
        print(f"Addition result: {result}")
        
        # Test multiplication
        result = tensor * tensor
        print(f"Multiplication result: {result}")
        
        # Test matmul
        try:
            result = celeris.matmul(tensor, tensor.transpose([1, 0]))
            print(f"Matrix multiplication result: {result}")
        except Exception as e:
            print(f"Matrix multiplication failed: {e}")
        
        # Test activation functions
        try:
            result = celeris.relu(tensor)
            print(f"ReLU result: {result}")
        except Exception as e:
            print(f"ReLU failed: {e}")
            
        try:
            result = celeris.sigmoid(tensor)
            print(f"Sigmoid result: {result}")
        except Exception as e:
            print(f"Sigmoid failed: {e}")
    except Exception as e:
        print(f"Tensor operations failed: {e}")

def test_device_conversion(tensor):
    """Test device conversion."""
    if tensor is None:
        print("Skipping device conversion test as tensor creation failed.")
        return
    
    print("\n=== Testing device conversion ===")
    
    try:
        # Convert to CPU
        cpu_tensor = tensor.to(celeris.DeviceType.CPU)
        print(f"Converted to CPU: {cpu_tensor}")
        
        # Convert back to CUDA
        cuda_tensor = cpu_tensor.to(celeris.DeviceType.CUDA)
        print(f"Converted back to CUDA: {cuda_tensor}")
    except Exception as e:
        print(f"Device conversion failed: {e}")

def test_numpy_conversion(tensor):
    """Test conversion to numpy."""
    if tensor is None:
        print("Skipping numpy conversion test as tensor creation failed.")
        return
    
    print("\n=== Testing numpy conversion ===")
    
    try:
        # Convert to numpy
        np_array = tensor.numpy()
        print(f"Converted to NumPy: shape={np_array.shape}, dtype={np_array.dtype}")
        print(np_array)
    except Exception as e:
        print(f"NumPy conversion failed: {e}")

def test_gradient_tracking():
    """Test gradient tracking."""
    print("\n=== Testing gradient tracking ===")
    
    try:
        # Create tensors
        x = celeris.randn([2, 3])
        w = celeris.randn([3, 1])
        
        # Set requires_grad
        x.requires_grad = True
        w.requires_grad = True
        
        # Forward pass
        y = celeris.matmul(x, w)
        
        # Check if gradients are being tracked
        print(f"x requires_grad: {x.requires_grad}")
        print(f"w requires_grad: {w.requires_grad}")
        print(f"y requires_grad: {y.requires_grad}")
        
        # Backward pass
        try:
            y.backward()
            print("Backward pass completed")
            
            # Check gradients
            print(f"x.grad: {x.grad}")
            print(f"w.grad: {w.grad}")
        except Exception as e:
            print(f"Backward pass failed: {e}")
    except Exception as e:
        print(f"Gradient tracking test failed: {e}")

def test_neural_network():
    """Test a simple neural network."""
    print("\n=== Testing simple neural network ===")
    
    try:
        # Create random input and target
        x = celeris.from_numpy(np.random.randn(10, 5).astype(np.float32))
        target = celeris.from_numpy(np.random.randn(10, 2).astype(np.float32))
        
        # Create weights and biases
        w1 = celeris.randn([5, 8])
        b1 = celeris.zeros([8])
        w2 = celeris.randn([8, 2])
        b2 = celeris.zeros([2])
        
        # Set requires_grad
        w1.requires_grad = True
        b1.requires_grad = True
        w2.requires_grad = True
        b2.requires_grad = True
        
        # Forward pass
        z1 = celeris.matmul(x, w1) + b1
        a1 = celeris.relu(z1)
        z2 = celeris.matmul(a1, w2) + b2
        
        # Compute loss
        loss = celeris.mse_loss(z2, target)
        print(f"Loss: {loss}")
        
        # Backward pass
        try:
            loss.backward()
            print("Backward pass completed")
        except Exception as e:
            print(f"Backward pass failed: {e}")
    except Exception as e:
        print(f"Neural network test failed: {e}")

def main():
    """Run all tests."""
    print("Starting Celeris tests...")
    
    # Test tensor creation
    tensor = test_tensor_creation()
    
    # Test tensor operations
    test_tensor_operations(tensor)
    
    # Test device conversion
    test_device_conversion(tensor)
    
    # Test numpy conversion
    test_numpy_conversion(tensor)
    
    # Test gradient tracking
    test_gradient_tracking()
    
    # Test neural network
    test_neural_network()
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    main() 