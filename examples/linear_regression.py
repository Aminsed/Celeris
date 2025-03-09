#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Linear Regression Example using Celeris

This example demonstrates how to use the Celeris library to implement a simple
linear regression model.
"""

import numpy as np
import matplotlib.pyplot as plt
import celeris

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

# Set requires_grad for parameters
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
    
    # Backward pass
    loss.backward()
    
    # Update parameters (simplified since backward is not fully implemented)
    # In a real implementation, we would use W.grad and b.grad
    # For now, we'll compute gradients manually
    dW = 2 * celeris.matmul(X_tensor.transpose([1, 0]), (y_pred - y_tensor)) / X.shape[0]
    db = 2 * celeris.mean(y_pred - y_tensor)
    
    # Update parameters
    W = W - learning_rate * dW
    b = b - learning_rate * db
    
    # Reset gradients
    W.requires_grad = True
    b.requires_grad = True
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.numpy()[0]:.4f}")

# Convert final parameters to numpy
W_numpy = W.numpy()
b_numpy = b.numpy()

print(f"Final parameters: W = {W_numpy[0][0]:.4f}, b = {b_numpy[0]:.4f}")
print(f"True parameters: W = 2.0000, b = 1.0000")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, X * W_numpy[0][0] + b_numpy[0], color='red', label='Fitted line')
plt.plot(X, X * 2 + 1, color='green', linestyle='--', label='True line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Celeris')
plt.legend()
plt.savefig('linear_regression.png')
plt.show() 