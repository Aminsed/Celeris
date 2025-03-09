#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MNIST Classifier Example using Celeris

This example demonstrates how to use the Celeris library to implement a simple
neural network for MNIST digit classification.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import celeris

# Load MNIST dataset
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
X = X.astype(np.float32)
y = y.astype(np.int32)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to one-hot encoding
def one_hot_encode(y, num_classes=10):
    y_one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    y_one_hot[np.arange(y.shape[0]), y.astype(int)] = 1
    return y_one_hot

y_train_one_hot = one_hot_encode(y_train)
y_test_one_hot = one_hot_encode(y_test)

# Convert to Celeris tensors
X_train_tensor = celeris.from_numpy(X_train.reshape(-1, 784))
y_train_tensor = celeris.from_numpy(y_train_one_hot)
X_test_tensor = celeris.from_numpy(X_test.reshape(-1, 784))
y_test_tensor = celeris.from_numpy(y_test_one_hot)

# Define neural network parameters
input_size = 784
hidden_size = 128
output_size = 10

# Initialize weights and biases
W1 = celeris.randn([input_size, hidden_size]) * 0.01
b1 = celeris.zeros([hidden_size])
W2 = celeris.randn([hidden_size, output_size]) * 0.01
b2 = celeris.zeros([output_size])

# Set requires_grad for parameters
W1.requires_grad = True
b1.requires_grad = True
W2.requires_grad = True
b2.requires_grad = True

# Training parameters
learning_rate = 0.01
num_epochs = 10
batch_size = 100
num_batches = len(X_train) // batch_size

# Training loop
for epoch in range(num_epochs):
    # Shuffle the data
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    y_train_one_hot_shuffled = y_train_one_hot[indices]
    
    total_loss = 0
    
    for batch in range(num_batches):
        # Get batch
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size
        
        X_batch = X_train_shuffled[start_idx:end_idx]
        y_batch = y_train_one_hot_shuffled[start_idx:end_idx]
        
        # Convert to Celeris tensors
        X_batch_tensor = celeris.from_numpy(X_batch.reshape(-1, 784))
        y_batch_tensor = celeris.from_numpy(y_batch)
        
        # Forward pass
        # First layer
        z1 = celeris.matmul(X_batch_tensor, W1) + b1
        a1 = celeris.relu(z1)
        
        # Output layer
        z2 = celeris.matmul(a1, W2) + b2
        y_pred = celeris.softmax(z2, dim=1)
        
        # Compute loss
        loss = celeris.cross_entropy_loss(y_pred, y_batch_tensor)
        total_loss += loss.numpy()[0]
        
        # Backward pass (simplified since backward is not fully implemented)
        # In a real implementation, we would use .backward() and access .grad
        # For now, we'll compute gradients manually
        
        # Output layer gradients
        dz2 = y_pred - y_batch_tensor
        dW2 = celeris.matmul(a1.transpose([1, 0]), dz2) / batch_size
        db2 = celeris.mean(dz2, dim=0)
        
        # Hidden layer gradients
        da1 = celeris.matmul(dz2, W2.transpose([1, 0]))
        dz1 = da1 * (z1 > 0)  # ReLU gradient
        dW1 = celeris.matmul(X_batch_tensor.transpose([1, 0]), dz1) / batch_size
        db1 = celeris.mean(dz1, dim=0)
        
        # Update parameters
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        
        # Reset requires_grad
        W1.requires_grad = True
        b1.requires_grad = True
        W2.requires_grad = True
        b2.requires_grad = True
    
    # Print epoch statistics
    avg_loss = total_loss / num_batches
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # Evaluate on test set every few epochs
    if (epoch + 1) % 2 == 0:
        # Forward pass on test set
        z1_test = celeris.matmul(X_test_tensor, W1) + b1
        a1_test = celeris.relu(z1_test)
        z2_test = celeris.matmul(a1_test, W2) + b2
        y_pred_test = celeris.softmax(z2_test, dim=1)
        
        # Convert predictions to numpy
        y_pred_np = y_pred_test.numpy()
        y_pred_classes = np.argmax(y_pred_np, axis=1)
        y_test_classes = np.argmax(y_test_one_hot, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred_classes == y_test_classes)
        print(f"Test Accuracy: {accuracy:.4f}")

# Final evaluation
z1_test = celeris.matmul(X_test_tensor, W1) + b1
a1_test = celeris.relu(z1_test)
z2_test = celeris.matmul(a1_test, W2) + b2
y_pred_test = celeris.softmax(z2_test, dim=1)

# Convert predictions to numpy
y_pred_np = y_pred_test.numpy()
y_pred_classes = np.argmax(y_pred_np, axis=1)
y_test_classes = np.argmax(y_test_one_hot, axis=1)

# Calculate accuracy
accuracy = np.mean(y_pred_classes == y_test_classes)
print(f"Final Test Accuracy: {accuracy:.4f}")

# Visualize some predictions
plt.figure(figsize=(12, 8))
for i in range(10):
    plt.subplot(2, 5, i+1)
    idx = np.random.randint(0, len(X_test))
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {y_pred_classes[idx]}, True: {y_test_classes[idx]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('mnist_predictions.png')
plt.show() 