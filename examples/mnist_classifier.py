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
import torch
import torch.nn.functional as F

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

# Initialize weights and biases with small random values
W1 = celeris.randn([input_size, hidden_size]) * 0.01
b1 = celeris.zeros([hidden_size])
W2 = celeris.randn([hidden_size, output_size]) * 0.01
b2 = celeris.zeros([output_size])

# Training parameters
learning_rate = 0.01
num_epochs = 10
batch_size = 100
num_batches = len(X_train) // batch_size

# Training loop
loss_history = []
for epoch in range(num_epochs):
    epoch_loss = 0.0
    
    # Process mini-batches
    for i in range(num_batches):
        # Get mini-batch
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        X_batch = X_train_tensor[start_idx:end_idx]
        y_batch = y_train_tensor[start_idx:end_idx]
        
        # Forward pass
        # First layer
        z1 = celeris.matmul(X_batch, W1) + b1
        a1 = celeris.relu(z1)
        
        # Output layer
        z2 = celeris.matmul(a1, W2) + b2
        
        # Use PyTorch's F.log_softmax and F.nll_loss for numerical stability
        log_probs = F.log_softmax(z2, dim=1)
        loss = F.nll_loss(log_probs, torch.argmax(y_batch, dim=1))
        epoch_loss += loss.item()
        
        # Compute gradients
        # For softmax + cross entropy, the gradient is (probs - targets)
        probs = F.softmax(z2, dim=1)
        dscores = probs.clone()
        dscores[torch.arange(batch_size), torch.argmax(y_batch, dim=1)] -= 1
        dscores = dscores / batch_size
        
        # Gradient of output layer
        dW2 = celeris.matmul(a1.t(), dscores)
        db2 = torch.sum(dscores, dim=0)
        
        # Gradient of hidden layer
        dhidden = celeris.matmul(dscores, W2.t())
        dhidden[z1 <= 0] = 0  # ReLU gradient
        dW1 = celeris.matmul(X_batch.t(), dhidden)
        db1 = torch.sum(dhidden, dim=0)
        
        # Update parameters
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
    
    # Print epoch statistics
    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Evaluate on test set
# Forward pass
z1 = celeris.matmul(X_test_tensor, W1) + b1
a1 = celeris.relu(z1)
z2 = celeris.matmul(a1, W2) + b2
probs = F.softmax(z2, dim=1)

# Get predictions
predicted_classes = torch.argmax(probs, dim=1).cpu().numpy()
actual_classes = np.argmax(y_test_one_hot, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_classes == actual_classes)
print(f"Test Accuracy: {accuracy:.4f}")

# Plot loss history
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('mnist_training_loss.png')

# Plot some predictions
plt.figure(figsize=(12, 8))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {predicted_classes[i]}, True: {actual_classes[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('mnist_predictions.png')
print("MNIST classifier example finished. Results saved to mnist_predictions.png") 