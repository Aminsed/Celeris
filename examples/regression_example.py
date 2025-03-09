#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression Example using Celeris (with celeris.nn mimicking torch.nn).
This example trains a simple linear regression model on synthetic data.
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from celeris.nn import Linear, MSELoss

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate synthetic data
np.random.seed(42)
x = np.random.rand(100, 1) * 10
y = 2 * x + 1 + np.random.randn(100, 1) * 1.5  # y = 2x + 1 + noise

# Convert to PyTorch tensors
x_tensor = torch.FloatTensor(x).to(device)
y_tensor = torch.FloatTensor(y).to(device)

# Define the linear regression model
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = Linear(1, 1)  # Using celeris.nn.Linear
        
    def forward(self, x):
        return self.linear(x)

# Initialize the model
model = LinearRegression().to(device)

# Loss and optimizer
criterion = MSELoss()  # Using celeris.nn.MSELoss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Get the parameters
with torch.no_grad():
    slope = model.linear.weight.item()
    intercept = model.linear.bias.item()
    print(f'Learned parameters: y = {slope:.4f}x + {intercept:.4f}')
    print(f'True parameters: y = 2x + 1')

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data points')
x_range = np.array([0, 10]).reshape(-1, 1)
y_pred = slope * x_range + intercept
plt.plot(x_range, y_pred, color='red', linewidth=2, label=f'Fitted line: y = {slope:.4f}x + {intercept:.4f}')
plt.plot(x_range, 2 * x_range + 1, color='green', linestyle='--', linewidth=2, label='True line: y = 2x + 1')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.legend()
plt.savefig('regression_result.png')
plt.close()

print("Regression example finished. Results saved to regression_result.png") 