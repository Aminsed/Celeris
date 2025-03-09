#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Example using Celeris (with celeris.nn mimicking torch.nn).
This example trains an LSTM model to predict a sine wave.
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from celeris.nn import LSTM, Linear, MSELoss

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate sine wave data
def generate_sine_wave(seq_length, num_samples, freq=0.1):
    """Generate sine wave data."""
    x = np.linspace(0, seq_length, num_samples)
    y = np.sin(2 * np.pi * freq * x)
    return x, y

# Parameters
seq_length = 100
input_size = 1
hidden_size = 32
num_layers = 1
output_size = 1
num_epochs = 100
learning_rate = 0.01

# Generate data
x, y = generate_sine_wave(10, seq_length)

# Create sequences for training
def create_sequences(data, seq_length):
    """Create sequences for training."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+1:i+seq_length+1]  # Next value prediction
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

x_seq, y_seq = create_sequences(y, 20)  # Use 20 time steps

# Convert to PyTorch tensors
x_tensor = torch.FloatTensor(x_seq).unsqueeze(-1).to(device)  # [batch, seq_len, input_size]
y_tensor = torch.FloatTensor(y_seq).unsqueeze(-1).to(device)  # [batch, seq_len, output_size]

# Define the LSTM model
class LSTMPredictor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMPredictor, self).__init__()
        self.lstm = LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = Linear(hidden_size, output_size)
        
    def forward(self, x):
        # LSTM output: output, (h_n, c_n)
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out)
        return predictions

# Initialize the model
model = LSTMPredictor(input_size, hidden_size, num_layers, output_size).to(device)

# Loss and optimizer
criterion = MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
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

# Generate predictions
model.eval()
with torch.no_grad():
    # Use the last sequence as input for prediction
    test_input = x_tensor[-1].unsqueeze(0)  # [1, seq_len, input_size]
    
    # Predict the next 50 values
    predictions = []
    current_input = test_input
    
    for _ in range(50):
        # Get prediction for the next time step
        pred = model(current_input)
        next_value = pred[:, -1, :]  # Get the last time step prediction
        predictions.append(next_value.item())
        
        # Update input sequence by removing the first value and adding the prediction
        new_input = torch.cat([current_input[:, 1:, :], next_value.unsqueeze(1)], dim=1)
        current_input = new_input

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(range(len(y)), y, 'b-', label='Original Sine Wave')
plt.plot(range(len(y), len(y) + len(predictions)), predictions, 'r-', label='Predicted Values')
plt.axvline(x=len(y)-1, color='g', linestyle='--', label='Prediction Start')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('LSTM Sine Wave Prediction')
plt.legend()
plt.savefig('lstm_prediction.png')
plt.close()

print("LSTM example finished. Results saved to lstm_prediction.png") 