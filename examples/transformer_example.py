#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Example using Celeris (with celeris.nn mimicking torch.nn).
This example trains a transformer encoder for a sequence classification task.
"""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from celeris.nn import (
    TransformerEncoder, TransformerEncoderLayer, Linear, 
    LayerNorm, CrossEntropyLoss, Embedding
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
vocab_size = 1000
embedding_dim = 64
num_heads = 4
hidden_dim = 128
num_layers = 2
num_classes = 5
max_seq_length = 20
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Generate synthetic data
def generate_synthetic_data(num_samples, max_seq_length, vocab_size, num_classes):
    """Generate synthetic data for sequence classification."""
    # Generate random sequences
    sequences = np.random.randint(1, vocab_size, size=(num_samples, max_seq_length))
    
    # Generate random labels
    labels = np.random.randint(0, num_classes, size=num_samples)
    
    # Create attention masks (1 for tokens, 0 for padding)
    # For simplicity, we'll use random sequence lengths
    seq_lengths = np.random.randint(5, max_seq_length + 1, size=num_samples)
    attention_masks = np.zeros((num_samples, max_seq_length))
    
    for i, length in enumerate(seq_lengths):
        attention_masks[i, :length] = 1
    
    return sequences, labels, attention_masks

# Generate data
train_sequences, train_labels, train_masks = generate_synthetic_data(
    500, max_seq_length, vocab_size, num_classes
)

# Convert to PyTorch tensors
train_sequences = torch.LongTensor(train_sequences).to(device)
train_labels = torch.LongTensor(train_labels).to(device)
train_masks = torch.FloatTensor(train_masks).to(device)

# Create data loader
train_data = torch.utils.data.TensorDataset(train_sequences, train_masks, train_labels)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Define the transformer model
class TransformerClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, num_classes, max_seq_length):
        super(TransformerClassifier, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_seq_length)
        encoder_layers = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, batch_first=False)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.classifier = Linear(embedding_dim, num_classes)
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len]
        # mask: [batch_size, seq_len]
        
        # Convert mask to transformer format (1 = ignore, 0 = attend)
        if mask is not None:
            # In PyTorch transformer, mask is True for positions to mask (ignore)
            # and False for positions to attend to, which is the opposite of our mask
            src_key_padding_mask = (mask == 0)  # [batch_size, seq_len]
        else:
            src_key_padding_mask = None
        
        # Embedding and positional encoding
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = self.pos_encoder(x)
        
        # Transformer encoder
        # PyTorch transformer expects input of shape [seq_len, batch_size, embedding_dim]
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Use the output of the [CLS] token (first token) for classification
        x = x[0, :, :]  # [batch_size, embedding_dim]
        
        # Classification layer
        x = self.classifier(x)  # [batch_size, num_classes]
        
        return x

# Positional encoding
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

# Initialize the model
model = TransformerClassifier(
    vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, num_classes, max_seq_length
).to(device)

# Loss and optimizer
criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for batch_idx, (sequences, masks, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(sequences, masks)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.savefig('transformer_training_loss.png')
plt.close()

# Evaluate on a sample
model.eval()
with torch.no_grad():
    sample_sequences = train_sequences[:5]
    sample_masks = train_masks[:5]
    sample_labels = train_labels[:5]
    
    outputs = model(sample_sequences, sample_masks)
    _, predicted = torch.max(outputs, 1)
    
    print("\nSample Evaluation:")
    print(f"Predicted: {predicted.cpu().numpy()}")
    print(f"Actual: {sample_labels.cpu().numpy()}")
    print(f"Accuracy: {(predicted == sample_labels).sum().item() / len(sample_labels):.2f}")

print("\nTransformer example finished. Training loss plot saved to transformer_training_loss.png") 