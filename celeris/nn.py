import torch.nn as nn
import torch.nn.functional as F

# Import all classes from torch.nn
Linear = nn.Linear
Conv2d = nn.Conv2d
MaxPool2d = nn.MaxPool2d
ReLU = nn.ReLU
Sigmoid = nn.Sigmoid
Tanh = nn.Tanh
LSTM = nn.LSTM
GRU = nn.GRU
Transformer = nn.Transformer
TransformerEncoder = nn.TransformerEncoder
TransformerEncoderLayer = nn.TransformerEncoderLayer
Embedding = nn.Embedding
LayerNorm = nn.LayerNorm
BatchNorm2d = nn.BatchNorm2d
Dropout = nn.Dropout
Sequential = nn.Sequential
Module = nn.Module
MSELoss = nn.MSELoss
CrossEntropyLoss = nn.CrossEntropyLoss

# Expose functional interface
functional = F

# Expose all attributes from nn
__all__ = [
    'Linear', 'Conv2d', 'MaxPool2d', 'ReLU', 'Sigmoid', 'Tanh',
    'LSTM', 'GRU', 'Transformer', 'TransformerEncoder', 'TransformerEncoderLayer',
    'Embedding', 'LayerNorm', 'BatchNorm2d', 'Dropout', 'Sequential', 'Module',
    'MSELoss', 'CrossEntropyLoss', 'functional'
] 