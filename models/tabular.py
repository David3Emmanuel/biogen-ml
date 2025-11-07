import torch
import torch.nn as nn

class TabularMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for processing
    structured tabular data.
    """
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            # First hidden layer
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second hidden layer
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer (feature vector)
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass for the tabular data.
        Input x shape: (batch_size, input_dim)
        Output shape: (batch_size, output_dim)
        """
        return self.layers(x)