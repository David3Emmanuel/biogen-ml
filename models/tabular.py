import torch.nn as nn

class TabularMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for processing
    structured tabular data. Outputs predictions directly for late fusion.
    """
    def __init__(self, input_dim, output_dim=2, dropout=0.3):
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
            
            # Output layer (predictions)
            # output_dim should be 2 for 1-year and 3-year risk predictions
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass for the tabular data.
        Input x shape: (batch_size, input_dim)
        Output shape: (batch_size, output_dim)
        """
        return self.layers(x)