import torch
import torch.nn as nn

from .tabular import TabularMLP
from .image import ImageCNN


class FusedModel(nn.Module):
    def __init__(self, num_tabular_features, fusion_weights=None):
        super().__init__()

        self.num_tabular_features = num_tabular_features
        
        # 1. Image Branch (pre-trained ResNet-18)
        # Outputs 2 predictions directly
        self.image_branch = ImageCNN(output_dim=2)
        self.image_branch.eval() 
        
        # 2. Tabular Branch (custom MLP)
        # Outputs 2 predictions directly
        self.tabular_branch = TabularMLP(input_dim=num_tabular_features, output_dim=2)
        
        # 3. Late Fusion: learnable weights for combining predictions
        # If fusion_weights not provided, initialize with equal weights [0.5, 0.5]
        if fusion_weights is None:
            self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        else:
            self.fusion_weights = nn.Parameter(torch.tensor(fusion_weights))
            
    def forward(self, image_tensor, tabular_tensor):
        # Did not use with torch.no_grad
        # gradients are needed for Grad-CAM
        
        # Get predictions from each branch
        # Shape: (Batch_Size, 2)
        image_predictions = self.image_branch(image_tensor)        
        tabular_predictions = self.tabular_branch(tabular_tensor)
        
        # Late fusion: weighted combination of predictions
        # Apply softmax to fusion weights to ensure they sum to 1
        weights = torch.softmax(self.fusion_weights, dim=0)
        
        # Shape: (Batch_Size, 2)
        # Output 0: 1-year risk (logit)
        # Output 1: 3-year risk (logit)
        output = weights[0] * image_predictions + weights[1] * tabular_predictions
        return output
