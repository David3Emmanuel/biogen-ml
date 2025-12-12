import torch.nn as nn
import torchvision.models as models

class ImageCNN(nn.Module):
    """
    A pre-trained ResNet-18 based model for image data.
    Outputs predictions directly for late fusion.
    """
    def __init__(self, output_dim=2):
        super().__init__()
        
        # Load pre-trained ResNet-18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Did not freeze parameters
        # gradients are needed for Grad-CAM
            
        # Replace the final layer to output predictions
        # output_dim should be 2 for 1-year and 3-year risk predictions
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass for the image data.
        Input x shape: (batch_size, 3, height, width)
        Output shape: (batch_size, output_dim)
        """
        return self.model(x)