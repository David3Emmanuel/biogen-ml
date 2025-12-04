import torch.nn as nn
import torchvision.models as models

class ImageCNN(nn.Module):
    """
    A pre-trained ResNet-18 based feature extractor for image data.
    The final classification head is replaced to output feature vectors.
    """
    def __init__(self, output_dim=128):
        super().__init__()
        
        # Load pre-trained ResNet-18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Did not freeze parameters
        # gradients are needed for Grad-CAM
            
        # Replace the final layer with a new linear layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, output_dim)
    
    def forward(self, x):
        """
        Forward pass for the image data.
        Input x shape: (batch_size, 3, height, width)
        Output shape: (batch_size, output_dim)
        """
        return self.model(x)