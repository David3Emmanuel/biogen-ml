import torch
import torch.nn as nn

from models.tabular import TabularMLP
from models.image import ImageCNN


class FusedModel(nn.Module):
    def __init__(self, num_tabular_features, img_feature_dim=128, tab_feature_dim=64):
        super().__init__()
        
        # 1. Image Branch (pre-trained ResNet-18)
        self.image_branch = ImageCNN(output_dim=img_feature_dim)
        self.image_branch.eval() 
        
        # 2. Tabular Branch (custom MLP)
        self.tabular_branch = TabularMLP(input_dim=num_tabular_features, output_dim=tab_feature_dim)
        
        # 3. Fusion Head (MLP)
        fused_dim = img_feature_dim + tab_feature_dim
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            # Final output layer with 2 neurons
            # Output 0: 1-year risk (logit)
            # Output 1: 3-year risk (logit)
            nn.Linear(128, 2) 
        )
            
    def forward(self, image_tensor, tabular_tensor):
        with torch.no_grad():
            image_features = self.image_branch(image_tensor)        
        tabular_features = self.tabular_branch(tabular_tensor)
        
        # Shape: (Batch_Size, img_feature_dim + tab_feature_dim)
        fused_features = torch.cat((image_features, tabular_features), dim=1) 
        
        # Shape: (Batch_Size, 2)
        output = self.fusion_head(fused_features)
        return output
