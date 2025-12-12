import torch
import torch.nn as nn
from typing import List

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models import FusedModel


class GradCamModelWrapper(nn.Module):
    """
    A wrapper for the fused model to make it compatible
    with the single-input grad-cam library.
    """
    def __init__(self, fused_model, dummy_tabular_tensor):
        super().__init__()
        self.model = fused_model
        self.dummy_tabular = dummy_tabular_tensor

    def forward(self, image_tensor):
        return self.model(image_tensor, self.dummy_tabular)


def explain_with_image(model: FusedModel, image_tensor: torch.Tensor, target_output_index: int):
    """
    Explain the fused model's prediction using Grad-CAM for the image input.
    
    Args:
        model (FusedModel): The fused model to explain.
        image_tensor (torch.Tensor): The input image tensor.
        target_output_index (int): The index of the output neuron to explain (0 or 1).
    
    Returns:
        grayscale_cam (numpy.ndarray): The computed heatmap as a grayscale array.
        visualization (numpy.ndarray): The heatmap visualization overlayed on the image.
    """
    
    assert target_output_index in [0, 1], "target_output_index must be 0 or 1"
    
    # 1. Prepare Model
    model.eval()
    
    # 2. CRITICAL FIX: Enable gradients on the input tensor
    # Even in eval mode, we need the input to require grad to build the graph
    # because the model weights are likely frozen (YOLO behavior).
    if not image_tensor.requires_grad:
        image_tensor = image_tensor.detach().requires_grad_(True)
        
    # 3. Setup Wrapper
    dummy_tab = torch.randn(1, model.num_tabular_features).to(image_tensor.device)
    wrapped_model = GradCamModelWrapper(model, dummy_tab)
    
    # 4. Define Target Layer
    # Targeting the last convolutional block of the YOLO backbone (usually layer -2)
    # The path is: FusedModel -> SpecializedImageWrapper -> CancerImageWrapper -> 
    #              YOLO -> ClassificationModel -> Sequential -> Layer -2
    target_layer = [wrapped_model.model.image_branch.wrapper.backbone.model.model[-2]]
    
    # 5. Initialize GradCAM
    cam = GradCAM(model=wrapped_model, target_layers=target_layer)
    targets = [ClassifierOutputTarget(category=target_output_index)]
    
    # 6. Generate CAM
    # This runs forward/backward passes. 
    # Since image_tensor.requires_grad=True, the graph will be built.
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # 7. Visualize
    rgb_img = image_tensor.detach().squeeze().permute(1, 2, 0).cpu().numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return grayscale_cam, visualization