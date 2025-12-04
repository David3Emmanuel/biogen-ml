import torch
import torch.nn as nn
from typing import List

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models import FusedModel
from .gradcam import GradCamModelWrapper, ClassifierOutputTarget

def explain_with_image(model: FusedModel, image_tensor: torch.Tensor, target_output_index: int):
    """
    Explain the fused model's prediction using Grad-CAM for the image input.
    
    Args:
        model (FusedModel): The fused model to explain.
        image_tensor (torch.Tensor): The input image tensor.
        tabular_tensor (torch.Tensor): The input tabular tensor.
        target_output_index (int): The index of the output neuron to explain (0 or 1).
    
    Returns:
        grayscale_cam (numpy.ndarray): The computed heatmap as a grayscale array.
        visualization (numpy.ndarray): The heatmap visualization overlayed on the image.
    """
    
    assert target_output_index in [0, 1], "target_output_index must be 0 or 1"
    
    model.eval()
    dummy_tab = torch.randn(1, model.num_tabular_features)
    wrapped_model = GradCamModelWrapper(model, dummy_tab)
    target_layer = [wrapped_model.model.image_branch.model.layer4[-1]]
    
    cam = GradCAM(model=wrapped_model, target_layers=target_layer)
    targets: List[nn.Module] = [ClassifierOutputTarget(output_index=target_output_index)]
    
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Visualize the heatmap
    # 'rgb_img' should be a 0-1 normalized numpy-array version of the input image
    # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    rgb_img = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return grayscale_cam, visualization