import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union

from models import FusedModel


def explain_with_shap(
    model: FusedModel,
    bg_tabular: torch.Tensor,
    image_tensor: torch.Tensor,
    test_tabular: torch.Tensor,
    target_output_index: int = 0,
    tabular_feature_names: Optional[List[str]] = None,
    max_bg_samples: int = 10,
    nsamples: int = 50
) -> Tuple[np.ndarray, shap.Explainer]:
    """
    
    Args:
        model (FusedModel): The fused model to explain.
        bg_tabular (torch.Tensor): Background tabular data (can provide more, will be subsampled).
        image_tensor (torch.Tensor): Test image data.
        test_tabular (torch.Tensor): Test tabular data to explain.
        target_output_index (int): Output neuron index (0 for 1-year, 1 for 3-year).
        tabular_feature_names (List[str], optional): Feature names for plotting.
        max_bg_samples (int): Maximum background samples (default: 10).
        nsamples (int): Number of SHAP evaluations per sample (default: 50, higher = more accurate but slower).
    
    Returns:
        tabular_shap_values (np.ndarray): SHAP values for tabular features.
        explainer (shap.Explainer): The SHAP explainer object.
    """
    
    assert target_output_index in [0, 1], "target_output_index must be 0 or 1"
    
    model.eval()
    device = next(model.parameters()).device
    
    # Use minimal background samples
    bg_tabular = bg_tabular[:max_bg_samples].to(device)
    
    # Ensure image_tensor has batch dimension
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    
    test_tabular = test_tabular.to(device)
    
    # Ensure test_tabular has batch dimension
    if test_tabular.ndim == 1:
        test_tabular = test_tabular.unsqueeze(0)  # Add batch dimension
    
    # Create wrapper that fixes image input
    class TabularOnlyWrapper:
        def __init__(self, model, images, target_idx):
            self.model = model
            self.images = images
            self.target_idx = target_idx
            
        def __call__(self, tabular_data):
            if isinstance(tabular_data, np.ndarray):
                tabular_tensor = torch.tensor(tabular_data, dtype=torch.float32, device=self.images.device)
            else:
                tabular_tensor = tabular_data
            
            batch_size = tabular_tensor.shape[0]
            images_batch = self.images[0:1].expand(batch_size, -1, -1, -1)
            
            with torch.no_grad():
                outputs = self.model(images_batch, tabular_tensor)
                return outputs[:, self.target_idx].detach().cpu().numpy()
    
    wrapper = TabularOnlyWrapper(model, image_tensor, target_output_index)
    
    # Convert to numpy
    bg_tabular_np = bg_tabular.detach().cpu().numpy()
    test_tabular_np = test_tabular.detach().cpu().numpy()
    
    # Use KernelExplainer with minimal sampling
    print(f"Initializing SHAP with {max_bg_samples} background samples...")
    explainer = shap.KernelExplainer(wrapper, bg_tabular_np)
    
    print(f"Computing SHAP values with {nsamples} samples per explanation...")
    tabular_shap_values = explainer.shap_values(test_tabular_np, nsamples=nsamples, silent=False)
    
    if isinstance(tabular_shap_values, torch.Tensor):
        tabular_shap_values = tabular_shap_values.detach().cpu().numpy()
    
    return tabular_shap_values, explainer


def plot_shap_force(
    explainer: shap.Explainer,
    tabular_shap_values: np.ndarray,
    test_tabular: torch.Tensor,
    target_output_index: int = 0,
    tabular_feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Generate a SHAP force plot for a single sample explaining an individual prediction.
    
    Args:
        explainer (shap.Explainer): The SHAP explainer object.
        tabular_shap_values (np.ndarray): SHAP values for tabular features.
                                         Shape: (num_samples, num_features)
        test_tabular (torch.Tensor): Test tabular data.
                                     Shape: (num_samples, num_features)
        target_output_index (int): The index of the output neuron (0 for 1-year, 1 for 3-year).
        tabular_feature_names (List[str], optional): Names of the tabular features.
        save_path (str, optional): Path to save the plot. If None, uses default name.
        show (bool): Whether to display the plot.
    """
    
    # Convert test_tabular to numpy if needed
    if isinstance(test_tabular, torch.Tensor):
        test_tabular_np = test_tabular.detach().cpu().numpy()
    else:
        test_tabular_np = test_tabular
    
    # Ensure test_tabular_np has batch dimension
    if test_tabular_np.ndim == 1:
        test_tabular_np = test_tabular_np.reshape(1, -1)
    
    # Generate default feature names if not provided
    if tabular_feature_names is None:
        num_features = tabular_shap_values.shape[1]
        tabular_feature_names = [f"Feature_{i}" for i in range(num_features)]
    
    # Get the expected value (base value) for the specified output
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        expected_value = explainer.expected_value[target_output_index]
    else:
        expected_value = explainer.expected_value
    
    # Create force plot
    shap.force_plot(
        expected_value,
        tabular_shap_values[0, :],
        test_tabular_np[0, :],
        feature_names=tabular_feature_names,
        show=show,
        matplotlib=True
    )
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved SHAP force plot to {save_path}")
    else:
        default_path = f'shap_force_plot.png'
        plt.savefig(default_path, bbox_inches='tight', dpi=150)
        print(f"Saved SHAP force plot to {default_path}")
    
    if not show:
        plt.close()


def plot_shap_waterfall(
    explainer: shap.Explainer,
    tabular_shap_values: np.ndarray,
    test_tabular: torch.Tensor,
    target_output_index: int = 0,
    tabular_feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Generate a SHAP waterfall plot for a single sample showing how each feature contributes.
    
    Args:
        explainer (shap.Explainer): The SHAP explainer object.
        tabular_shap_values (np.ndarray): SHAP values for tabular features.
                                         Shape: (num_samples, num_features)
        test_tabular (torch.Tensor): Test tabular data.
                                     Shape: (num_samples, num_features)
        target_output_index (int): The index of the output neuron (0 for 1-year, 1 for 3-year).
        tabular_feature_names (List[str], optional): Names of the tabular features.
        save_path (str, optional): Path to save the plot. If None, uses default name.
        show (bool): Whether to display the plot.
    """
    
    # Convert test_tabular to numpy if needed
    if isinstance(test_tabular, torch.Tensor):
        test_tabular_np = test_tabular.detach().cpu().numpy()
    else:
        test_tabular_np = test_tabular
    
    # Ensure test_tabular_np has batch dimension
    if test_tabular_np.ndim == 1:
        test_tabular_np = test_tabular_np.reshape(1, -1)
    
    # Generate default feature names if not provided
    if tabular_feature_names is None:
        num_features = tabular_shap_values.shape[1]
        tabular_feature_names = [f"Feature_{i}" for i in range(num_features)]
    
    # Get the expected value (base value) for the specified output
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        expected_value = explainer.expected_value[target_output_index]
    else:
        expected_value = explainer.expected_value
    
    # Create an Explanation object for waterfall plot
    explanation = shap.Explanation(
        values=tabular_shap_values[0, :],
        base_values=expected_value,
        data=test_tabular_np[0, :],
        feature_names=tabular_feature_names
    )
    
    # Create waterfall plot
    plt.figure()
    shap.waterfall_plot(explanation, show=False)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved SHAP waterfall plot to {save_path}")
    else:
        default_path = f'shap_waterfall_plot.png'
        plt.savefig(default_path, bbox_inches='tight', dpi=150)
        print(f"Saved SHAP waterfall plot to {default_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
