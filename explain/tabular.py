import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union

from models import FusedModel


def explain_with_shap(
    model: FusedModel,
    bg_tabular: torch.Tensor,
    test_images: torch.Tensor,
    test_tabular: torch.Tensor,
    target_output_index: int = 0,
    tabular_feature_names: Optional[List[str]] = None,
    max_bg_samples: int = 10,
    nsamples: int = 50
) -> Tuple[np.ndarray, shap.Explainer]:
    """
    FASTEST OPTION: Lightweight SHAP explanation for tabular features only.
    
    This is the recommended function for quick explanations. It:
    - Uses a minimal background dataset (10 samples by default)
    - Uses KernelExplainer with reduced sampling (50 evaluations)
    - Fixes image input to the mean test image
    - Typically 5-10x faster than the standard explain_with_shap
    
    Args:
        model (FusedModel): The fused model to explain.
        bg_tabular (torch.Tensor): Background tabular data (can provide more, will be subsampled).
        test_images (torch.Tensor): Test image data (uses mean for speed).
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
    test_images = test_images.to(device)
    test_tabular = test_tabular.to(device)
    
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
    
    # Use mean test image
    mean_test_image = test_images.mean(dim=0, keepdim=True)
    wrapper = TabularOnlyWrapper(model, mean_test_image, target_output_index)
    
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


def plot_shap_summary(
    tabular_shap_values: np.ndarray,
    test_tabular: torch.Tensor,
    tabular_feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Generate a SHAP summary plot showing global feature importance.
    
    Args:
        tabular_shap_values (np.ndarray): SHAP values for tabular features.
                                         Shape: (num_samples, num_features)
        test_tabular (torch.Tensor): Test tabular data.
                                     Shape: (num_samples, num_features)
        tabular_feature_names (List[str], optional): Names of the tabular features.
        save_path (str, optional): Path to save the plot. If None, uses default name.
        show (bool): Whether to display the plot.
    """
    
    # Convert test_tabular to numpy if needed
    if isinstance(test_tabular, torch.Tensor):
        test_tabular_np = test_tabular.detach().cpu().numpy()
    else:
        test_tabular_np = test_tabular
    
    # Generate default feature names if not provided
    if tabular_feature_names is None:
        num_features = tabular_shap_values.shape[1]
        tabular_feature_names = [f"Feature_{i}" for i in range(num_features)]
    
    # Create summary plot
    plt.figure()
    shap.summary_plot(
        tabular_shap_values,
        test_tabular_np,
        feature_names=tabular_feature_names,
        show=False
    )
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved SHAP summary plot to {save_path}")
    else:
        plt.savefig('shap_summary_plot.png', bbox_inches='tight', dpi=150)
        print("Saved SHAP summary plot to shap_summary_plot.png")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_shap_force(
    explainer: shap.Explainer,
    tabular_shap_values: np.ndarray,
    test_tabular: torch.Tensor,
    sample_index: int = 0,
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
        sample_index (int): Index of the sample to explain.
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
        tabular_shap_values[sample_index, :],
        test_tabular_np[sample_index, :],
        feature_names=tabular_feature_names,
        show=show,
        matplotlib=True
    )
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved SHAP force plot to {save_path}")
    else:
        default_path = f'shap_force_plot_sample_{sample_index}.png'
        plt.savefig(default_path, bbox_inches='tight', dpi=150)
        print(f"Saved SHAP force plot to {default_path}")
    
    if not show:
        plt.close()


def plot_shap_waterfall(
    explainer: shap.Explainer,
    tabular_shap_values: np.ndarray,
    test_tabular: torch.Tensor,
    sample_index: int = 0,
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
        sample_index (int): Index of the sample to explain.
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
        values=tabular_shap_values[sample_index, :],
        base_values=expected_value,
        data=test_tabular_np[sample_index, :],
        feature_names=tabular_feature_names
    )
    
    # Create waterfall plot
    plt.figure()
    shap.waterfall_plot(explanation, show=False)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved SHAP waterfall plot to {save_path}")
    else:
        default_path = f'shap_waterfall_plot_sample_{sample_index}.png'
        plt.savefig(default_path, bbox_inches='tight', dpi=150)
        print(f"Saved SHAP waterfall plot to {default_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
