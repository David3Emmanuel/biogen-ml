import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

from models import FusedModel


def explain_with_shap(
    model: FusedModel,
    bg_images: torch.Tensor,
    bg_tabular: torch.Tensor,
    test_images: torch.Tensor,
    test_tabular: torch.Tensor,
    target_output_index: int = 0,
    tabular_feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, shap.Explainer]:
    """
    Explain the fused model's prediction using SHAP for tabular features.
    
    Args:
        model (FusedModel): The fused model to explain.
        bg_images (torch.Tensor): Background image data for SHAP (e.g., 100 samples).
                                  Shape: (num_bg_samples, 3, 224, 224)
        bg_tabular (torch.Tensor): Background tabular data for SHAP.
                                   Shape: (num_bg_samples, num_features)
        test_images (torch.Tensor): Test image data to explain.
                                    Shape: (num_test_samples, 3, 224, 224)
        test_tabular (torch.Tensor): Test tabular data to explain.
                                     Shape: (num_test_samples, num_features)
        target_output_index (int): The index of the output neuron to explain (0 for 1-year, 1 for 3-year).
        tabular_feature_names (List[str], optional): Names of the tabular features for plotting.
    
    Returns:
        tabular_shap_values (np.ndarray): SHAP values for the tabular features.
                                         Shape: (num_test_samples, num_features)
        explainer (shap.Explainer): The SHAP explainer object for further use.
    """
    
    assert target_output_index in [0, 1], "target_output_index must be 0 or 1"
    
    model.eval()
    
    # Ensure all tensors are on the same device as the model
    device = next(model.parameters()).device
    bg_images = bg_images.to(device)
    bg_tabular = bg_tabular.to(device)
    test_images = test_images.to(device)
    test_tabular = test_tabular.to(device)
    
    # Initialize GradientExplainer with a LIST of background data
    explainer = shap.GradientExplainer(model, [bg_images, bg_tabular])
    
    # Calculate SHAP values for the test data
    # This returns a list of arrays, one for each output dimension
    shap_values_list = explainer.shap_values([test_images, test_tabular])
    
    # Get explanations for the specified output (0: 1-year, 1: 3-year)
    # This returns a list of 2 (for our 2 inputs: image, tabular)
    if isinstance(shap_values_list, list) and len(shap_values_list) > target_output_index:
        shap_values_for_output = shap_values_list[target_output_index]
    else:
        shap_values_for_output = shap_values_list
    
    # Get the SHAP values for the *tabular data only* (input index 1)
    # shap_values_for_output is a list: [image_shap, tabular_shap]
    if isinstance(shap_values_for_output, list) and len(shap_values_for_output) > 1:
        tabular_shap_values = shap_values_for_output[1]
    else:
        # Fallback if structure is different
        tabular_shap_values = shap_values_for_output
    
    # Convert to numpy if it's still a tensor
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
