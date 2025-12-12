import torch
import matplotlib.pyplot as plt
import time


def test_explain_with_tabular():
    """Test the Tabular Explainability (SHAP)"""
    from explain.tabular import (
        explain_with_shap,
        plot_shap_force,
        plot_shap_waterfall
    )
    from models import FusedModel
    
    # Create a sample fused model
    num_tabular_features = 10
    model = FusedModel(num_tabular_features=num_tabular_features)
    model.eval()
    
    # Generate background data for SHAP (will be subsampled automatically)
    print("Generating background data for SHAP...")
    bg_tabular = torch.randn(100, num_tabular_features)
    
    # Generate test data
    print("Generating test data...")
    test_image = torch.randn(3, 224, 224)
    test_tabular = torch.randn(1, num_tabular_features)
    
    # Define feature names for better visualization
    tabular_feature_names = [
        "Age", "BMI", "Blood_Pressure", "Glucose", "Cholesterol",
        "Heart_Rate", "Exercise_Hours", "Sleep_Hours", "Stress_Level", "Diet_Score"
    ]
    
    # Test explanation for both output indices (1-year and 3-year risk)
    for target_output_index in [0, 1]:
        output_label = "1-year" if target_output_index == 0 else "3-year"
        print(f"\n{'='*60}")
        print(f"SHAP explanations for {output_label} risk prediction...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Calculate SHAP values
        tabular_shap_values, explainer = explain_with_shap(
            model=model,
            bg_tabular=bg_tabular,
            image_tensor=test_image,
            test_tabular=test_tabular,
            target_output_index=target_output_index,
            tabular_feature_names=tabular_feature_names,
            max_bg_samples=10,  # Use only 10 background samples for speed
            nsamples=50  # Use 50 SHAP samples (increase for more accuracy)
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ SHAP computation completed in {elapsed:.2f} seconds")
        print(f"SHAP values shape: {tabular_shap_values.shape}")
        
        
        # Waterfall plot for the first patient
        print(f"Generating SHAP waterfall plot for patient 0 ({output_label} risk)...")
        plot_shap_waterfall(
            explainer=explainer,
            tabular_shap_values=tabular_shap_values,
            test_tabular=test_tabular,
            target_output_index=target_output_index,
            tabular_feature_names=tabular_feature_names,
            save_path=f'shap_waterfall_{output_label}_risk.png',
            show=False
        )
        
    print(f"\n{'='*60}")
    print("All SHAP explanations completed successfully!")
    print(f"{'='*60}")


def test_explain_with_image():
    from explain import explain_with_image
    from models import FusedModel
    
    # Create a sample fused model
    num_tabular_features = 10
    model = FusedModel(num_tabular_features=num_tabular_features)
    model.eval()
    
    # Create a sample image tensor (batch_size=1, channels=3, height=224, width=224)
    # ResNet-18 expects 224x224 images
    image_tensor = torch.randn(1, 3, 224, 224)
    
    # Test explanation for both output indices (1-year and 3-year risk)
    for target_output_index in [0, 1]:
        print(f"\nGenerating Grad-CAM explanation for output {target_output_index}...")
        
        grayscale_cam, visualization = explain_with_image(
            model=model,
            image_tensor=image_tensor,
            target_output_index=target_output_index
        )
        
        print(f"Grayscale CAM shape: {grayscale_cam.shape}")
        print(f"Visualization shape: {visualization.shape}")
        
        # Display the visualization
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        # Convert tensor to numpy and transpose for display
        original_img = image_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
        # Normalize to 0-1 range for display
        original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
        plt.imshow(original_img)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(grayscale_cam, cmap='jet')
        plt.title(f'Grad-CAM Heatmap (Output {target_output_index})')
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(visualization)
        plt.title(f'Overlay Visualization (Output {target_output_index})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'gradcam_output_{target_output_index}.png')
        print(f"Saved visualization to gradcam_output_{target_output_index}.png")
        plt.close()
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
        print("Running explainability tests...")
        print("\n" + "="*60)
        print("TEST 1: Tabular Explainability (SHAP)")
        print("="*60)
        test_explain_with_tabular()
        
        print("\n" + "="*60)
        print("TEST 2: Image Explainability (Grad-CAM)")
        print("="*60)
        test_explain_with_image()
