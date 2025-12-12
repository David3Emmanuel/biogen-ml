import torch
import matplotlib.pyplot as plt
import time
import os
import numpy as np


def test_explain_with_tabular():
    """Test the Tabular Explainability (SHAP)"""
    from explain.tabular import (
        explain_with_shap,
        plot_shap_force,
        plot_shap_waterfall
    )
    from models import FusedModel
    
    print("\n" + "="*60)
    print("TEST 1: Tabular Explainability (SHAP)")
    print("="*60)
    
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
    test_tabular = torch.randn(num_tabular_features)
    
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
        print(f"Generating SHAP waterfall plot for patient ({output_label} risk)...")
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
    
    print("\n" + "="*60)
    print("TEST 2: Image Explainability (Grad-CAM)")
    print("="*60)
    
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


def test_explain_combined():
    """Test combined explainability for both image and tabular data"""
    from explain.tabular import (
        explain_with_shap,
        plot_shap_waterfall
    )
    from explain import explain_with_image
    from models import FusedModel
    
    print("\n" + "="*60)
    print("TEST 3: Combined Image + Tabular Explainability")
    print("="*60)
    
    # Create a sample fused model
    num_tabular_features = 10
    model = FusedModel(num_tabular_features=num_tabular_features)
    model.eval()
    
    # Generate background data for SHAP
    print("Generating background data for SHAP...")
    bg_tabular = torch.randn(100, num_tabular_features)
    
    # Generate test data
    print("Generating test image and tabular data...")
    test_image = torch.randn(1, 3, 224, 224)  # Batch format for image
    test_tabular = torch.randn(num_tabular_features)
    
    # Define feature names for better visualization
    tabular_feature_names = [
        "Age", "BMI", "Blood_Pressure", "Glucose", "Cholesterol",
        "Heart_Rate", "Exercise_Hours", "Sleep_Hours", "Stress_Level", "Diet_Score"
    ]
    
    # Test combined explanation for both output indices
    for target_output_index in [0, 1]:
        output_label = "1-year" if target_output_index == 0 else "3-year"
        print(f"\n{'='*60}")
        print(f"Combined explanations for {output_label} risk prediction...")
        print(f"{'='*60}")
        
        # 1. Image Explainability (Grad-CAM)
        print(f"\n1. Generating Grad-CAM for image data ({output_label} risk)...")
        start_time = time.time()
        
        grayscale_cam, visualization = explain_with_image(
            model=model,
            image_tensor=test_image,
            target_output_index=target_output_index
        )
        
        elapsed = time.time() - start_time
        print(f"✓ Grad-CAM completed in {elapsed:.2f} seconds")
        
        # 2. Tabular Explainability (SHAP)
        print(f"\n2. Generating SHAP for tabular data ({output_label} risk)...")
        start_time = time.time()
        
        tabular_shap_values, explainer = explain_with_shap(
            model=model,
            bg_tabular=bg_tabular,
            image_tensor=test_image[0],  # Remove batch dimension for SHAP
            test_tabular=test_tabular,
            target_output_index=target_output_index,
            tabular_feature_names=tabular_feature_names,
            max_bg_samples=10,
            nsamples=50
        )
        
        elapsed = time.time() - start_time
        print(f"✓ SHAP completed in {elapsed:.2f} seconds")
        
        # 3. Create combined visualization
        print(f"\n3. Creating combined visualization ({output_label} risk)...")
        
        # Generate SHAP waterfall plot separately
        plot_shap_waterfall(
            explainer=explainer,
            tabular_shap_values=tabular_shap_values,
            test_tabular=test_tabular,
            target_output_index=target_output_index,
            tabular_feature_names=tabular_feature_names,
            save_path=f'temp_shap_waterfall_{output_label}.png',
            show=False
        )
        
        # Create combined figure
        fig = plt.figure(figsize=(20, 6))
        
        # Original image
        ax1 = plt.subplot(1, 3, 1)
        original_img = test_image[0].permute(1, 2, 0).detach().cpu().numpy()
        original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
        plt.imshow(original_img)
        plt.title(f'Original Medical Image\n({output_label} Risk)', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # Grad-CAM visualization
        ax2 = plt.subplot(1, 3, 2)
        plt.imshow(visualization)
        plt.title(f'Grad-CAM Explanation\n(Image Contribution)', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # Load and display SHAP waterfall plot
        ax3 = plt.subplot(1, 3, 3)
        if os.path.exists(f'temp_shap_waterfall_{output_label}.png'):
            shap_img = plt.imread(f'temp_shap_waterfall_{output_label}.png')
            plt.imshow(shap_img)
            plt.axis('off')
            plt.title(f'SHAP Explanation\n(Tabular Feature Contributions)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_path = f'combined_explanation_{output_label}_risk.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved combined visualization to {save_path}")
        plt.close()
        
        # Clean up temporary file
        if os.path.exists(f'temp_shap_waterfall_{output_label}.png'):
            os.remove(f'temp_shap_waterfall_{output_label}.png')
        
        # 4. Print model prediction with explanations
        print(f"\n4. Model Prediction Summary ({output_label} risk):")
        with torch.no_grad():
            # Get model prediction
            prediction = model(test_image, test_tabular.unsqueeze(0))
            risk_score = prediction[0, target_output_index].item()
            print(f"   Risk Score: {risk_score:.4f}")
            print(f"   Risk Probability: {torch.sigmoid(prediction[0, target_output_index]).item()*100:.2f}%")
        
        # Top contributing tabular features
        shap_abs = np.abs(tabular_shap_values[0])  # Get first sample
        top_k = 3
        top_indices = np.argsort(shap_abs)[::-1][:top_k]
        
        print(f"\n   Top {top_k} Contributing Tabular Features:")
        for i, idx in enumerate(top_indices, 1):
            feature_name = tabular_feature_names[idx]
            shap_val = tabular_shap_values[0, idx]  # Access from 2D array
            feature_val = test_tabular[idx].item()
            impact = "increases" if shap_val > 0 else "decreases"
            print(f"   {i}. {feature_name}: {feature_val:.3f} (SHAP: {shap_val:+.4f}, {impact} risk)")
        
    print(f"\n{'='*60}")
    print("Combined explainability test completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_explain_with_tabular()
    test_explain_with_image()
    test_explain_combined()
