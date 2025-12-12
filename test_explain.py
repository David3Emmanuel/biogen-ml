import torch
import matplotlib.pyplot as plt
import time
import os
import numpy as np


def test_explain_with_tabular():
    """Test the Tabular Explainability (SHAP) using fused module"""
    from explain import explain_both_outputs, create_shap_waterfall_plots
    from models import FusedModel
    
    print("\n" + "="*60)
    print("TEST 1: Tabular Explainability (SHAP)")
    print("="*60)
    
    # Create a sample fused model
    num_tabular_features = 10
    # CORRECT: cancer_type is provided
    model = FusedModel('breast', num_tabular_features=num_tabular_features)
    model.eval()
    
    # Generate background data for SHAP (will be subsampled automatically)
    print("Generating background data for SHAP...")
    bg_tabular = torch.randn(100, num_tabular_features)
    
    # Generate test data
    print("Generating test data...")
    test_image = torch.randn(1, 3, 224, 224)
    test_tabular = torch.randn(num_tabular_features)
    
    # Define feature names for better visualization
    tabular_feature_names = [
        "Age", "BMI", "Blood_Pressure", "Glucose", "Cholesterol",
        "Heart_Rate", "Exercise_Hours", "Sleep_Hours", "Stress_Level", "Diet_Score"
    ]
    
    print(f"\n{'='*60}")
    print(f"Generating SHAP explanations for both outputs...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Generate explanations for both outputs at once
    explanations = explain_both_outputs(
        model=model,
        image_tensor=test_image,
        test_tabular=test_tabular,
        bg_tabular=bg_tabular,
        tabular_feature_names=tabular_feature_names,
        max_bg_samples=10,  # Use only 10 background samples for speed
        nsamples=50  # Use 50 SHAP samples (increase for more accuracy)
    )
    
    elapsed = time.time() - start_time
    print(f"\n✓ SHAP computation completed in {elapsed:.2f} seconds")
    
    for output_label, exp in explanations.items():
        print(f"  - {output_label}: SHAP values shape {exp.tabular_shap_values.shape}")
    
    # Create waterfall plots
    print(f"\nGenerating SHAP waterfall plots...")
    saved_paths = create_shap_waterfall_plots(
        explanations=explanations,
        test_tabular=test_tabular,
        tabular_feature_names=tabular_feature_names
    )
    
    for output_label, path in saved_paths.items():
        print(f"  ✓ Saved {output_label} waterfall plot to {path}")
        
    print(f"\n{'='*60}")
    print("All SHAP explanations completed successfully!")
    print(f"{'='*60}")


def test_explain_with_image():
    """Test Image Explainability (Grad-CAM) using fused module"""
    from explain import explain_both_outputs
    from models import FusedModel
    
    print("\n" + "="*60)
    print("TEST 2: Image Explainability (Grad-CAM)")
    print("="*60)
    
    # Create a sample fused model
    num_tabular_features = 10
    # FIX: Added 'breast' as the cancer_type argument
    model = FusedModel('breast', num_tabular_features=num_tabular_features)
    model.eval()
    
    # Generate test data
    # Note: requires_grad will be handled inside explain_with_image
    image_tensor = torch.randn(1, 3, 224, 224)
    test_tabular = torch.randn(num_tabular_features)
    bg_tabular = torch.randn(100, num_tabular_features)
    
    print("\nGenerating Grad-CAM explanations for both outputs...")
    
    # Generate explanations for both outputs
    explanations = explain_both_outputs(
        model=model,
        image_tensor=image_tensor,
        test_tabular=test_tabular,
        bg_tabular=bg_tabular,
        max_bg_samples=10,
        nsamples=50
    )
    
    # Create visualizations for each output
    for output_label, exp in explanations.items():
        print(f"\nProcessing {output_label} risk...")
        print(f"  Grayscale CAM shape: {exp.grayscale_cam.shape}")
        print(f"  Visualization shape: {exp.visualization.shape}")
        
        # Display the visualization
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        original_img = image_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
        original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
        plt.imshow(original_img)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(exp.grayscale_cam, cmap='jet')
        plt.title(f'Grad-CAM Heatmap ({output_label})')
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(exp.visualization)
        plt.title(f'Overlay Visualization ({output_label})')
        plt.axis('off')
        
        plt.tight_layout()
        save_path = f'gradcam_{output_label}_risk.png'
        plt.savefig(save_path)
        print(f"  ✓ Saved visualization to {save_path}")
        plt.close()
    
    print("\nTest completed successfully!")


def test_explain_combined():
    """Test combined explainability for both image and tabular data using fused module"""
    from explain import (
        explain_both_outputs,
        create_shap_waterfall_plots,
        summarize_prediction,
        visualize_combined_explanation
    )
    from models import FusedModel
    
    print("\n" + "="*60)
    print("TEST 3: Combined Image + Tabular Explainability")
    print("="*60)
    
    # Create a sample fused model
    num_tabular_features = 10
    # FIX: Added 'breast' as the cancer_type argument
    model = FusedModel('breast', num_tabular_features=num_tabular_features)
    model.eval()
    
    # Generate background data for SHAP
    print("Generating test data...")
    bg_tabular = torch.randn(100, num_tabular_features)
    test_image = torch.randn(1, 3, 224, 224)
    test_tabular = torch.randn(num_tabular_features)
    
    # Define feature names for better visualization
    tabular_feature_names = [
        "Age", "BMI", "Blood_Pressure", "Glucose", "Cholesterol",
        "Heart_Rate", "Exercise_Hours", "Sleep_Hours", "Stress_Level", "Diet_Score"
    ]
    
    print(f"\n{'='*60}")
    print(f"Generating comprehensive explanations for both outputs...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Generate all explanations at once
    explanations = explain_both_outputs(
        model=model,
        image_tensor=test_image,
        test_tabular=test_tabular,
        bg_tabular=bg_tabular,
        tabular_feature_names=tabular_feature_names,
        max_bg_samples=10,
        nsamples=50
    )
    
    elapsed = time.time() - start_time
    print(f"\n✓ All explanations completed in {elapsed:.2f} seconds")
    
    # Print prediction summaries
    for output_label, exp in explanations.items():
        print(f"\n{'='*60}")
        print(f"{output_label.upper()} Risk Prediction Summary")
        print(f"{'='*60}")
        
        summary = summarize_prediction(
            explanation=exp,
            test_tabular=test_tabular,
            tabular_feature_names=tabular_feature_names,
            top_k=3
        )
        
        print(f"Risk Score: {summary['risk_score']:.4f}")
        print(f"Risk Probability: {summary['risk_percentage']:.2f}%")
        print(f"\nTop {len(summary['top_features'])} Contributing Tabular Features:")
        
        for i, feature in enumerate(summary['top_features'], 1):
            print(f"  {i}. {feature['feature_name']}: {feature['feature_value']:.3f} "
                  f"(SHAP: {feature['shap_value']:+.4f}, {feature['impact']} risk)")
    
    # Create SHAP waterfall plots
    print(f"\n{'='*60}")
    print("Creating SHAP waterfall plots...")
    print(f"{'='*60}")
    
    waterfall_paths = create_shap_waterfall_plots(
        explanations=explanations,
        test_tabular=test_tabular,
        tabular_feature_names=tabular_feature_names
    )
    
    for output_label, path in waterfall_paths.items():
        print(f"✓ Saved {output_label} waterfall to {path}")
    
    # Create combined visualization
    print(f"\n{'='*60}")
    print("Creating combined visualization...")
    print(f"{'='*60}")
    
    # Create temporary waterfall plots for visualization
    for output_label, exp in explanations.items():
        from explain.tabular import plot_shap_waterfall
        plot_shap_waterfall(
            explainer=exp.explainer,
            tabular_shap_values=exp.tabular_shap_values,
            test_tabular=test_tabular,
            target_output_index=exp.target_output_index,
            tabular_feature_names=tabular_feature_names,
            save_path=f'temp_shap_waterfall_{output_label}.png',
            show=False
        )
    
    combined_path = visualize_combined_explanation(
        image_tensor=test_image,
        explanations=explanations,
        tabular_feature_names=tabular_feature_names,
        save_path='combined_explanation_both_risks.png',
        dpi=150
    )
    
    print(f"✓ Saved combined visualization to {combined_path}")
    
    print(f"\n{'='*60}")
    print("Combined explainability test completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_explain_with_tabular()
    test_explain_with_image()
    test_explain_combined()