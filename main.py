"""
Main script for running inference and explanation on a single image.

Usage:
    python main.py <image_path> [--cancer-type {breast,cervical}] [--output-dir <path>]

Example:
    python main.py sample_image.jpg --cancer-type breast --output-dir results/
"""

import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from models import FusedModel
from explain import (
    explain_both_outputs,
    create_shap_waterfall_plots,
    summarize_prediction,
    visualize_combined_explanation
)


# Default tabular feature values (synthesized)
DEFAULT_TABULAR_FEATURES = {
    'Age': 45.0,              # Middle age (years)
    'BMI': 25.0,              # Normal BMI
    'Blood_Pressure': 120.0,  # Normal systolic BP (mmHg)
    'Glucose': 100.0,         # Normal fasting glucose (mg/dL)
    'Cholesterol': 200.0,     # Borderline cholesterol (mg/dL)
    'Heart_Rate': 72.0,       # Normal resting heart rate (bpm)
    'Exercise_Hours': 3.0,    # Moderate exercise (hours/week)
    'Sleep_Hours': 7.0,       # Recommended sleep (hours/day)
    'Stress_Level': 5.0,      # Moderate stress (1-10 scale)
    'Diet_Score': 6.0         # Moderate diet quality (1-10 scale)
}


def load_and_preprocess_image(image_path: str) -> torch.Tensor:
    """
    Load and preprocess an image for model inference.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image tensor (1, 3, 224, 224)
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Resize to expected dimensions
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    
    # Convert to tensor and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Normalize using ImageNet stats (common for ResNet models)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Convert to tensor (H, W, C) -> (C, H, W)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    
    # Add batch dimension (C, H, W) -> (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor


def synthesize_tabular_data(
    feature_names: list,
    custom_values: dict = None  # type: ignore
) -> torch.Tensor:
    """
    Synthesize tabular data for inference when real data is not available.
    
    Args:
        feature_names: List of feature names
        custom_values: Optional dict of feature_name: value to override defaults
        
    Returns:
        Tensor of synthesized tabular features
    """
    values = []
    for name in feature_names:
        if custom_values and name in custom_values:
            values.append(custom_values[name])
        elif name in DEFAULT_TABULAR_FEATURES:
            values.append(DEFAULT_TABULAR_FEATURES[name])
        else:
            # Unknown feature: use neutral value
            values.append(5.0)
    
    return torch.tensor(values, dtype=torch.float32)


def generate_background_data(
    num_tabular_features: int,
    num_samples: int = 100
) -> torch.Tensor:
    """
    Generate synthetic background data for SHAP explanations.
    
    Args:
        num_tabular_features: Number of tabular features
        num_samples: Number of background samples
        
    Returns:
        Background data tensor (num_samples, num_tabular_features)
    """
    # Generate data around default values with some variation
    bg_data = []
    feature_names = list(DEFAULT_TABULAR_FEATURES.keys())[:num_tabular_features]
    
    for _ in range(num_samples):
        sample = []
        for name in feature_names:
            base_value = DEFAULT_TABULAR_FEATURES[name]
            # Add random variation (±20%)
            noise = np.random.randn() * base_value * 0.2
            sample.append(base_value + noise)
        bg_data.append(sample)
    
    return torch.tensor(bg_data, dtype=torch.float32)


def print_prediction_results(
    predictions: dict,
    feature_names: list,
    test_tabular: torch.Tensor
):
    """
    Print formatted prediction results with feature values.
    
    Args:
        predictions: Dict containing prediction results for each output
        feature_names: List of tabular feature names
        test_tabular: The tabular features used for prediction
    """
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    for output_label, summary in predictions.items():
        print(f"\n{output_label.upper()} Risk:")
        print(f"  Risk Score (logit): {summary['risk_score']:.4f}")
        print(f"  Risk Probability: {summary['risk_percentage']:.2f}%")
        print(f"\n  Top Contributing Tabular Features:")
        
        for i, feature in enumerate(summary['top_features'], 1):
            sign = "↑" if feature['impact'] == 'increases' else "↓"
            print(f"    {i}. {feature['feature_name']}: {feature['feature_value']:.3f} "
                  f"(SHAP: {feature['shap_value']:+.4f}) {sign}")
    
    print("\n" + "="*70)
    print("INPUT TABULAR FEATURES")
    print("="*70)
    
    for name, value in zip(feature_names, test_tabular.tolist()):
        print(f"  {name:20s}: {value:.3f}")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Run inference and generate explanations for an image.'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the input image (jpg, png, bmp, jpeg, etc.)'
    )
    parser.add_argument(
        '--cancer-type',
        type=str,
        choices=['breast', 'cervical'],
        default='breast',
        help='Type of cancer model to use (default: breast)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Directory to save explanation visualizations (default: outputs/)'
    )
    parser.add_argument(
        '--max-bg-samples',
        type=int,
        default=10,
        help='Maximum background samples for SHAP (default: 10)'
    )
    parser.add_argument(
        '--nsamples',
        type=int,
        default=50,
        help='Number of SHAP samples for explanation (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Validate image path
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("BIOGEN ML - IMAGE INFERENCE AND EXPLANATION")
    print("="*70)
    print(f"Image: {args.image_path}")
    print(f"Cancer Type: {args.cancer_type}")
    print(f"Output Directory: {args.output_dir}")
    print("="*70 + "\n")
    
    # Configuration
    NUM_TABULAR_FEATURES = 10
    TABULAR_FEATURE_NAMES = [
        "Age", "BMI", "Blood_Pressure", "Glucose", "Cholesterol",
        "Heart_Rate", "Exercise_Hours", "Sleep_Hours", "Stress_Level", "Diet_Score"
    ]
    
    # Load and preprocess image
    print("Loading image...")
    try:
        image_tensor = load_and_preprocess_image(args.image_path)
        print(f"✓ Image loaded and preprocessed: {image_tensor.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    
    # Synthesize tabular data
    print("\nSynthesizing tabular data...")
    test_tabular = synthesize_tabular_data(TABULAR_FEATURE_NAMES)
    bg_tabular = generate_background_data(NUM_TABULAR_FEATURES, num_samples=100)
    print(f"✓ Tabular data synthesized: {test_tabular.shape}")
    print(f"✓ Background data generated: {bg_tabular.shape}")
    
    # Load model
    print(f"\nLoading {args.cancer_type} cancer model...")
    model = FusedModel(
        cancer_type=args.cancer_type,
        num_tabular_features=NUM_TABULAR_FEATURES
    )
    model.eval()
    print("✓ Model loaded successfully")
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        predictions = model(image_tensor, test_tabular.unsqueeze(0))
        one_year_logit = predictions[0, 0].item()
        three_year_logit = predictions[0, 1].item()
        one_year_prob = torch.sigmoid(predictions[0, 0]).item()
        three_year_prob = torch.sigmoid(predictions[0, 1]).item()
    
    print(f"✓ Inference complete")
    print(f"  1-year risk: {one_year_prob:.2%}")
    print(f"  3-year risk: {three_year_prob:.2%}")
    
    # Generate explanations
    print("\n" + "="*70)
    print("GENERATING EXPLANATIONS (This may take a minute...)")
    print("="*70)
    
    explanations = explain_both_outputs(
        model=model,
        image_tensor=image_tensor,
        test_tabular=test_tabular,
        bg_tabular=bg_tabular,
        tabular_feature_names=TABULAR_FEATURE_NAMES,
        max_bg_samples=args.max_bg_samples,
        nsamples=args.nsamples
    )
    
    print("\n✓ Explanations generated successfully")
    
    # Summarize predictions
    prediction_summaries = {}
    for output_label, exp in explanations.items():
        summary = summarize_prediction(
            explanation=exp,
            test_tabular=test_tabular,
            tabular_feature_names=TABULAR_FEATURE_NAMES,
            top_k=5
        )
        prediction_summaries[output_label] = summary
    
    # Print results
    print_prediction_results(
        prediction_summaries,
        TABULAR_FEATURE_NAMES,
        test_tabular
    )
    
    # Save SHAP waterfall plots
    print("\nSaving SHAP waterfall plots...")
    waterfall_paths = create_shap_waterfall_plots(
        explanations=explanations,
        test_tabular=test_tabular,
        tabular_feature_names=TABULAR_FEATURE_NAMES
    )
    
    for output_label, path in waterfall_paths.items():
        new_path = os.path.join(args.output_dir, os.path.basename(path))
        os.rename(path, new_path)
        print(f"  ✓ {output_label}: {new_path}")
    
    # Save combined visualization
    print("\nCreating combined visualization...")
    combined_path = os.path.join(args.output_dir, 'combined_explanation.png')
    visualize_combined_explanation(
        image_tensor=image_tensor,
        explanations=explanations,
        tabular_feature_names=TABULAR_FEATURE_NAMES,
        save_path=combined_path,
        dpi=150
    )
    print(f"  ✓ Combined visualization: {combined_path}")
    
    # Save individual Grad-CAM visualizations
    print("\nSaving individual Grad-CAM visualizations...")
    for output_label, exp in explanations.items():
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        original_img = image_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_img = original_img * std + mean
        original_img = np.clip(original_img, 0, 1)
        plt.imshow(original_img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(exp.grayscale_cam, cmap='jet')
        plt.title(f'Grad-CAM Heatmap\n({output_label})')
        plt.colorbar()
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(exp.visualization)
        plt.title(f'Overlay\n({output_label})')
        plt.axis('off')
        
        plt.tight_layout()
        gradcam_path = os.path.join(args.output_dir, f'gradcam_{output_label}.png')
        plt.savefig(gradcam_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {output_label}: {gradcam_path}")
    
    print("\n" + "="*70)
    print("COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nAll results saved to: {args.output_dir}/")
    print("\nGenerated files:")
    for file in sorted(os.listdir(args.output_dir)):
        print(f"  - {file}")
    print()


if __name__ == "__main__":
    main()
