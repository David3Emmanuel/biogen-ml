import torch
import matplotlib.pyplot as plt

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
    test_explain_with_image()