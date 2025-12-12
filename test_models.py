import torch

def test_tabular_mlp():
    from models import TabularMLP
    
    batch_size = 16
    input_dim = 10
    output_dim = 2  # Now defaults to 2 for late fusion

    input_tensor = torch.randn(batch_size, input_dim)

    model = TabularMLP(input_dim=input_dim, output_dim=output_dim)

    output_tensor = model(input_tensor)

    assert output_tensor.shape == (batch_size, output_dim), "Output shape mismatch"
    print("TabularMLP test passed!")

def test_image_cnn():
    from models import ImageCNN
    
    batch_size = 8
    channels = 3
    height = 224
    width = 224
    output_dim = 2  # Now defaults to 2 for late fusion

    input_tensor = torch.randn(batch_size, channels, height, width)

    model = ImageCNN(output_dim=output_dim)

    output_tensor = model(input_tensor)

    assert output_tensor.shape == (batch_size, output_dim), "Output shape mismatch"
    print("ImageCNN test passed!")

def test_fused_model():
    from models import FusedModel

    batch_size = 4
    tabular_input_dim = 10
    image_channels = 3
    image_height = 224
    image_width = 224

    tabular_input = torch.randn(batch_size, tabular_input_dim)
    image_input = torch.randn(batch_size, image_channels, image_height, image_width)

    fused_model = FusedModel(
        cancer_type='breast',
        num_tabular_features=tabular_input_dim
    )

    output_tensor = fused_model(image_input, tabular_input)

    assert output_tensor.shape == (batch_size, 2), "Output shape mismatch"
    print("FusedModel test passed!")

if __name__ == "__main__":
    test_tabular_mlp()
    test_image_cnn()
    test_fused_model()
