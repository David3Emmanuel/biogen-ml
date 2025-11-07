import torch

def test_tabular_mlp():
    from models.tabular import TabularMLP
    
    batch_size = 16
    input_dim = 10
    output_dim = 5

    input_tensor = torch.randn(batch_size, input_dim)

    model = TabularMLP(input_dim=input_dim, output_dim=output_dim)

    output_tensor = model(input_tensor)

    assert output_tensor.shape == (batch_size, output_dim), "Output shape mismatch"
    print("TabularMLP test passed!")

def test_image_cnn():
    from models.image import ImageCNN
    
    batch_size = 8
    channels = 3
    height = 224
    width = 224
    output_dim = 128

    input_tensor = torch.randn(batch_size, channels, height, width)

    model = ImageCNN(output_dim=output_dim)

    output_tensor = model(input_tensor)

    assert output_tensor.shape == (batch_size, output_dim), "Output shape mismatch"
    print("ImageCNN test passed!")

if __name__ == "__main__":
    test_tabular_mlp()
    test_image_cnn()
