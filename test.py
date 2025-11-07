import torch

from models.tabular import TabularMLP

def test_tabular_mlp():
    batch_size = 16
    input_dim = 10
    output_dim = 5

    input_tensor = torch.randn(batch_size, input_dim)

    model = TabularMLP(input_dim=input_dim, output_dim=output_dim)

    output_tensor = model(input_tensor)

    assert output_tensor.shape == (batch_size, output_dim), "Output shape mismatch"
    print("TabularMLP test passed!")

if __name__ == "__main__":
    test_tabular_mlp()
