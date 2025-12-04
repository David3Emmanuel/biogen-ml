import torch.nn as nn

class GradCamModelWrapper(nn.Module):
    """
    A wrapper for the fused model to make it compatible
    with the single-input grad-cam library.
    """
    def __init__(self, fused_model, dummy_tabular_tensor):
        super().__init__()
        self.model = fused_model
        self.dummy_tabular = dummy_tabular_tensor

    def forward(self, image_tensor):
        return self.model(image_tensor, self.dummy_tabular)

class ClassifierOutputTarget(nn.Module):
    """
    A custom target class for grad-cam to explain
    a specific regression output neuron.
    
    Re-used from pytorch_grad_cam.utils.model_targets.ClassifierOutputTarget
    """
    def __init__(self, output_index):
        self.output_index = output_index
    def __call__(self, model_output):
        # model_output shape is (batch_size, 2)
        # This returns the specific logit we want to explain.
        if len(model_output.shape) == 1:
            return model_output[self.output_index]
        return model_output[:, self.output_index]
