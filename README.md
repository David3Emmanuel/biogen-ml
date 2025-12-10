# Biogen ML

A multimodal machine learning project for risk prediction using image and tabular data, with explainability features.

## Overview

This project implements a fused neural network that combines:
- **Image data**: Processed through a pre-trained ResNet-18 CNN
- **Tabular data**: Processed through a custom MLP
- **Fusion**: Combined predictions for 1-year and 3-year risk assessment

Includes Grad-CAM explainability for image inputs and a FastAPI-based REST API for serving predictions.

## Project Structure

```
biogen-ml/
├── models/          # Neural network models
│   ├── image.py     # Image CNN (ResNet-18 based)
│   ├── tabular.py   # Tabular MLP
│   ├── fused.py     # Fused multimodal model
│   └── __init__.py
├── explain/         # Explainability tools
│   ├── gradcam.py   # Grad-CAM utilities
│   ├── image.py     # Image explanation functions
│   └── __init__.py
├── api/             # FastAPI REST API
│   ├── main.py      # API application
│   └── __init__.py
├── test_*.py        # Test files
├── requirements.txt # Python dependencies
└── README_API.md    # API-specific documentation
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Tests

```bash
# Test all components
python test_models.py
python test_explain.py
python test_api.py
```

### Running the API

```bash
# Start the FastAPI server
python api/main.py
```

Access the API documentation at `http://localhost:8000/docs`

### Using the Models Directly

```python
from models import FusedModel
import torch

# Initialize model
model = FusedModel(num_tabular_features=10)

# Create sample inputs
image_tensor = torch.randn(1, 3, 224, 224)
tabular_tensor = torch.randn(1, 10)

# Get predictions
outputs = model(image_tensor, tabular_tensor)
# outputs: [1-year_risk_logit, 3-year_risk_logit]
```

### Generating Explanations

```python
from explain import explain_with_image

# Generate Grad-CAM heatmap
grayscale_cam, visualization = explain_with_image(
    model=model,
    image_tensor=image_tensor,
    target_output_index=0  # 0 for 1-year, 1 for 3-year
)
```

## API Endpoints

- `POST /predict`: Make predictions with image + tabular data
- `POST /explain`: Generate Grad-CAM explanations
- `GET /health`: Health check

See `README_API.md` for detailed API documentation.

## Dependencies

- PyTorch & TorchVision
- FastAPI & Uvicorn
- pytorch-grad-cam
- PIL (Pillow)
- matplotlib
- numpy

## Model Configuration

- **Image input**: 224x224 RGB images (ResNet-18 standard)
- **Tabular features**: Configurable number (default 10 in examples)
- **Outputs**: 2 logits (1-year and 3-year risk)

## Development

- Models are defined in `models/`
- Tests ensure functionality
- API serves the models for production use
- Grad-CAM provides interpretability

## License

[Add license information here]