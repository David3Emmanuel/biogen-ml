# Biogen ML API

This API serves the multimodal risk prediction models using FastAPI.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the API

Run the API server:

```bash
python api/main.py
```

Or using uvicorn directly:

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`

## Endpoints

### POST /predict
Make predictions with image and tabular data.

- **Parameters**:
  - `image`: Image file (JPEG/PNG)
  - `tabular`: JSON string of tabular features (list of floats, length must match `NUM_TABULAR_FEATURES`)

- **Response**: JSON with predictions and logits

### POST /explain
Generate Grad-CAM explanation for the image.

- **Parameters**:
  - `image`: Image file (JPEG/PNG)
  - `target_output`: Integer (0 or 1) for which output to explain

- **Response**: JSON with grayscale CAM and base64-encoded visualization

### POST /train
Perform a single training step on the model.

- **Parameters**:
  - `image`: Image file (JPEG/PNG)
  - `tabular`: JSON string of tabular features (list of floats)
  - `targets`: JSON string of target labels (list of 2 floats, e.g., [0.8, 0.2])

- **Response**: JSON with loss, predictions, and targets

### POST /save_model
Save the current model state.

- **Parameters**:
  - `filepath`: String path to save the model (default: "model_checkpoint.pth")

- **Response**: Success message

### POST /load_model
Load a saved model state.

- **Parameters**:
  - `filepath`: String path to the saved model file (default: "model_checkpoint.pth")

- **Response**: Success message

### GET /health
Health check endpoint.

## Configuration

Currently, `NUM_TABULAR_FEATURES` is hardcoded to 10. Update this in `api/main.py` to match your model's input dimensions.

## Notes

- The model is loaded globally on first request, with Adam optimizer and BCEWithLogitsLoss initialized for training.
- Images are resized to 224x224 and normalized for ResNet-18.
- Tabular data expects a JSON array of floats matching `NUM_TABULAR_FEATURES` (currently 10).
- Training performs a single gradient descent step per request.
- Model can be saved/loaded for persistence.
- In production, consider using a proper model registry, database for training data, authentication, and async handling.