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

### GET /health
Health check endpoint.

## Configuration

Currently, `NUM_TABULAR_FEATURES` is hardcoded to 10. Update this in `api/main.py` to match your model's input dimensions.

## Notes

- The model is loaded globally on first request.
- Images are resized to 224x224 and normalized for ResNet-18.
- In production, consider using a proper model registry and async handling.