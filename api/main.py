from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from PIL import Image
import io
import numpy as np
from typing import List, Dict, Any
import base64
import json
import os

from models import FusedModel
from explain import explain_with_image

app = FastAPI(title="Biogen ML API", description="API for multimodal risk prediction models")

# Global model instance (in production, consider proper model management)
MODEL = None
OPTIMIZER = None
LOSS_FN = None
NUM_TABULAR_FEATURES = 10  # This should match your training data; make configurable

def load_model():
    global MODEL, OPTIMIZER, LOSS_FN
    if MODEL is None:
        MODEL = FusedModel(num_tabular_features=NUM_TABULAR_FEATURES)
        MODEL.train()  # Set to training mode initially
        # Initialize optimizer and loss
        OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=1e-4)
        LOSS_FN = nn.BCEWithLogitsLoss()  # Binary cross-entropy for multi-label
    return MODEL

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess uploaded image to tensor expected by the model."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        # Resize to 224x224 as expected by ResNet-18
        image = image.resize((224, 224))
        # Convert to tensor and normalize (basic normalization, adjust if needed)
        transform = torch.nn.Sequential(
            torch.nn.ToTensor(),
            torch.nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

@app.post("/train")
async def train(
    image: UploadFile = File(...),
    tabular: str = None,  # JSON string of tabular features
    targets: str = None   # JSON string of target labels [1_year_risk, 3_year_risk]
) -> Dict[str, Any]:
    """
    Perform a single training step on the model.

    - **image**: Upload an image file (JPEG/PNG)
    - **tabular**: JSON string of tabular features as a list of floats
    - **targets**: JSON string of target labels as a list of floats [0 or 1]
    """
    try:
        # Load model
        model = load_model()
        model.train()  # Ensure training mode

        # Process image
        image_bytes = await image.read()
        image_tensor = preprocess_image(image_bytes)

        # Process tabular data
        if tabular is None or targets is None:
            raise HTTPException(status_code=400, detail="Tabular data and targets are required")
        try:
            tabular_data = json.loads(tabular)
            if not isinstance(tabular_data, list) or len(tabular_data) != NUM_TABULAR_FEATURES:
                raise ValueError("Tabular data must be a list of floats with correct length")
            tabular_tensor = torch.tensor(tabular_data, dtype=torch.float32).unsqueeze(0)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON for tabular data")

        # Process targets
        try:
            target_data = json.loads(targets)
            if not isinstance(target_data, list) or len(target_data) != 2:
                raise ValueError("Targets must be a list of 2 floats")
            target_tensor = torch.tensor(target_data, dtype=torch.float32).unsqueeze(0)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON for targets")

        # Training step
        OPTIMIZER.zero_grad()
        outputs = model(image_tensor, tabular_tensor)
        loss = LOSS_FN(outputs, target_tensor)
        loss.backward()
        OPTIMIZER.step()

        return {
            "loss": loss.item(),
            "predictions": torch.sigmoid(outputs).squeeze().tolist(),
            "targets": target_data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    tabular: str = None  # JSON string of tabular features
) -> Dict[str, Any]:
    """
    Make predictions using the fused model.

    - **image**: Upload an image file (JPEG/PNG)
    - **tabular**: JSON string of tabular features as a list of floats
    """
    try:
        # Load model
        model = load_model()
        model.eval()  # Set to evaluation mode for inference

        # Process image
        image_bytes = await image.read()
        image_tensor = preprocess_image(image_bytes)

        # Process tabular data
        if tabular is None:
            raise HTTPException(status_code=400, detail="Tabular data is required")
        try:
            tabular_data = json.loads(tabular)
            if not isinstance(tabular_data, list) or len(tabular_data) != NUM_TABULAR_FEATURES:
                raise ValueError("Tabular data must be a list of floats with correct length")
            tabular_tensor = torch.tensor(tabular_data, dtype=torch.float32).unsqueeze(0)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON for tabular data")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid tabular data: {str(e)}")

        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor, tabular_tensor)
            # Apply sigmoid for probabilities (assuming binary classification)
            probabilities = torch.sigmoid(outputs).squeeze().tolist()

        return {
            "predictions": {
                "1_year_risk": probabilities[0],
                "3_year_risk": probabilities[1]
            },
            "logits": outputs.squeeze().tolist()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/explain")
async def explain(
    image: UploadFile = File(...),
    target_output: int = 0  # 0 for 1-year, 1 for 3-year
) -> Dict[str, Any]:
    #Generate Grad-CAM explanation for the image input.
    #- image: Upload an image file (JPEG/PNG)
    #- target_output: Which output to explain (0 or 1)
    try:
        if target_output not in [0, 1]:
            raise HTTPException(status_code=400, detail="target_output must be 0 or 1")

        # Load model
        model = load_model()

        # Process image
        image_bytes = await image.read()
        image_tensor = preprocess_image(image_bytes)

        # Generate explanation
        grayscale_cam, visualization = explain_with_image(
            model=model,
            image_tensor=image_tensor,
            target_output_index=target_output
        )

        # Convert visualization to base64 for JSON response
        vis_pil = Image.fromarray((visualization * 255).astype(np.uint8))
        buffer = io.BytesIO()
        vis_pil.save(buffer, format="PNG")
        vis_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Convert grayscale CAM to list for JSON
        cam_list = grayscale_cam.tolist()

        return {
            "grayscale_cam": cam_list,
            "visualization_base64": vis_base64,
            "target_output": target_output
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.post("/save_model")
async def save_model(filepath: str = "model_checkpoint.pth") -> Dict[str, str]:
    """
    Save the current model state to a file.

    - **filepath**: Path to save the model (default: model_checkpoint.pth)
    """
    try:
        if MODEL is None:
            raise HTTPException(status_code=400, detail="Model not loaded")
        torch.save(MODEL.state_dict(), filepath)
        return {"message": f"Model saved to {filepath}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")

@app.post("/load_model")
async def load_saved_model(filepath: str = "model_checkpoint.pth") -> Dict[str, str]:
    """
    Load a saved model state from a file.

    - **filepath**: Path to the saved model file
    """
    try:
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Model file not found")
        model = load_model()
        model.load_state_dict(torch.load(filepath))
        model.eval()  # Set to eval after loading
        return {"message": f"Model loaded from {filepath}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)