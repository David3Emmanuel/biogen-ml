import pytest
from fastapi.testclient import TestClient
import torch
import json
import io
from PIL import Image
import numpy as np

from api.main import app

client = TestClient(app)

def create_test_image():
    """Create a simple test image."""
    # Create a 224x224 RGB image
    img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes

def create_test_tabular():
    """Create test tabular data."""
    return [0.1] * 10  # 10 features as per NUM_TABULAR_FEATURES

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_success():
    """Test successful prediction."""
    image_bytes = create_test_image()
    tabular_data = create_test_tabular()

    response = client.post(
        "/predict",
        files={"image": ("test.png", image_bytes, "image/png")},
        data={"tabular": json.dumps(tabular_data)}
    )

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "logits" in data
    assert "1_year_risk" in data["predictions"]
    assert "3_year_risk" in data["predictions"]
    assert len(data["logits"]) == 2

def test_predict_missing_image():
    """Test prediction with missing image."""
    tabular_data = create_test_tabular()

    response = client.post(
        "/predict",
        data={"tabular": json.dumps(tabular_data)}
    )

    assert response.status_code == 422  # FastAPI validation error

def test_predict_missing_tabular():
    """Test prediction with missing tabular data."""
    image_bytes = create_test_image()

    response = client.post(
        "/predict",
        files={"image": ("test.png", image_bytes, "image/png")}
    )

    assert response.status_code == 400
    assert "Tabular data is required" in response.json()["detail"]

def test_predict_invalid_tabular():
    """Test prediction with invalid tabular data."""
    image_bytes = create_test_image()

    response = client.post(
        "/predict",
        files={"image": ("test.png", image_bytes, "image/png")},
        data={"tabular": "invalid json"}
    )

    assert response.status_code == 400
    assert "Invalid JSON" in response.json()["detail"]

def test_predict_wrong_tabular_length():
    """Test prediction with wrong tabular data length."""
    image_bytes = create_test_image()
    tabular_data = [0.1] * 5  # Wrong length

    response = client.post(
        "/predict",
        files={"image": ("test.png", image_bytes, "image/png")},
        data={"tabular": json.dumps(tabular_data)}
    )

    assert response.status_code == 400
    assert "length" in response.json()["detail"]

def test_explain_success():
    """Test successful explanation."""
    image_bytes = create_test_image()

    response = client.post(
        "/explain",
        files={"image": ("test.png", image_bytes, "image/png")},
        data={"target_output": 0}
    )

    assert response.status_code == 200
    data = response.json()
    assert "grayscale_cam" in data
    assert "visualization_base64" in data
    assert "target_output" in data
    assert data["target_output"] == 0

def test_explain_invalid_target():
    """Test explanation with invalid target output."""
    image_bytes = create_test_image()

    response = client.post(
        "/explain",
        files={"image": ("test.png", image_bytes, "image/png")},
        data={"target_output": 2}  # Invalid
    )

    assert response.status_code == 400
    assert "must be 0 or 1" in response.json()["detail"]

if __name__ == "__main__":
    pytest.main([__file__])