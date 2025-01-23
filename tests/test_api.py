import pytest
from fastapi.testclient import TestClient
from src.animals.api import app  # Import FastAPI app from api.py
import os

client = TestClient(app)  # TestClient allows us to make requests to the API

def test_hello_world():
    """Test that the / route returns 'Hello from the backend!'"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello from the backend!"}

@pytest.mark.skipif(not os.path.exists("data/raw/raw-img/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg"), reason="Test image not found")
def test_classify_endpoint():
    """Test that the /classify endpoint works by sending a mock image"""
    with open("data/raw/raw-img/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg", "rb") as image_file:
        response = client.post(
            "/classify/",
            files={"file": ("test_image.jpg", image_file, "image/jpeg")}
        )
    assert response.status_code == 200
    assert "prediction" in response.json()  # Assuming the response contains a 'prediction' key

@pytest.mark.skipif(not os.path.exists("data/raw/raw-img/cane"), reason="Test data not found")
def test_post_data_endpoint():
    """Test that the /post_data endpoint uploads data correctly."""
    mock_data = {"key": "value"}
    response = client.post(
        "/post_data",
        json=mock_data
    )
    assert response.status_code == 200
    assert response.json() == {"status": "success"}  # Example response, adjust as needed

def main():
    test_hello_world()
    test_classify_endpoint()
    test_post_data_endpoint()

if __name__ == "__main__":
    main()
