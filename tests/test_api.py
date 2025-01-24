import os
import io

from fastapi.testclient import TestClient
from PIL import Image

from src.animals.api import app  # Import FastAPI app from api.py
from starlette.datastructures import UploadFile, Headers
from torchvision import transforms


client = TestClient(app)  # TestClient allows us to make requests to the API



def test_hello_world():
    """Test that the / route returns 'Hello from the backend!'"""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello from the backend!"}


def test_classify_endpoint():
    """Test the /classify endpoint with a mock file."""
    with TestClient(app) as client:
        # Create a temporary image in memory
        img = Image.new("RGB", (300, 300), color=(255, 0, 0))  # Create a red square image
        image_bytes = io.BytesIO()
        img.save(image_bytes, format="JPEG")
        image_bytes.seek(0)  # Reset the pointer to the start of the file

        # Save the temporary file to disk to match endpoint expectations
        temp_filename = "test_image.jpeg"
        with open(temp_filename, "wb") as f:
            f.write(image_bytes.getvalue())

        # Send the file to the endpoint
        with open(temp_filename, "rb") as f:
            response = client.post(
                "/classify/",
                files={"file": ("test_image.jpeg", f, "image/jpeg")},
            )

        # Clean up temporary file
        os.remove(temp_filename)

        # Validate the response
        assert response.status_code == 200
        response_json = response.json()
        assert "prediction" in response_json
        assert "filename" in response_json
        assert response_json["filename"] == "test_image.jpeg"
    



def main():
    test_hello_world()
    test_classify_endpoint()


if __name__ == "__main__":
    main()
