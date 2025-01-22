import datetime
import json
from contextlib import asynccontextmanager
from src.animals.model import AnimalModel
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms
import requests
import os
from pydantic import BaseModel
from google.cloud import storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to start and stop the lifespan events of the FastAPI application."""
    global model, transform, animals_classes
    # Load model
    model = AnimalModel("resnet18", 10)
    model.eval()

    url = "https://storage.googleapis.com/31animals/models/AnimalModel.pth"
    r = requests.get(url)

    if r.status_code == 200:
        with open("AnimalModel_gcd.pth", "wb") as f:
            f.write(r.content)
        print("File downloaded successfully to AnimalModel_gcd.pth")
    else:
        print(f"Failed to download file. HTTP Status: {r.status_code}")
        print(r.text)

    model.load_state_dict(torch.load("AnimalModel_gcd.pth", map_location=torch.device('cpu')))

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5177, 0.5003, 0.4126], std=[0.2659, 0.2610, 0.2785]),
        ],
    )

    animals_classes = {0: "dog", 1: "horse", 2: "elephant", 
                       3: "butterfly",  4: "chicken", 5: "cat", 6: "cow", 7: "sheep", 8: "spider", 9: "squirrel"}

    yield

    # Clean up
    del model
    del transform
    del animals_classes
    os.remove("AnimalModel_gcd.pth")


app = FastAPI(lifespan=lifespan)

class DataPayload(BaseModel):
    x: str
    probabilities: list

def predict_image(image_path: str) -> str:
    """Predict image class (or classes) given image path and return the result."""
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
    _, predicted_idx = torch.max(output, 1)
    return output.softmax(dim=-1), animals_classes[predicted_idx.item()]


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the backend!"}


# FastAPI endpoint for image classification
@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    """Classify image endpoint."""
    try:
        contents = await file.read()
        with open(file.filename, "wb") as f:
            f.write(contents)
        probabilities, prediction = predict_image(file.filename)
        return {"filename": file.filename, "prediction": prediction, "probabilities": probabilities.squeeze().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500) from e
    
@app.post("/post_data")
async def post_data(payload: DataPayload):
    """
    Endpoint to receive data from the frontend.
    """
    # Validate and process the received data
    try:
        img_input = payload.x
        img_probabilities = payload.probabilities

        print(f"Received image label: {img_input}, probabilities: {img_probabilities}")

        time_stamp = datetime.datetime.now(tz=datetime.UTC).isoformat()
        data = {'target': img_input, **{animal: prob for animal, prob in zip(animals_classes.values(), img_probabilities)}}

        # Upload the data to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket("gcp_monitoring_animals")
        blob = bucket.blob(f"current/predictions_{time_stamp}.json")
        blob.upload_from_string(json.dumps(data))

        return {"message": "Data received and uploaded successfully!"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing data: {str(e)}")