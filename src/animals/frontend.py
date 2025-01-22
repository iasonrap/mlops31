import os

import pandas as pd
import requests
import streamlit as st
from google.cloud import run_v2

@st.cache_resource  
def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/mlops31/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "animal-classify-app":
            print(f"Found service: {service.name} with URI: {service.uri}")
            return service.uri
    name = os.environ.get("BACKEND", None)
    return name

def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/classify/"
    response = requests.post(predict_url, files={"file": image}, timeout=30)
    if response.status_code == 200:
        return response.json()
    else: 
        print(f"Error: {response.status_code} - {response.text}")
        return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)
    
    animals_classes = {0: "dog", 1: "horse", 2: "elephant", 
                       3: "butterfly",  4: "chicken", 5: "cat", 6: "cow", 7: "sheep", 8: "spider", 9: "squirrel"}

    st.title("Image Classification")
    st.subheader("Classify images of following animals:")
    st.write(f"- {', '.join(animals_classes.values())}")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        result = classify_image(image, backend=backend)

        if result is not None:
            prediction = result["prediction"]
            probabilities = result["probabilities"]

            # show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.write("Prediction:", prediction)

            # make a nice bar chart
            data = {"Class": [animals_classes[i] for i in range(10)], "Probability": probabilities}
            df = pd.DataFrame(data)
            df.set_index("Class", inplace=True)
            st.bar_chart(df, y="Probability")

            # If the probabilities seem unsure
            if max(probabilities) < 0.5:
                st.write("The may be a bit confused about this one...")
            else:
                st.write("The model is confident!")
        else:
            st.write("Failed to get prediction")
        
        # We can insert a button to vote if the prediction was correct or not and send the feedback to the backend

if __name__ == "__main__":
    main()