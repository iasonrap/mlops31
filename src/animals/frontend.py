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
    
    def click_button():
            st.session_state.clicked = True
    
    def unclick_button():
            st.session_state.clicked = False
    
    animals_classes = {0: "dog", 1: "horse", 2: "elephant", 
                       3: "butterfly",  4: "chicken", 5: "cat", 6: "cow", 7: "sheep", 8: "spider", 9: "squirrel"}
    
    if st.session_state.get("clicked") is None:
        st.session_state.clicked = False

    st.title("Animal Classification")
    st.subheader("Classify images of following animals:")
    st.write(f"- {', '.join(animals_classes.values())}")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], on_change=unclick_button)

    if uploaded_file is not None:
        image = uploaded_file.read()
        result = classify_image(image, backend=backend)

        if result is not None:
            prediction = result["prediction"]
            probabilities = result["probabilities"]

            # show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.title(f"It is a {prediction}!")

            # If the probabilities seem unsure
            if max(probabilities) < 0.7:
                st.write("The model may be a bit confused about this one...")
            else:
                st.write("The model is confident!")

            # make a nice bar chart
            data = {"Class": [animals_classes[i] for i in range(10)], "Probability": probabilities}
            df = pd.DataFrame(data)
            df.set_index("Class", inplace=True)
            st.bar_chart(df, y="Probability")

            st.subheader("Provide data for model evaluation:")
            img_input = st.selectbox("What did you upload a picture of?", options=animals_classes.values(), index=0)

            agree_to_submit = st.checkbox("I confirm that I know what animal is in the picture.")

            if st.session_state.clicked:
                st.button("Submit data", disabled=True)
                payload = {"x": img_input, "probabilities": probabilities}
                response = post_to_backend(backend, payload)

                if response and response.ok:
                    st.write("Data submitted successfully, thank you!")
                else:
                    st.write("Failed to submit data.")
 
            elif st.button("Submit data", disabled=not agree_to_submit, on_click=click_button):
                pass
        else:
            st.write("Failed to get prediction")
        
        # We can insert a button to vote if the prediction was correct or not and send the feedback to the backend

def post_to_backend(backend_url: str, payload: dict):
    """Posts data to the backend server."""
    try:
        response = requests.post(f"{backend_url}/post_data", json=payload)
        return response
    except requests.RequestException as e:
        st.error(f"Error posting data to backend: {e}")
        return None


if __name__ == "__main__":
    main()